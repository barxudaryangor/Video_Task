"""
Unified pipeline for all 24 "state" queries.

Same core mechanism as object_pipeline.py (Grounding DINO detection +
SigLIP crop scoring), reused for every state query regardless of its
sub-type (stationary, open/closed, spatial relation, pose, absence,
counting) -- one consistent technique, not eight bespoke ones.

Two modes, chosen per query:

  - PRESENCE (most queries): detect an anchor noun phrase, crop, and
    score the crop against the query's FULL text (which already
    encodes "is open" / "is stationary" / "near the stairs" / etc).
    The anchor phrase only localizes WHERE to look; the full text is
    what actually gets compared via SigLIP.

  - ABSENCE (5 queries whose text says something is NOT present,
    e.g. "there are no cars in the parking lot"): detect the anchor
    across the whole video, and the query is true during the GAPS
    where it was never detected -- comparing a crop against a
    negated sentence doesn't mean anything, so these skip the SigLIP
    step entirely and work off detector_score alone.

This does not pretend to solve every state query equally well --
spatial relations, pose, and counting queries get the same generic
treatment and will likely score lower than presence/absence queries
that map more directly onto detection. The point is one honest,
uniform attempt across all 24, rather than hand-picking easy ones.
"""

import argparse
from pathlib import Path
from difflib import SequenceMatcher

import cv2
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoModel,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

import sam2_utils
import color_utils
import debug_viz
import tracking_utils


def parse_args():
    # Parsed inside a function, not at module import time: state_pipeline.py
    # is sometimes imported for its functions (extract_anchor_and_mode,
    # process_video) rather than run as a script, and that must not
    # require/consume sys.argv.
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-debug", action="store_true", help="Save debug PNGs + debug_index.csv rows.")
    parser.add_argument("--debug-query", type=str, default=None, help="Restrict the whole run to one query_id (its video only).")
    parser.add_argument("--debug-max-images", type=int, default=20, help="Max debug images saved per query.")
    parser.add_argument(
        "--debug-category",
        type=str,
        default=None,
        help="Override the frame_checks/{category}/ folder name (default: 'state'). "
        "Use this to route a test run's images to a separate folder without touching "
        "images already saved under the default one.",
    )
    return parser.parse_args()

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
QUERIES_PATH = DATA_DIR / "queries.csv"

SIGLIP_MODEL_NAME = "google/siglip2-giant-opt-patch16-384"
DETECTOR_MODEL_NAME = "IDEA-Research/grounding-dino-base"

SAMPLE_INTERVAL_SEC = 0.5
DETECTOR_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
CROP_MARGIN = 0.10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIGLIP_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
DETECTOR_DTYPE = torch.float32

# ============================================================
# Automatic anchor phrase + mode extraction
#
# Earlier versions used a hand-written per-query_id lookup table --
# built by reading each query's text and manually deciding what to
# detect. That does not generalize: it only "works" for these exact
# 24 known strings, and requires a human to read and hand-annotate
# any new query text the same way. This replaces it with a syntactic
# rule applied uniformly to whatever text comes in:
#
#   1. Find the sentence's grammatical subject (nsubj/nsubjpass, or
#      attr for existential "there is/are X" sentences).
#   2. Detect negation structurally -- a "no" determiner attached to
#      the subject ("no cars", "no one") -- rather than a keyword
#      list; "absence" mode means the query is true during the GAPS
#      where that subject is not detected.
#   3. The anchor is the subject's own dependency subtree (this
#      naturally captures noun-internal modifiers like "no cars IN
#      THE PARKING LOT" or "two groups OF PEOPLE", since those attach
#      under the subject in the parse) plus any prepositional phrases
#      attached to the main verb ("standing NEAR A TABLE", "with a
#      phone TO THEIR EAR") -- this is what keeps two similarly-worded
#      queries in the same video ("a person near a table" vs "a
#      person on the stairs") from colliding on an identical anchor.
#   4. "no one" specifically maps to the concrete noun "person" --
#      a fixed fact about English ("no one" = "nobody" = "no person"),
#      not something inferred from having read this specific dataset.
# ============================================================

_NLP = spacy.load("en_core_web_sm")


def extract_anchor_and_mode(query_text):
    doc = _NLP(query_text)
    root = next(t for t in doc if t.dep_ == "ROOT")

    subject = next(
        (t for t in doc if t.dep_ in ("nsubj", "nsubjpass", "attr")),
        None,
    )
    if subject is None:
        return query_text, "presence"

    negated = any(
        child.dep_ == "det" and child.lemma_ == "no"
        for child in subject.children
    ) or (subject.dep_ == "det" and subject.lemma_ == "no")

    subtree = sorted(subject.subtree, key=lambda t: t.i)
    lo, hi = subtree[0].i, subtree[-1].i
    used = set(range(lo, hi + 1))

    if negated and doc[lo].lemma_ == "no":
        lo += 1  # drop the leading negation determiner from the anchor text

    if subject.lemma_ == "one" and subject.i == lo:
        # "no one" / generic "one" -> concrete, detectable noun
        rest = doc[subject.i + 1 : hi + 1].text
        core_np = ("person " + rest).strip() if rest else "person"
    else:
        core_np = doc[lo : hi + 1].text

    extra_parts = []
    for child in root.children:
        if child.dep_ == "prep" and child.i not in used:
            plo, phi = child.left_edge.i, child.right_edge.i
            if any(i in used for i in range(plo, phi + 1)):
                continue
            extra_parts.append(doc[plo : phi + 1].text)
            used.update(range(plo, phi + 1))

    anchor = " ".join([core_np] + extra_parts).strip()
    mode = "absence" if negated else "presence"
    return anchor, mode


MIN_CROP_SCORE = 0.13          # start from the object-category calibration
MAX_GAP_SEC = 5.0
MIN_DURATION_SEC = 0.1
BOUNDARY_PAD_SEC = 0.25
MIN_ABSENCE_DURATION_SEC = 3.0  # ignore trivial gaps in an otherwise-continuous presence


def get_embedding_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    raise TypeError(f"Unexpected output type: {type(output)}")


def _or_nan(value):
    """None -> nan (tracking fields that genuinely don't apply yet)."""
    return value if value is not None else float("nan")


def normalize_text(text):
    return str(text).lower().strip().rstrip(".").strip()


def match_detection_to_query(label, anchor_texts):
    label = normalize_text(label)
    best_i, best_r = None, -1.0
    for i, a in enumerate(anchor_texts):
        r = SequenceMatcher(None, label, normalize_text(a)).ratio()
        if r > best_r:
            best_r, best_i = r, i
    return best_i


def encode_texts(processor, model, texts):
    inputs = processor(text=texts, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.get_text_features(**inputs)
    return F.normalize(get_embedding_tensor(out).float(), p=2, dim=-1)


def encode_images(processor, model, images):
    if not images:
        return None
    inputs = processor(images=images, return_tensors="pt")
    pv = inputs["pixel_values"].to(DEVICE, dtype=SIGLIP_DTYPE)
    with torch.inference_mode():
        out = model.get_image_features(pixel_values=pv)
    return F.normalize(get_embedding_tensor(out).float(), p=2, dim=-1)


def crop_with_margin(image, box):
    width, height = image.size
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    mx, my = bw * CROP_MARGIN, bh * CROP_MARGIN
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(width, int(x1 + mx)), min(height, int(y1 + my))
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1))


def detect_objects(image, anchor_texts, detector_processor, detector_model):
    inputs = detector_processor(images=image, text=[anchor_texts], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = detector_model(**inputs)
    results = detector_processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=DETECTOR_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)],
    )[0]
    boxes = results["boxes"].detach().cpu()
    scores = results["scores"].detach().cpu()
    labels = results.get("text_labels") or results.get("labels")
    return [
        {"box": b.tolist(), "score": float(s.item()), "label": str(l)}
        for b, s, l in zip(boxes, scores, labels)
    ]


def process_video(
    video_id,
    video_queries,
    siglip_processor,
    siglip_model,
    detector_processor,
    detector_model,
    sam2_processor,
    sam2_model,
    debug_policy=None,
    debug_category="state",
):
    query_ids = video_queries["query_id"].tolist()
    query_texts = video_queries["query_text"].tolist()
    anchor_mode = [extract_anchor_and_mode(t) for t in query_texts]
    anchors = [a for a, _ in anchor_mode]
    modes = [m for _, m in anchor_mode]
    # Generic (not per-query) color-word extraction from the query's own
    # full text -- same mechanism as object_pipeline.py.
    query_colors = [color_utils.extract_color_words(t) for t in query_texts]

    # Generic (not per-query) pseudo-tracking: one tracker per query,
    # scoped to this one video. See tracking_utils.py for exactly what
    # this is and is not (NOT real video-object tracking with memory).
    trackers = [tracking_utils.QueryTracker() for _ in range(len(query_ids))]

    full_text_embeddings = encode_texts(siglip_processor, siglip_model, query_texts)

    video_path = DATA_DIR / f"{video_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    frame_index = 0
    next_sample_time = 0.0
    progress = tqdm(total=frame_count, desc=f"{video_id} state")

    while True:
        success, frame = cap.read()
        if not success:
            break

        target_frame = round(next_sample_time * fps)
        if frame_index >= target_frame:
            timestamp = frame_index / fps
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            detections = detect_objects(image, anchors, detector_processor, detector_model)

            best_by_query = {}
            for det in detections:
                qi = match_detection_to_query(det["label"], anchors)
                if qi is None:
                    continue
                if qi not in best_by_query or det["score"] > best_by_query[qi]["score"]:
                    best_by_query[qi] = det

            crop_images, crop_query_idxs = [], []
            for qi, det in best_by_query.items():
                crop = crop_with_margin(image, det["box"])
                if crop is not None:
                    crop_images.append(crop)
                    crop_query_idxs.append(qi)

            crop_embeddings = encode_images(siglip_processor, siglip_model, crop_images)

            # ----------------------------------------
            # SAM2: precise mask for each query's winning box.
            # Additive only -- best_by_query / crop_score above (the old
            # bbox-based evidence) are not touched here.
            # ----------------------------------------

            masked_crop_images, masked_crop_query_idxs = [], []
            sam_bboxes, mask_areas = {}, {}
            frame_masks, masked_crop_by_qi = {}, {}
            color_scores, dominant_colors = {}, {}

            for qi, det in best_by_query.items():
                mask, mask_bbox, mask_area = sam2_utils.predict_mask(
                    image, det["box"], sam2_processor, sam2_model, DEVICE
                )
                sam_bboxes[qi] = mask_bbox
                mask_areas[qi] = mask_area
                frame_masks[qi] = mask

                # Color evidence from pixels INSIDE the mask only.
                color_score, dominant_color = color_utils.masked_color_score(
                    image, mask, query_colors[qi]
                )
                color_scores[qi] = color_score
                dominant_colors[qi] = dominant_color

                masked_crop = sam2_utils.masked_crop_with_margin(image, mask, mask_bbox)
                if masked_crop is not None:
                    masked_crop_images.append(masked_crop)
                    masked_crop_query_idxs.append(qi)
                    masked_crop_by_qi[qi] = masked_crop

            masked_crop_embeddings = encode_images(
                siglip_processor, siglip_model, masked_crop_images
            )

            # ----------------------------------------
            # Pseudo-tracking: stitch this frame's winning box (if any)
            # onto each query's own running tracker. See tracking_utils.py.
            # ----------------------------------------

            frame_track_fields = [
                trackers[qi].update(
                    timestamp,
                    best_by_query[qi]["box"] if qi in best_by_query else None,
                    image.width,
                    image.height,
                )
                for qi in range(len(query_ids))
            ]

            for qi in range(len(query_ids)):
                det_score = best_by_query[qi]["score"] if qi in best_by_query else 0.0
                crop_score = float("nan")
                if qi in crop_query_idxs and crop_embeddings is not None:
                    idx = crop_query_idxs.index(qi)
                    crop_score = float((crop_embeddings[idx] @ full_text_embeddings[qi]).item())

                box = best_by_query[qi]["box"] if qi in best_by_query else (np.nan,) * 4

                sam_box = sam_bboxes.get(qi)
                if sam_box is None:
                    sam_box = (np.nan,) * 4
                mask_area = mask_areas.get(qi, 0)

                masked_crop_score = float("nan")
                if qi in masked_crop_query_idxs and masked_crop_embeddings is not None:
                    idx = masked_crop_query_idxs.index(qi)
                    masked_crop_score = float(
                        (masked_crop_embeddings[idx] @ full_text_embeddings[qi]).item()
                    )

                color_score = color_scores.get(qi)
                expected_color = ",".join(query_colors[qi]) if query_colors[qi] else float("nan")

                track_fields = frame_track_fields[qi]

                rows.append({
                    "video_id": video_id,
                    "query_id": query_ids[qi],
                    "query_text": query_texts[qi],
                    "automatic_anchor": anchors[qi],
                    "mode": modes[qi],
                    "time": timestamp,
                    "detector_score": det_score,
                    "crop_score": crop_score,
                    "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                    "mask_area": mask_area,
                    "sam_x0": sam_box[0], "sam_y0": sam_box[1],
                    "sam_x1": sam_box[2], "sam_y1": sam_box[3],
                    "masked_crop_score": masked_crop_score,
                    "expected_color": expected_color,
                    "color_score": color_score if color_score is not None else float("nan"),
                    # ---- Pseudo-tracking evidence (additive, new) ----
                    # See tracking_utils.py: consecutive independent
                    # per-frame detections stitched into tracks, NOT
                    # real video-object tracking with memory.
                    "track_id": _or_nan(track_fields["track_id"]),
                    "center_x_norm": _or_nan(track_fields["center_x_norm"]),
                    "center_y_norm": _or_nan(track_fields["center_y_norm"]),
                    "width": _or_nan(track_fields["width"]),
                    "height": _or_nan(track_fields["height"]),
                    "area": _or_nan(track_fields["area"]),
                    "dx": _or_nan(track_fields["dx"]),
                    "dy": _or_nan(track_fields["dy"]),
                    "speed": _or_nan(track_fields["speed"]),
                    "delta_area": _or_nan(track_fields["delta_area"]),
                    "track_stability": _or_nan(track_fields["track_stability"]),
                })

                # ----------------------------------------
                # Debug images (opt-in). Every value here already
                # existed above -- this only decides whether to render it.
                # ----------------------------------------

                if debug_policy is not None and qi in best_by_query:
                    bbox_score_val = crop_score if np.isfinite(crop_score) else None
                    masked_score_val = masked_crop_score if np.isfinite(masked_crop_score) else None

                    slots = debug_policy.slots_for(
                        query_ids[qi], det_score, masked_score_val, bbox_score_val, timestamp
                    )
                    if slots:
                        bbox_crop_img = crop_with_margin(image, best_by_query[qi]["box"])
                        fields = {
                            "full_query": query_texts[qi],
                            "automatic_anchor": anchors[qi],
                            "detector_prompt": anchors[qi],
                            "detector_score": det_score,
                            "sam_bbox": sam_box,
                            "sam_mask_area": int(mask_area),
                            "siglip_bbox_score": bbox_score_val,
                            "siglip_masked_score": masked_score_val,
                            "expected_color": (
                                ",".join(query_colors[qi]) if query_colors[qi] else None
                            ),
                            "color_score": color_scores.get(qi),
                            "track_id": track_fields["track_id"],
                            "center_x_norm": track_fields["center_x_norm"],
                            "center_y_norm": track_fields["center_y_norm"],
                            "dx": track_fields["dx"],
                            "dy": track_fields["dy"],
                            "speed": track_fields["speed"],
                        }
                        debug_viz.save_detection_debug(
                            category=debug_category,
                            query_id=query_ids[qi],
                            video_id=video_id,
                            timestamp=timestamp,
                            frame_idx=frame_index,
                            detection_id=1,
                            image=image,
                            box=best_by_query[qi]["box"],
                            mask=frame_masks.get(qi),
                            bbox_crop=bbox_crop_img,
                            masked_crop=masked_crop_by_qi.get(qi),
                            fields=fields,
                        )

            next_sample_time += SAMPLE_INTERVAL_SEC

        frame_index += 1
        progress.update(1)

    progress.close()
    cap.release()
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    queries = pd.read_csv(QUERIES_PATH)
    state_queries = queries[queries.query_type == "state"].copy()

    if args.debug_query:
        # Restrict to that one query's own video -- no reason to touch
        # the other 7 videos just to debug a single query.
        state_queries = state_queries[state_queries.query_id == args.debug_query].copy()
        if state_queries.empty:
            raise SystemExit(f"--debug-query {args.debug_query!r} is not a state query_id")

    if args.save_debug:
        print(f"Debug mode ON -- max_images_per_query={args.debug_max_images}, debug_query={args.debug_query!r}")

    print("Loading SigLIP...")
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, dtype=SIGLIP_DTYPE).to(DEVICE).eval()

    print("Loading Grounding DINO...")
    detector_processor = AutoProcessor.from_pretrained(DETECTOR_MODEL_NAME)
    detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR_MODEL_NAME, dtype=DETECTOR_DTYPE
    ).to(DEVICE).eval()

    print(f"Loading SAM2 ({sam2_utils.SAM2_MODEL_NAME})...")
    sam2_processor, sam2_model = sam2_utils.load_sam2(DEVICE)

    OUTPUT_DIR.mkdir(exist_ok=True)
    all_dfs = []

    for video_id in sorted(state_queries.video_id.unique()):
        video_queries = state_queries[state_queries.video_id == video_id].reset_index(drop=True)
        print(f"\n=== {video_id} ===")
        for _, r in video_queries.iterrows():
            anchor, mode = extract_anchor_and_mode(r.query_text)
            print(f"  {r.query_id} [{mode}] anchor={anchor!r} | {r.query_text}")

        debug_policy = None
        if args.save_debug:
            _probe = cv2.VideoCapture(str(DATA_DIR / f"{video_id}.mp4"))
            _fps = _probe.get(cv2.CAP_PROP_FPS)
            _frame_count = _probe.get(cv2.CAP_PROP_FRAME_COUNT)
            _probe.release()
            video_duration = _frame_count / _fps if _fps else None
            debug_policy = debug_viz.DebugPolicy(
                max_images=args.debug_max_images, video_duration=video_duration
            )

        df = process_video(
            video_id, video_queries,
            siglip_processor, siglip_model,
            detector_processor, detector_model,
            sam2_processor, sam2_model,
            debug_policy=debug_policy,
            debug_category=args.debug_category or "state",
        )
        all_dfs.append(df)
        torch.cuda.empty_cache()

    result_df = pd.concat(all_dfs, ignore_index=True)

    # Same reasoning as object_pipeline.py: a --debug-query run only
    # covers one query and must never overwrite the full 24-query CSV.
    if args.debug_query:
        out_path = OUTPUT_DIR / f"state_frame_scores_debug_{args.debug_query}.csv"
    else:
        out_path = OUTPUT_DIR / "state_frame_scores.csv"

    result_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(result_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
