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


# ============================================================
# DEBUG CLI FLAGS
#
# Off by default -- a full 8-video run should never silently start
# dumping images. See debug_viz.DebugPolicy for the (generic, not
# per-query-id) selection of which frames are worth saving.
# ============================================================

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--save-debug", action="store_true", help="Save debug PNGs + debug_index.csv rows.")
_arg_parser.add_argument("--debug-query", type=str, default=None, help="Restrict the whole run to one query_id (its video only).")
_arg_parser.add_argument("--debug-max-images", type=int, default=20, help="Max debug images saved per query.")
_arg_parser.add_argument(
    "--debug-category",
    type=str,
    default=None,
    help="Override the frame_checks/{category}/ folder name (default: 'object'). "
    "Use this to route a test run's images to a separate folder without touching "
    "images already saved under the default one.",
)
DEBUG_ARGS = _arg_parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

QUERIES_PATH = DATA_DIR / "queries.csv"


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

SIGLIP_MODEL_NAME = "google/siglip2-giant-opt-patch16-384"

DETECTOR_MODEL_NAME = "IDEA-Research/grounding-dino-base"


# ------------------------------------------------------------
# Video sampling
# ------------------------------------------------------------

SAMPLE_INTERVAL_SEC = 0.5


# ------------------------------------------------------------
# Grounding DINO thresholds
# ------------------------------------------------------------

DETECTOR_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20


# ------------------------------------------------------------
# Crop settings
# ------------------------------------------------------------

# Slightly enlarge the detector box.
#
# Example:
#
# detector finds only torso,
# but query = "man with a red backpack"
#
# Margin helps include backpack/clothing.
CROP_MARGIN = 0.10


# ------------------------------------------------------------
# Fusion weights
#
# No global (whole-frame) SigLIP signal here on purpose: for an
# "object" query the target is usually a small part of the frame, so
# a whole-frame embedding is dominated by background regardless of
# whether the object is present. It fails for the same underlying
# reason (small object) as the detector missing it -- it is not an
# independent fallback, so it cannot rescue the cases detector+crop
# miss, and including it only adds noise on frames where nothing was
# ever detected (its z-score still finds a "relative peak" in pure
# background drift).
# ------------------------------------------------------------

DETECTOR_WEIGHT = 0.375
CROP_WEIGHT = 0.625


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGLIP_DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

DETECTOR_DTYPE = torch.float32


# ============================================================
# HELPERS
# ============================================================

def get_embedding_tensor(output):
    """
    Transformers 5.x may return BaseModelOutputWithPooling.

    Other versions may return Tensor directly.
    """

    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    raise TypeError(
        f"Unexpected output type: {type(output)}"
    )


def _or_nan(value):
    """None -> np.nan (for tracking fields that genuinely don't apply
    yet), otherwise pass the value through unchanged."""
    return value if value is not None else np.nan


def normalize_text(text):
    """
    Used only for matching Grounding DINO returned labels
    back to the original query.
    """

    return (
        str(text)
        .lower()
        .strip()
        .rstrip(".")
        .strip()
    )


def match_detection_to_query(
    detected_label,
    query_texts,
):
    """
    Grounding DINO normally returns the text label that
    generated the detection.

    We still use fuzzy matching because wording/punctuation
    may differ slightly.
    """

    detected_label = normalize_text(
        detected_label
    )

    best_index = None
    best_ratio = -1.0

    for index, query in enumerate(query_texts):

        query_normalized = normalize_text(
            query
        )

        ratio = SequenceMatcher(
            None,
            detected_label,
            query_normalized,
        ).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_index = index

    return best_index


# ============================================================
# AUTOMATIC, PER-VIDEO DETECTOR-PROMPT SHORTENING
#
# Grounding DINO's phrase-grounding is measurably worse on long,
# compound descriptive text than on a short concrete noun phrase --
# the same lesson state_pipeline.py already learned for its own anchor
# extraction. Object queries previously had no shortening step at all:
# the full query_text was always sent as the detector prompt. Two
# concrete, verified failures on the real data:
#
#   q030 "a man wearing a blue jacket and purple pants" -> 0/432
#   sampled frames produced ANY detection at all. Not a threshold
#   problem -- the detector never proposed a box for this text.
#
#   q046 "a person wearing a red cap" -> 5/516 frames detected, and
#   even those never clear crop_score's own MIN_CROP_SCORE gate in
#   build_submission.py -- a red cap is a small accessory, and
#   Grounding DINO's box for the whole compound sentence doesn't
#   reliably localize it.
#
# This shortens ONLY the pattern behind both failures: a head noun
# followed by a "wearing/carrying/holding"-style participial clause
# (spaCy "acl") with its own direct object. It tries, per query, in
# order:
#
#   1. the bare head noun phrase ("a man", "a person") -- the detector
#      only has to find the whole person, not a specific small
#      accessory, which maximizes recall. SigLIP crop-scoring below
#      still always runs against the FULL original query_text
#      (encode_texts(query_texts), unchanged) -- shortening the
#      detector prompt never throws away the attribute information
#      needed to actually verify a match, only what's needed to first
#      find the person/object in the frame.
#   2. "<head> with <object>" using only the acl's own direct object,
#      NEVER a second "and X" conjunct (that conjunction is exactly
#      what breaks q030) -- shorter than the original but keeps enough
#      to tell two similarly-worded queries in the same video apart.
#   3. the original query_text, unchanged, if even (2) collides.
#
# Two queries in the same video landing on an IDENTICAL detector
# prompt is a real, previously-hit bug (see state_pipeline.py's own
# anchor-collision fix): Grounding DINO given duplicate text prompts
# in one batched call can't tell which detection belongs to which, so
# one starves. Every candidate here is checked against every OTHER
# query's current prompt in the same video before being adopted, so
# this can never introduce that collision -- worst case, a query
# simply keeps its original full text, exactly the old behavior.
#
# Queries with no such clause (e.g. "a man with a red backpack", "a
# yellow car", "a yellow dumpster filled with sand" -- "with sand" is
# a prepositional object of "filled", not a direct object, so it does
# not match this pattern) are returned byte-for-byte unchanged.
# ============================================================

_NLP = spacy.load("en_core_web_sm")

_NP_MODIFIER_DEPS = ("det", "compound", "amod", "nummod", "poss")


def _np_span_text(doc, token):
    """
    token + its own direct det/compound/amod/nummod/poss children, as
    contiguous source text -- never a conjunct (e.g. the "purple pants"
    joined to "jacket" by "and" stays out of this span on purpose).
    """

    children = [
        child
        for child in token.children
        if child.dep_ in _NP_MODIFIER_DEPS
    ]

    span = sorted(
        children + [token],
        key=lambda t: t.i,
    )

    return doc[span[0].i : span[-1].i + 1].text


def _wearing_clause_candidates(query_text):
    """
    If query_text has a head noun with a "wearing/carrying/..."
    participial (acl) clause with its own direct object, returns
    (bare_head, head_with_object). Otherwise returns None.
    """

    doc = _NLP(query_text)

    root = next(
        (token for token in doc if token.dep_ == "ROOT"),
        None,
    )

    if root is None or root.pos_ not in ("NOUN", "PROPN"):
        return None

    acl = next(
        (child for child in root.children if child.dep_ == "acl"),
        None,
    )

    if acl is None:
        return None

    obj = next(
        (
            child
            for child in acl.children
            if child.dep_ in ("dobj", "attr", "dative")
        ),
        None,
    )

    if obj is None:
        return None

    head_text = _np_span_text(doc, root)
    obj_text = _np_span_text(doc, obj)

    return head_text, f"{head_text} with {obj_text}"


def build_detector_prompts(query_texts):
    """
    One Grounding DINO detector prompt per query_text (same order), for
    one video's batched call. See the module comment above for the
    full reasoning; this never changes what SigLIP compares crops
    against (encode_texts always uses the original query_texts).
    """

    prompts = list(query_texts)  # default: unchanged

    for index, text in enumerate(query_texts):

        candidates = _wearing_clause_candidates(text)

        if candidates is None:
            continue

        bare, head_with_object = candidates

        others = [
            prompts[j]
            for j in range(len(prompts))
            if j != index
        ]

        if bare not in others:
            prompts[index] = bare
        elif head_with_object not in others:
            prompts[index] = head_with_object
        # else: leave prompts[index] as the original query_text.

    return prompts


# ============================================================
# LOAD SIGLIP
# ============================================================

print("\n================================")
print("Loading SigLIP")
print("================================")

print(f"Model:  {SIGLIP_MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"SigLIP dtype: {SIGLIP_DTYPE}")


siglip_processor = AutoProcessor.from_pretrained(
    SIGLIP_MODEL_NAME
)


siglip_model = AutoModel.from_pretrained(
    SIGLIP_MODEL_NAME,
    dtype=SIGLIP_DTYPE,
)


siglip_model = siglip_model.to(
    DEVICE
)

siglip_model.eval()


SIGLIP_DTYPE = next(
    siglip_model.parameters()
).dtype


print("SigLIP loaded.")


# ============================================================
# LOAD GROUNDING DINO
# ============================================================

print("\n================================")
print("Loading Grounding DINO")
print("================================")

print(
    f"Model: {DETECTOR_MODEL_NAME}"
)


detector_processor = AutoProcessor.from_pretrained(
    DETECTOR_MODEL_NAME
)


detector_model = (
    AutoModelForZeroShotObjectDetection
    .from_pretrained(
        DETECTOR_MODEL_NAME,
        dtype=DETECTOR_DTYPE,
    )
)


detector_model = detector_model.to(
    DEVICE
)

detector_model.eval()


print("Grounding DINO loaded.")


# ============================================================
# LOAD SAM2
# ============================================================

print("\n================================")
print("Loading SAM2")
print("================================")

print(f"Model: {sam2_utils.SAM2_MODEL_NAME}")

sam2_processor, sam2_model = sam2_utils.load_sam2(DEVICE)

print("SAM2 loaded.")


# ============================================================
# SIGLIP TEXT ENCODER
# ============================================================

def encode_texts(texts):
    """
    texts:
        list[str]

    Returns:
        normalized tensor:
        [num_queries, embedding_dim]
    """

    inputs = siglip_processor(
        text=texts,
        padding="max_length",
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        output = (
            siglip_model
            .get_text_features(
                **inputs
            )
        )

        features = get_embedding_tensor(
            output
        )

    features = features.float()

    features = F.normalize(
        features,
        p=2,
        dim=-1,
    )

    return features


# ============================================================
# SIGLIP IMAGE ENCODER
# ============================================================

def encode_images(images):
    """
    images:
        list[PIL.Image]

    Returns:
        normalized image embeddings:
        [num_images, embedding_dim]
    """

    if len(images) == 0:
        return None

    inputs = siglip_processor(
        images=images,
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(
        device=DEVICE,
        dtype=SIGLIP_DTYPE,
    )

    with torch.inference_mode():

        output = (
            siglip_model
            .get_image_features(
                pixel_values=pixel_values
            )
        )

        features = get_embedding_tensor(
            output
        )

    features = features.float()

    features = F.normalize(
        features,
        p=2,
        dim=-1,
    )

    return features


# ============================================================
# CROP
# ============================================================

def crop_box_with_margin(
    image,
    box,
    margin_ratio=CROP_MARGIN,
):
    """
    box:
        [x0, y0, x1, y1]

    Adds a small margin around the detected object.
    """

    width, height = image.size

    x0, y0, x1, y1 = [
        float(value)
        for value in box
    ]

    box_width = x1 - x0
    box_height = y1 - y0

    margin_x = (
        box_width * margin_ratio
    )

    margin_y = (
        box_height * margin_ratio
    )

    x0 = max(
        0,
        int(x0 - margin_x),
    )

    y0 = max(
        0,
        int(y0 - margin_y),
    )

    x1 = min(
        width,
        int(x1 + margin_x),
    )

    y1 = min(
        height,
        int(y1 + margin_y),
    )

    if x1 <= x0 or y1 <= y0:
        return None

    crop = image.crop(
        (
            x0,
            y0,
            x1,
            y1,
        )
    )

    return crop


# ============================================================
# GROUNDING DINO
# ============================================================

def detect_objects(
    image,
    query_texts,
):
    """
    Detect all object queries for one frame
    in ONE Grounding DINO inference.

    query_texts example:

        [
            "a man with a red backpack",
            "a cyclist",
            "a yellow car",
        ]

    Returns a list of detections.
    """

    text_labels = [
        query_texts
    ]

    inputs = detector_processor(
        images=image,
        text=text_labels,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = detector_model(
            **inputs
        )

    results = (
        detector_processor
        .post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=DETECTOR_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[
                (
                    image.height,
                    image.width,
                )
            ],
        )
    )

    result = results[0]

    boxes = result[
        "boxes"
    ].detach().cpu()

    scores = result[
        "scores"
    ].detach().cpu()

    # Current Transformers exposes text_labels.
    detected_labels = result.get(
        "text_labels"
    )

    if detected_labels is None:
        detected_labels = result.get(
            "labels"
        )

    detections = []

    for box, score, label in zip(
        boxes,
        scores,
        detected_labels,
    ):

        detections.append(
            {
                "box": box.tolist(),
                "detector_score":
                    float(score.item()),
                "label":
                    str(label),
            }
        )

    return detections


# ============================================================
# DETECTOR + CROP SIGLIP
# ============================================================

def extract_detector_crop_scores(
    video_path,
    query_texts,
    detector_prompts,
    query_embeddings,
    query_ids,
    video_id,
    debug_policy=None,
):
    """
    Second pass through the video.

    For every sampled frame:

        Grounding DINO (conditioned on detector_prompts -- possibly
        shortened, see build_detector_prompts)
             ↓
        bounding boxes
             ↓
            crops
             ↓
        SigLIP on crops (always scored against the FULL query_texts)

    Returns one detector score and crop score
    for every query at every timestamp.
    """

    # Generic (not per-query) color-word extraction: whatever color
    # words (if any) appear in each query's own text, e.g. "a yellow
    # car" -> ["yellow"], "a yellow-and-white car" -> ["yellow","white"].
    query_colors = [color_utils.extract_color_words(t) for t in query_texts]

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    num_queries = len(
        query_texts
    )

    # Generic (not per-query) pseudo-tracking: one tracker per query,
    # scoped to this one video. See tracking_utils.py for exactly what
    # this is and is not (NOT real video-object tracking with memory).
    trackers = [tracking_utils.QueryTracker() for _ in range(num_queries)]

    timestamps = []

    detector_rows = []
    crop_rows = []

    box_rows = []

    # SAM2-derived evidence, additive alongside the rectangular-crop
    # signals above -- the old detector_rows/crop_rows/box_rows are left
    # completely untouched.
    mask_area_rows = []
    sam_box_rows = []
    masked_crop_rows = []
    color_score_rows = []

    # Pseudo-tracking evidence (dicts per query per frame; see
    # tracking_utils.QueryTracker). Additive only, same as the SAM2/color
    # evidence above.
    track_rows = []

    frame_index = 0
    next_sample_time = 0.0

    progress = tqdm(
        total=frame_count,
        desc=(
            f"{video_path.stem} detector"
        ),
    )

    while True:

        success, frame = cap.read()

        if not success:
            break

        target_frame = round(
            next_sample_time
            * fps
        )

        if frame_index >= target_frame:

            timestamp = (
                frame_index / fps
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(
                rgb
            )

            # ----------------------------------------
            # Detect objects
            # ----------------------------------------

            detections = detect_objects(
                image,
                detector_prompts,
            )

            # Defaults:
            # no detection
            frame_detector_scores = (
                np.zeros(
                    num_queries,
                    dtype=np.float32,
                )
            )

            frame_crop_scores = (
                np.full(
                    num_queries,
                    np.nan,
                    dtype=np.float32,
                )
            )

            frame_boxes = [
                None
                for _ in range(
                    num_queries
                )
            ]

            # ----------------------------------------
            # Build crops
            # ----------------------------------------

            crop_images = []
            crop_query_indices = []
            crop_detection_scores = []
            crop_boxes = []

            for detection in detections:

                # Grounding DINO's returned label text echoes back
                # whatever we asked it to detect -- detector_prompts,
                # not the (possibly longer/different) original
                # query_texts -- so it must be matched against that
                # same list, or a shortened prompt would never map
                # back to its query_index.
                query_index = (
                    match_detection_to_query(
                        detection["label"],
                        detector_prompts,
                    )
                )

                crop = crop_box_with_margin(
                    image,
                    detection["box"],
                )

                if crop is None:
                    continue

                crop_images.append(
                    crop
                )

                crop_query_indices.append(
                    query_index
                )

                crop_detection_scores.append(
                    detection[
                        "detector_score"
                    ]
                )

                crop_boxes.append(
                    detection["box"]
                )

            # ----------------------------------------
            # Crop SigLIP
            # ----------------------------------------

            if crop_images:

                crop_embeddings = (
                    encode_images(
                        crop_images
                    )
                )

                for (
                    crop_embedding,
                    query_index,
                    detector_score,
                    box,
                ) in zip(
                    crop_embeddings,
                    crop_query_indices,
                    crop_detection_scores,
                    crop_boxes,
                ):

                    text_embedding = (
                        query_embeddings[
                            query_index
                        ]
                    )

                    crop_score = (
                        crop_embedding
                        @ text_embedding
                    ).item()

                    # Keep best detection for
                    # each query on this frame.

                    old_detector_score = (
                        frame_detector_scores[
                            query_index
                        ]
                    )

                    old_crop_score = (
                        frame_crop_scores[
                            query_index
                        ]
                    )

                    replace = False

                    if np.isnan(
                        old_crop_score
                    ):
                        replace = True

                    elif (
                        crop_score
                        > old_crop_score
                    ):
                        replace = True

                    if replace:

                        frame_detector_scores[
                            query_index
                        ] = (
                            detector_score
                        )

                        frame_crop_scores[
                            query_index
                        ] = (
                            crop_score
                        )

                        frame_boxes[
                            query_index
                        ] = box

            # ----------------------------------------
            # Pseudo-tracking: stitch this frame's winning box (if any)
            # onto each query's own running tracker. Depends only on
            # frame_boxes + this frame's image size -- independent of
            # the SAM2/color evidence below.
            # ----------------------------------------

            frame_track_fields = [
                trackers[query_index].update(
                    timestamp, frame_boxes[query_index], image.width, image.height
                )
                for query_index in range(num_queries)
            ]

            # ----------------------------------------
            # SAM2: precise mask for each query's winning
            # rectangular box from the pass above.
            #
            # Additive only -- frame_boxes / frame_crop_scores
            # (the old bbox-based evidence) are not touched here.
            # ----------------------------------------

            frame_mask_areas = np.zeros(num_queries, dtype=np.int64)
            frame_sam_boxes = [None for _ in range(num_queries)]
            frame_color_scores = np.full(num_queries, np.nan, dtype=np.float32)
            frame_dominant_colors = [None for _ in range(num_queries)]

            masked_crop_images = []
            masked_crop_query_indices = []

            # Kept only for this frame's debug-image rendering below;
            # not persisted into the CSV (the mask itself, not just its
            # bbox/area, isn't tabular data).
            frame_masks = {}
            frame_masked_crop_imgs = {}

            for query_index in range(num_queries):
                box = frame_boxes[query_index]
                if box is None:
                    continue

                mask, mask_bbox, mask_area = sam2_utils.predict_mask(
                    image, box, sam2_processor, sam2_model, DEVICE
                )
                frame_mask_areas[query_index] = mask_area
                frame_sam_boxes[query_index] = mask_bbox
                frame_masks[query_index] = mask

                # Color evidence, restricted to pixels INSIDE the SAM2
                # mask -- never the rectangular box, which is exactly
                # what let a beige car pass as "a yellow car" before
                # (the box's background pixels, not the car itself,
                # were doing the arguing).
                color_score, dominant_color = color_utils.masked_color_score(
                    image, mask, query_colors[query_index]
                )
                if color_score is not None:
                    frame_color_scores[query_index] = color_score
                    frame_dominant_colors[query_index] = dominant_color

                masked_crop = sam2_utils.masked_crop_with_margin(image, mask, mask_bbox)
                if masked_crop is not None:
                    masked_crop_images.append(masked_crop)
                    masked_crop_query_indices.append(query_index)
                    frame_masked_crop_imgs[query_index] = masked_crop

            frame_masked_crop_scores = np.full(num_queries, np.nan, dtype=np.float32)

            if masked_crop_images:
                masked_crop_embeddings = encode_images(masked_crop_images)
                for embedding, query_index in zip(
                    masked_crop_embeddings, masked_crop_query_indices
                ):
                    text_embedding = query_embeddings[query_index]
                    frame_masked_crop_scores[query_index] = (
                        embedding @ text_embedding
                    ).item()

            # ----------------------------------------
            # Debug images (opt-in, see debug_viz.DebugPolicy).
            # Every value below already existed above -- this only
            # decides whether to render it, never computes anything new.
            # ----------------------------------------

            if debug_policy is not None:
                for query_index in range(num_queries):
                    box = frame_boxes[query_index]
                    if box is None:
                        continue

                    det_score = float(frame_detector_scores[query_index])
                    bbox_score = frame_crop_scores[query_index]
                    bbox_score = float(bbox_score) if np.isfinite(bbox_score) else None
                    masked_score = frame_masked_crop_scores[query_index]
                    masked_score = float(masked_score) if np.isfinite(masked_score) else None

                    slots = debug_policy.slots_for(
                        query_ids[query_index], det_score, masked_score, bbox_score, timestamp
                    )
                    if not slots:
                        continue

                    bbox_crop_img = crop_box_with_margin(image, box)

                    color_score_val = frame_color_scores[query_index]
                    color_score_val = float(color_score_val) if np.isfinite(color_score_val) else None
                    expected_color_str = (
                        ",".join(query_colors[query_index]) if query_colors[query_index] else None
                    )

                    track_fields = frame_track_fields[query_index]

                    fields = {
                        "full_query": query_texts[query_index],
                        "automatic_anchor": (
                            detector_prompts[query_index]
                            if detector_prompts[query_index] != query_texts[query_index]
                            else None
                        ),
                        "detector_prompt": detector_prompts[query_index],
                        "detector_score": det_score,
                        "sam_bbox": frame_sam_boxes[query_index],
                        "sam_mask_area": int(frame_mask_areas[query_index]),
                        "siglip_bbox_score": bbox_score,
                        "siglip_masked_score": masked_score,
                        "expected_color": expected_color_str,
                        "color_score": color_score_val,
                        "track_id": track_fields["track_id"],
                        "center_x_norm": track_fields["center_x_norm"],
                        "center_y_norm": track_fields["center_y_norm"],
                        "dx": track_fields["dx"],
                        "dy": track_fields["dy"],
                        "speed": track_fields["speed"],
                    }
                    debug_viz.save_detection_debug(
                        category=DEBUG_ARGS.debug_category or "object",
                        query_id=query_ids[query_index],
                        video_id=video_id,
                        timestamp=timestamp,
                        frame_idx=frame_index,
                        detection_id=1,
                        image=image,
                        box=box,
                        mask=frame_masks.get(query_index),
                        bbox_crop=bbox_crop_img,
                        masked_crop=frame_masked_crop_imgs.get(query_index),
                        fields=fields,
                    )

            timestamps.append(
                timestamp
            )

            detector_rows.append(
                frame_detector_scores
            )

            crop_rows.append(
                frame_crop_scores
            )

            box_rows.append(
                frame_boxes
            )

            mask_area_rows.append(
                frame_mask_areas
            )

            sam_box_rows.append(
                frame_sam_boxes
            )

            masked_crop_rows.append(
                frame_masked_crop_scores
            )

            color_score_rows.append(
                frame_color_scores
            )

            track_rows.append(
                frame_track_fields
            )

            next_sample_time += (
                SAMPLE_INTERVAL_SEC
            )

        frame_index += 1

        progress.update(1)

    progress.close()

    cap.release()

    timestamps = np.asarray(
        timestamps,
        dtype=np.float32,
    )

    detector_scores = np.stack(
        detector_rows
    )

    crop_scores = np.stack(
        crop_rows
    )

    mask_areas = np.stack(
        mask_area_rows
    )

    masked_crop_scores = np.stack(
        masked_crop_rows
    )

    color_scores = np.stack(
        color_score_rows
    )

    return (
        timestamps,
        detector_scores,
        crop_scores,
        box_rows,
        mask_areas,
        sam_box_rows,
        masked_crop_scores,
        color_scores,
        track_rows,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def zscore_column(values):
    """
    Z-score normalization for one query over time.

    NaN values stay NaN.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    valid = np.isfinite(
        values
    )

    output = np.full_like(
        values,
        np.nan,
    )

    if valid.sum() == 0:
        return output

    valid_values = values[
        valid
    ]

    mean = valid_values.mean()
    std = valid_values.std()

    if std < 1e-8:

        output[
            valid
        ] = 0.0

        return output

    output[
        valid
    ] = (
        valid_values - mean
    ) / std

    return output


# ============================================================
# FUSION
# ============================================================

def fuse_scores(
    detector_scores,
    crop_scores,
):
    """
    We do NOT combine raw values directly because:

        detector:
            approximately 0...1

        SigLIP cosine:
            completely different scale

    First normalize each signal over time,
    then fuse normalized signals.

    Missing crop score means detector found
    no usable crop for that frame.
    """

    num_frames, num_queries = (
        detector_scores.shape
    )

    fused = np.zeros(
        (
            num_frames,
            num_queries,
        ),
        dtype=np.float32,
    )

    detector_z_all = np.zeros_like(
        detector_scores,
        dtype=np.float32,
    )

    crop_z_all = np.full_like(
        crop_scores,
        np.nan,
        dtype=np.float32,
    )

    for query_index in range(
        num_queries
    ):

        detector_z = zscore_column(
            detector_scores[
                :,
                query_index
            ]
        )

        crop_z = zscore_column(
            crop_scores[
                :,
                query_index
            ]
        )

        detector_z_all[
            :,
            query_index
        ] = detector_z

        crop_z_all[
            :,
            query_index
        ] = crop_z

        # --------------------------------------------
        # Dynamic fusion
        # --------------------------------------------

        for frame_index in range(
            num_frames
        ):

            # Detector is always represented:
            # 0 when nothing detected.
            values = [
                detector_z[
                    frame_index
                ]
            ]

            weights = [
                DETECTOR_WEIGHT
            ]

            # Crop exists only if detector
            # produced an object.
            crop_value = crop_z[
                frame_index
            ]

            if np.isfinite(
                crop_value
            ):

                values.append(
                    crop_value
                )

                weights.append(
                    CROP_WEIGHT
                )

            values = np.asarray(
                values,
                dtype=np.float32,
            )

            weights = np.asarray(
                weights,
                dtype=np.float32,
            )

            fused[
                frame_index,
                query_index,
            ] = np.sum(
                values * weights
            ) / np.sum(
                weights
            )

    return (
        fused,
        detector_z_all,
        crop_z_all,
    )


# ============================================================
# LOAD QUERIES
# ============================================================

queries = pd.read_csv(
    QUERIES_PATH
)


object_queries = queries[
    queries["query_type"]
    == "object"
].copy()

if DEBUG_ARGS.debug_query:
    # Restrict to that one query's own video -- no reason to touch the
    # other 7 videos just to debug a single query.
    object_queries = object_queries[
        object_queries["query_id"] == DEBUG_ARGS.debug_query
    ].copy()
    if object_queries.empty:
        raise SystemExit(
            f"--debug-query {DEBUG_ARGS.debug_query!r} is not an object query_id"
        )


print(
    f"\nObject queries: "
    f"{len(object_queries)}"
)

if DEBUG_ARGS.save_debug:
    print(
        f"Debug mode ON -- max_images_per_query={DEBUG_ARGS.debug_max_images}, "
        f"debug_query={DEBUG_ARGS.debug_query!r}"
    )


OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

all_rows = []


video_ids = sorted(
    object_queries[
        "video_id"
    ].unique()
)


for video_id in video_ids:

    print(
        "\n\n================================"
    )

    print(
        f"PROCESSING {video_id}"
    )

    print(
        "================================"
    )

    video_path = (
        DATA_DIR
        / f"{video_id}.mp4"
    )

    if not video_path.exists():

        raise FileNotFoundError(
            video_path
        )

    # --------------------------------------------------------
    # Queries for this video
    # --------------------------------------------------------

    video_queries = (
        object_queries[
            object_queries[
                "video_id"
            ]
            == video_id
        ]
        .reset_index(
            drop=True
        )
    )

    query_ids = (
        video_queries[
            "query_id"
        ].tolist()
    )

    query_texts = (
        video_queries[
            "query_text"
        ].tolist()
    )


    print(
        "\nQueries:"
    )

    # See build_detector_prompts: most queries keep their full query_text
    # as the detector prompt unchanged; a "wearing/carrying/..." clause
    # gets shortened (to a bare head noun, or "<head> with <object>") when
    # that is safe (no collision with another query in this same video).
    # Printed explicitly (not inferred) so this is always visible, the
    # same way state_pipeline.py prints its spaCy-derived anchor for each
    # query.
    detector_prompts = build_detector_prompts(query_texts)

    for query_id, query_text, detector_prompt in zip(
        query_ids,
        query_texts,
        detector_prompts,
    ):

        print(
            f"  {query_id} | query: {query_text!r} | detector_prompt: {detector_prompt!r}"
        )


    # --------------------------------------------------------
    # Encode query text once
    # --------------------------------------------------------

    query_embeddings = encode_texts(
        query_texts
    )


    # --------------------------------------------------------
    # Grounding DINO + crop SigLIP
    # --------------------------------------------------------

    debug_policy = None
    if DEBUG_ARGS.save_debug:
        _probe = cv2.VideoCapture(str(video_path))
        _fps = _probe.get(cv2.CAP_PROP_FPS)
        _frame_count = _probe.get(cv2.CAP_PROP_FRAME_COUNT)
        _probe.release()
        video_duration = _frame_count / _fps if _fps else None
        debug_policy = debug_viz.DebugPolicy(
            max_images=DEBUG_ARGS.debug_max_images,
            video_duration=video_duration,
        )

    (
        detector_timestamps,
        detector_scores,
        crop_scores,
        boxes,
        mask_areas,
        sam_boxes,
        masked_crop_scores,
        color_scores,
        track_rows,
    ) = extract_detector_crop_scores(
        video_path=video_path,
        query_texts=query_texts,
        detector_prompts=detector_prompts,
        query_embeddings=query_embeddings,
        query_ids=query_ids,
        video_id=video_id,
        debug_policy=debug_policy,
    )


    # --------------------------------------------------------
    # Fuse
    # --------------------------------------------------------

    (
        fused_scores,
        detector_z,
        crop_z,
    ) = fuse_scores(
        detector_scores=
            detector_scores,
        crop_scores=crop_scores,
    )


    # --------------------------------------------------------
    # Save rows
    # --------------------------------------------------------

    for query_index in range(
        len(query_ids)
    ):

        query_id = query_ids[
            query_index
        ]

        query_text = query_texts[
            query_index
        ]

        print(
            f"\nTop results "
            f"{query_id} | "
            f"{query_text}"
        )


        top_indices = np.argsort(
            fused_scores[
                :,
                query_index
            ]
        )[::-1][:10]


        for idx in top_indices:

            timestamp = (
                detector_timestamps[
                    idx
                ]
            )

            print(
                f"{timestamp:7.2f}s | "
                f"fused={fused_scores[idx, query_index]:7.3f} | "
                f"det={detector_scores[idx, query_index]:7.3f} | "
                f"crop={crop_scores[idx, query_index]:7.4f}"
            )


        for frame_index in range(
            len(
                detector_timestamps
            )
        ):

            box = boxes[
                frame_index
            ][
                query_index
            ]

            if box is None:

                x0 = np.nan
                y0 = np.nan
                x1 = np.nan
                y1 = np.nan

            else:

                (
                    x0,
                    y0,
                    x1,
                    y1,
                ) = box

            sam_box = sam_boxes[frame_index][query_index]
            if sam_box is None:
                sam_x0, sam_y0, sam_x1, sam_y1 = (np.nan,) * 4
            else:
                sam_x0, sam_y0, sam_x1, sam_y1 = sam_box

            mask_area_value = int(mask_areas[frame_index, query_index])

            masked_crop_score_value = masked_crop_scores[frame_index, query_index]
            masked_crop_score_value = (
                float(masked_crop_score_value)
                if np.isfinite(masked_crop_score_value)
                else np.nan
            )

            color_score_value = color_scores[frame_index, query_index]
            color_score_value = (
                float(color_score_value) if np.isfinite(color_score_value) else np.nan
            )

            query_color_words = color_utils.extract_color_words(query_text)
            expected_color_value = ",".join(query_color_words) if query_color_words else np.nan

            track_fields = track_rows[frame_index][query_index]

            all_rows.append(
                {
                    "video_id":
                        video_id,

                    "query_id":
                        query_id,

                    "query_text":
                        query_text,

                    "time":
                        float(
                            detector_timestamps[
                                frame_index
                            ]
                        ),

                    "detector_score":
                        float(
                            detector_scores[
                                frame_index,
                                query_index,
                            ]
                        ),

                    "crop_score":
                        float(
                            crop_scores[
                                frame_index,
                                query_index,
                            ]
                        )
                        if np.isfinite(
                            crop_scores[
                                frame_index,
                                query_index,
                            ]
                        )
                        else np.nan,

                    "detector_z":
                        float(
                            detector_z[
                                frame_index,
                                query_index,
                            ]
                        ),

                    "crop_z":
                        float(
                            crop_z[
                                frame_index,
                                query_index,
                            ]
                        )
                        if np.isfinite(
                            crop_z[
                                frame_index,
                                query_index,
                            ]
                        )
                        else np.nan,

                    "fused_score":
                        float(
                            fused_scores[
                                frame_index,
                                query_index,
                            ]
                        ),

                    "x0":
                        x0,

                    "y0":
                        y0,

                    "x1":
                        x1,

                    "y1":
                        y1,

                    # ---- SAM2-derived evidence (additive, new) ----

                    "mask_area":
                        mask_area_value,

                    "sam_x0":
                        sam_x0,

                    "sam_y0":
                        sam_y0,

                    "sam_x1":
                        sam_x1,

                    "sam_y1":
                        sam_y1,

                    "masked_crop_score":
                        masked_crop_score_value,

                    "expected_color":
                        expected_color_value,

                    "color_score":
                        color_score_value,

                    # ---- Pseudo-tracking evidence (additive, new) ----
                    # See tracking_utils.py: consecutive independent
                    # per-frame detections stitched into tracks, NOT
                    # real video-object tracking with memory.

                    "track_id":
                        _or_nan(track_fields["track_id"]),

                    "center_x_norm":
                        _or_nan(track_fields["center_x_norm"]),

                    "center_y_norm":
                        _or_nan(track_fields["center_y_norm"]),

                    "width":
                        _or_nan(track_fields["width"]),

                    "height":
                        _or_nan(track_fields["height"]),

                    "area":
                        _or_nan(track_fields["area"]),

                    "dx":
                        _or_nan(track_fields["dx"]),

                    "dy":
                        _or_nan(track_fields["dy"]),

                    "speed":
                        _or_nan(track_fields["speed"]),

                    "delta_area":
                        _or_nan(track_fields["delta_area"]),

                    "track_stability":
                        _or_nan(track_fields["track_stability"]),
                }
            )


    # --------------------------------------------------------
    # Free temporary GPU memory
    # --------------------------------------------------------

    torch.cuda.empty_cache()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_rows
)


# A --debug-query run only ever covers one query -- writing that to the
# normal filename would silently clobber the full 24-query CSV built by
# a real run (this happened once already: a Prompt-2 debug test on q003
# overwrote the full object_frame_scores.csv down to 442 rows). Anything
# that restricts the query set gets its own filename instead.
if DEBUG_ARGS.debug_query:
    output_path = (
        OUTPUT_DIR
        / f"object_frame_scores_debug_{DEBUG_ARGS.debug_query}.csv"
    )
else:
    output_path = (
        OUTPUT_DIR
        / "object_frame_scores.csv"
    )


results_df.to_csv(
    output_path,
    index=False,
)


print(
    "\n\n================================"
)

print(
    "OBJECT PIPELINE FINISHED"
)

print(
    "================================"
)

print(
    f"Rows: {len(results_df)}"
)

print(
    f"Saved: {output_path}"
)