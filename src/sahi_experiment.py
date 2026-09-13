"""
SAHI-style tiled detection experiment on video_01 and video_06.

Standalone (does not import object_pipeline.py, which runs its whole
pipeline at import time). Slices each sampled frame into a 2x2
overlapping grid + the full frame itself, runs Grounding DINO on each
tile separately (so small objects aren't downsampled away with the
rest of the frame), maps boxes back to full-frame coordinates, and
keeps the highest-scoring box per query per frame across all tiles.

Only covers video_01's and video_06's own object queries, at a coarser
1.0s sample interval, to keep this a quick comparison rather than a
full rerun.
"""

from pathlib import Path
from difflib import SequenceMatcher

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoModel,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
QUERIES_PATH = DATA_DIR / "queries.csv"

SIGLIP_MODEL_NAME = "google/siglip2-giant-opt-patch16-384"
DETECTOR_MODEL_NAME = "IDEA-Research/grounding-dino-base"

SAMPLE_INTERVAL_SEC = 1.0
DETECTOR_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
CROP_MARGIN = 0.10

TILE_GRID = (2, 2)      # cols, rows
TILE_OVERLAP = 0.20     # fraction of tile size

MIN_CROP_SCORE = 0.13   # same as current build_submission.py, for comparability
MAX_GAP_SEC = 5.0
BOUNDARY_PAD_SEC = 0.25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIGLIP_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

VIDEOS = ["video_01", "video_06"]


def load_models():
    print("Loading SigLIP...")
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, dtype=SIGLIP_DTYPE).to(DEVICE).eval()
    print("Loading Grounding DINO...")
    detector_processor = AutoProcessor.from_pretrained(DETECTOR_MODEL_NAME)
    detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR_MODEL_NAME, dtype=torch.float32
    ).to(DEVICE).eval()
    print("Models loaded.")
    return siglip_processor, siglip_model, detector_processor, detector_model


def encode_texts(processor, model, texts):
    inputs = processor(text=texts, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.get_text_features(**inputs)
    feats = out if torch.is_tensor(out) else out.pooler_output
    return F.normalize(feats.float(), p=2, dim=-1)


def encode_images(processor, model, images):
    if not images:
        return None
    inputs = processor(images=images, return_tensors="pt")
    pv = inputs["pixel_values"].to(DEVICE, dtype=SIGLIP_DTYPE)
    with torch.inference_mode():
        out = model.get_image_features(pixel_values=pv)
    feats = out if torch.is_tensor(out) else out.pooler_output
    return F.normalize(feats.float(), p=2, dim=-1)


def normalize_text(text):
    return str(text).lower().strip().rstrip(".").strip()


def match_detection_to_query(label, query_texts):
    label = normalize_text(label)
    best_i, best_r = None, -1.0
    for i, q in enumerate(query_texts):
        r = SequenceMatcher(None, label, normalize_text(q)).ratio()
        if r > best_r:
            best_r, best_i = r, i
    return best_i


def get_tiles(width, height):
    cols, rows = TILE_GRID
    step_x, step_y = width / cols, height / rows
    ov_x, ov_y = step_x * TILE_OVERLAP, step_y * TILE_OVERLAP
    tiles = [(0, 0, width, height)]  # full frame first
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(c * step_x - ov_x))
            y0 = max(0, int(r * step_y - ov_y))
            x1 = min(width, int((c + 1) * step_x + ov_x))
            y1 = min(height, int((r + 1) * step_y + ov_y))
            tiles.append((x0, y0, x1, y1))
    return tiles


def detect_tiled(image, query_texts, detector_processor, detector_model):
    """Runs the detector on the full frame + each tile, maps boxes back
    to full-frame coordinates, and keeps the best-scoring detection per
    query across all tiles (a query can only have one true instance
    location per frame in this dataset's queries)."""
    width, height = image.size
    tiles = get_tiles(width, height)

    best_by_query = {}  # query_index -> (score, box_in_full_frame_coords, label)

    text_labels = [query_texts]

    for (tx0, ty0, tx1, ty1) in tiles:
        tile_img = image.crop((tx0, ty0, tx1, ty1))
        if tile_img.width < 20 or tile_img.height < 20:
            continue

        inputs = detector_processor(images=tile_img, text=text_labels, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = detector_model(**inputs)

        results = detector_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=DETECTOR_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(tile_img.height, tile_img.width)],
        )[0]

        boxes = results["boxes"].detach().cpu()
        scores = results["scores"].detach().cpu()
        labels = results.get("text_labels") or results.get("labels")

        for box, score, label in zip(boxes, scores, labels):
            qi = match_detection_to_query(label, query_texts)
            if qi is None:
                continue
            x0, y0, x1, y1 = box.tolist()
            full_box = (x0 + tx0, y0 + ty0, x1 + tx0, y1 + ty0)
            score = float(score.item())
            if qi not in best_by_query or score > best_by_query[qi][0]:
                best_by_query[qi] = (score, full_box, label)

    return best_by_query


def crop_with_margin(image, box):
    width, height = image.size
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, by = x1 - x0, y1 - y0
    mx, my = bw * CROP_MARGIN, by * CROP_MARGIN
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(width, int(x1 + mx)), min(height, int(y1 + my))
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1))


def process_video(video_id, queries_df, siglip_processor, siglip_model, detector_processor, detector_model):
    video_path = DATA_DIR / f"{video_id}.mp4"
    video_queries = queries_df[queries_df.video_id == video_id].reset_index(drop=True)
    query_ids = video_queries.query_id.tolist()
    query_texts = video_queries.query_text.tolist()

    query_embeddings = encode_texts(siglip_processor, siglip_model, query_texts)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    n_samples = int(duration / SAMPLE_INTERVAL_SEC) + 1
    rows = []

    for i in tqdm(range(n_samples), desc=f"{video_id} tiled"):
        t = i * SAMPLE_INTERVAL_SEC
        frame_idx = int(round(t * fps))
        if frame_idx >= frame_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        best_by_query = detect_tiled(image, query_texts, detector_processor, detector_model)

        crop_images, crop_query_idxs = [], []
        for qi, (score, box, label) in best_by_query.items():
            crop = crop_with_margin(image, box)
            if crop is not None:
                crop_images.append(crop)
                crop_query_idxs.append(qi)

        crop_embeddings = encode_images(siglip_processor, siglip_model, crop_images)

        for qi in range(len(query_ids)):
            det_score = best_by_query[qi][0] if qi in best_by_query else 0.0
            crop_score = float("nan")
            if qi in crop_query_idxs and crop_embeddings is not None:
                idx = crop_query_idxs.index(qi)
                crop_score = float((crop_embeddings[idx] @ query_embeddings[qi]).item())

            rows.append({
                "video_id": video_id,
                "query_id": query_ids[qi],
                "query_text": query_texts[qi],
                "time": t,
                "detector_score": det_score,
                "crop_score": crop_score,
            })

    cap.release()
    return pd.DataFrame(rows)


def extract_intervals(group, video_end):
    times = group["time"].to_numpy()
    detector_scores = group["detector_score"].to_numpy()
    crop_scores = group["crop_score"].to_numpy()
    crop_filled = np.where(np.isfinite(crop_scores), crop_scores, -1.0)
    evidence = (detector_scores > 0) & (crop_filled >= MIN_CROP_SCORE)
    idxs = np.where(evidence)[0]
    if len(idxs) == 0:
        return []
    merged = [[idxs[0], idxs[0]]]
    for idx in idxs[1:]:
        if times[idx] - times[merged[-1][1]] <= MAX_GAP_SEC:
            merged[-1][1] = idx
        else:
            merged.append([idx, idx])
    results = []
    for s, e in merged:
        results.append((max(0.0, times[s] - BOUNDARY_PAD_SEC), min(video_end, times[e] + BOUNDARY_PAD_SEC)))
    return results


def main():
    queries_df = pd.read_csv(QUERIES_PATH)
    queries_df = queries_df[(queries_df.video_id.isin(VIDEOS)) & (queries_df.query_type == "object")]

    siglip_processor, siglip_model, detector_processor, detector_model = load_models()

    all_dfs = []
    for video_id in VIDEOS:
        df = process_video(video_id, queries_df, siglip_processor, siglip_model, detector_processor, detector_model)
        all_dfs.append(df)

    result_df = pd.concat(all_dfs, ignore_index=True)
    result_df.to_csv(OUTPUT_DIR / "sahi_experiment_scores.csv", index=False)

    print("\n" + "=" * 60)
    print("SAHI TILED EXPERIMENT RESULTS")
    print("=" * 60)

    for video_id in VIDEOS:
        vg = result_df[result_df.video_id == video_id]
        video_end = vg.time.max()
        print(f"\n--- {video_id} ---")
        for qid, g in vg.groupby("query_id"):
            text = g.query_text.iloc[0]
            valid_crop = g.crop_score.dropna()
            max_crop = valid_crop.max() if len(valid_crop) else float("nan")
            max_det = g.detector_score.max()
            intervals = extract_intervals(g.sort_values("time"), video_end)
            interval_str = ";".join(f"{s:.2f}-{e:.2f}" for s, e in intervals) if intervals else "NONE"
            print(f"  {qid} | {text!r:45s} max_det={max_det:.2f} max_crop={max_crop:.3f} -> {interval_str}")

    print(f"\nSaved raw scores -> {OUTPUT_DIR / 'sahi_experiment_scores.csv'}")


if __name__ == "__main__":
    main()
