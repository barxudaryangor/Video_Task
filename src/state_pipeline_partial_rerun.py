"""
Reruns state_pipeline.py's detection for only video_02 and video_03
(the two videos whose queries had a duplicate anchor-phrase collision,
now fixed in QUERY_CONFIG) and merges the corrected rows back into
outputs/state_frame_scores.csv, replacing only those videos' rows.
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor, AutoModelForZeroShotObjectDetection

from state_pipeline import (
    QUERIES_PATH,
    OUTPUT_DIR,
    SIGLIP_MODEL_NAME,
    DETECTOR_MODEL_NAME,
    SIGLIP_DTYPE,
    DETECTOR_DTYPE,
    DEVICE,
    QUERY_CONFIG,
    process_video,
)

VIDEOS_TO_REDO = ["video_02", "video_03"]


def main():
    queries = pd.read_csv(QUERIES_PATH)
    state_queries = queries[queries.query_type == "state"].copy()

    print("Loading SigLIP...")
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, dtype=SIGLIP_DTYPE).to(DEVICE).eval()

    print("Loading Grounding DINO...")
    detector_processor = AutoProcessor.from_pretrained(DETECTOR_MODEL_NAME)
    detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR_MODEL_NAME, dtype=DETECTOR_DTYPE
    ).to(DEVICE).eval()

    new_dfs = []
    for video_id in VIDEOS_TO_REDO:
        video_queries = state_queries[state_queries.video_id == video_id].reset_index(drop=True)
        print(f"\n=== {video_id} (redo) ===")
        for _, r in video_queries.iterrows():
            print(f"  {r.query_id} [{QUERY_CONFIG[r.query_id]['mode']}] anchor={QUERY_CONFIG[r.query_id]['anchor']!r}")
        df = process_video(video_id, video_queries, siglip_processor, siglip_model, detector_processor, detector_model)
        new_dfs.append(df)
        torch.cuda.empty_cache()

    new_df = pd.concat(new_dfs, ignore_index=True)

    existing_path = OUTPUT_DIR / "state_frame_scores.csv"
    existing = pd.read_csv(existing_path)
    kept = existing[~existing.video_id.isin(VIDEOS_TO_REDO)]

    merged = pd.concat([kept, new_df], ignore_index=True)
    merged.to_csv(existing_path, index=False)
    print(f"\nMerged. Total rows: {len(merged)} -> {existing_path}")


if __name__ == "__main__":
    main()
