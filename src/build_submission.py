"""
Score(t) -> intervals -> submission.csv

Consumes outputs/object_frame_scores.csv (fused SigLIP + Grounding DINO
scores per query per sampled timestamp, produced by object_pipeline.py)
and turns each query's score curve into predicted [start, end] intervals
in the competition's submission format.

Queries with no frame-level scores yet (state/action) are written as
NONE so the submission file always has one row per query_id.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

QUERIES_PATH = DATA_DIR / "queries.csv"
OBJECT_SCORES_PATH = OUTPUT_DIR / "object_frame_scores.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# ------------------------------------------------------------------
# Segmentation is driven directly by an ABSOLUTE per-frame evidence
# gate, not by a smoothed/z-scored curve.
#
# An earlier version smoothed fused_score (a per-query z-score) and
# ran hysteresis on it. That construction has two failure modes that
# both surfaced on real queries:
#
#  1. Smoothing can average away a single strong, isolated true
#     detection. q001 ("a man with a red backpack") has a real,
#     independently-confirmed appearance at t=37s -- one sampled
#     frame, detector_score=0.42, crop_score=0.088 (both clear the
#     gate on their own). Averaged with 0.5s neighbors that have no
#     detection at all, the smoothed curve only reaches 0.32, well
#     under the 0.75 hysteresis entry bar -- a real hit, deleted by
#     smoothing tuned for a different problem.
#
#  2. z-scoring is relative to each query's own mean, so it can drop
#     below the hysteresis EXIT bar even while the object is present
#     in literally every sampled frame -- q055 ("a yellow dumpster
#     filled with sand") has detector_score > 0 and crop_score above
#     the gate in all 600/600 sampled frames of its video (the object
#     never leaves view), yet the smoothed/hysteresis version still
#     fragmented it into 5 separate intervals.
#
# Using the same detector_score>0 / crop_score>=MIN_CROP_SCORE gate
# directly, per frame, and merging only the surviving frames (bridging
# short real gaps with MAX_GAP_SEC) fixes both: single strong frames
# survive on their own merit, and a query with no real gap in
# evidence stays one interval.
# ------------------------------------------------------------------

# detector_score > 0 already means Grounding DINO matched this exact
# query phrase to a box above its own confidence threshold (0 is the
# "nothing detected" filler, not a low score).
#
# crop_score is raw SigLIP cosine similarity between the *cropped*
# detected region and the query text.
#
# Recalibrated for siglip2-giant-opt-patch16-384 by sweeping the
# threshold against known-verified queries (see report) rather than
# picking one number from a couple of examples:
#
#   - Too low (<=0.10): confirmed false positives survive as
#     confident, large false intervals -- q003 ("a yellow car" ->
#     really a beige car) and q021 ("a blue umbrella" -> really red)
#     both persist as near-whole-video matches. This is a real loss,
#     not a neutral guess: these look like genuine empty-ground-truth
#     queries, where NONE scores 1.0 but a wrong non-empty prediction
#     scores 0 -- a confident wrong guess here forfeits a free point,
#     unlike a query that truly has *some* answer we merely mislocate.
#   - 0.13: both confirmed false positives (q003, q021) correctly
#     collapse to NONE, while confirmed true positives survive:
#     q002 ("a cyclist", verified) keeps 12s of coverage, q055/q057
#     (verified continuously-present objects) are essentially
#     unaffected. The smallest confirmed true positive (a small child
#     with a red backpack, q001) is reduced to a marginal 0.5s -- a
#     real cost, but smaller than the cost of keeping q003/q021 wrong.
#   - Higher (>=0.14): q001 disappears entirely and q002 starts
#     eroding -- costs a confirmed true positive for no further gain.
#
# 0.13 is not perfect: one confirmed false positive survives regardless
# of this threshold (q065, "yellow shorts" -> a static background
# artifact glimpsed through a window, scoring in the SAME range as
# real matches) -- that failure mode needs a different signal
# (e.g. box-position stability over time) than a crop-score cutoff
# can provide, and is left as a documented limitation.
MIN_CROP_SCORE = 0.13

# Bridge short dropouts (the detector missing a frame or two of an
# otherwise continuously-visible object) so one gap doesn't split a
# single event into two intervals. Large enough to bridge ordinary
# frame-to-frame detector misses, small enough not to transitively
# chain distant, unrelated sightings into one interval.
MAX_GAP_SEC = 5.0

# Drop candidate intervals shorter than this. Set near zero: a single
# sampled frame that clears the evidence gate on its own is already
# real, grounded evidence (Section 3.1 in the report) -- it does not
# need a sustained run of frames to be trusted the way a raw z-score
# peak would.
MIN_DURATION_SEC = 0.1

# Half a sample step, added to both ends of a kept interval since the
# true object boundary lies somewhere between two samples.
BOUNDARY_PAD_SEC = 0.25


def extract_intervals(group, video_end):
    times = group["time"].to_numpy()
    detector_scores = group["detector_score"].to_numpy()
    crop_scores = group["crop_score"].to_numpy()

    crop_scores_filled = np.where(np.isfinite(crop_scores), crop_scores, -1.0)
    evidence = (detector_scores > 0) & (crop_scores_filled >= MIN_CROP_SCORE)

    evidence_idxs = np.where(evidence)[0]
    if len(evidence_idxs) == 0:
        return []

    # Merge evidence frames separated by a short gap in time.
    merged = [[evidence_idxs[0], evidence_idxs[0]]]
    for idx in evidence_idxs[1:]:
        gap = times[idx] - times[merged[-1][1]]
        if gap <= MAX_GAP_SEC:
            merged[-1][1] = idx
        else:
            merged.append([idx, idx])

    results = []
    for start_idx, end_idx in merged:
        start_time = max(0.0, times[start_idx] - BOUNDARY_PAD_SEC)
        end_time = min(video_end, times[end_idx] + BOUNDARY_PAD_SEC)

        if end_time - start_time < MIN_DURATION_SEC:
            continue

        results.append((start_time, end_time))

    return results


def format_prediction(intervals):
    if not intervals:
        return "NONE"
    return ";".join(f"{start:.2f} {end:.2f}" for start, end in intervals)


def main():
    queries = pd.read_csv(QUERIES_PATH)
    scores_df = pd.read_csv(OBJECT_SCORES_PATH)

    video_end_by_id = {}
    for video_id, group in scores_df.groupby("video_id"):
        video_end_by_id[video_id] = group["time"].max()

    predictions = {}
    gate_stats = {}

    for query_id, group in scores_df.groupby("query_id"):
        group = group.sort_values("time")
        video_id = group["video_id"].iloc[0]
        video_end = video_end_by_id[video_id]

        intervals = extract_intervals(group=group, video_end=video_end)

        predictions[query_id] = format_prediction(intervals)

        detector_scores = group["detector_score"].to_numpy()
        crop_scores = group["crop_score"].to_numpy()
        valid_crop = crop_scores[np.isfinite(crop_scores)]
        gate_stats[query_id] = {
            "max_detector": detector_scores.max(),
            "max_crop": valid_crop.max() if valid_crop.size else float("nan"),
        }

    rows = []
    for _, row in queries.iterrows():
        query_id = row["query_id"]
        prediction = predictions.get(query_id, "NONE")
        rows.append({"query_id": query_id, "prediction": prediction})

    submission_df = pd.DataFrame(rows)

    missing_scores = set(queries["query_id"]) - set(predictions)
    if missing_scores:
        print(
            f"{len(missing_scores)} queries have no frame scores yet "
            f"(state/action) -> written as NONE"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nWrote {len(submission_df)} rows -> {SUBMISSION_PATH}")

    object_query_ids = set(scores_df["query_id"].unique())
    non_none = sum(
        1 for qid in object_query_ids if predictions[qid] != "NONE"
    )
    print(f"Object queries with a predicted interval: {non_none}/{len(object_query_ids)}")

    print("\nPer-query predictions (object):")
    for query_id in sorted(object_query_ids, key=lambda x: int(x[1:])):
        text = queries.loc[queries["query_id"] == query_id, "query_text"].iloc[0]
        stats = gate_stats[query_id]
        print(
            f"  {query_id} | {text!r:55s} "
            f"(max_det={stats['max_detector']:.2f}, max_crop={stats['max_crop']:.3f}) "
            f"-> {predictions[query_id]}"
        )


if __name__ == "__main__":
    main()
