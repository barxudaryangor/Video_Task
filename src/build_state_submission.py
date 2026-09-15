"""
Score(t) -> intervals for the 24 "state" queries.

Consumes outputs/state_frame_scores.csv (produced by state_pipeline.py)
and, per query's mode:

  - "presence": same evidence-gate + gap-merge segmentation as object
    queries (detector_score>0 and crop_score>=MIN_CROP_SCORE).
    Queries whose text is fundamentally about NOT MOVING ("stationary",
    "parked") get an extra check: the kept segment's box must not
    wander much in position, since a single frame cannot tell stationary
    from moving -- only the box's position across time can.

  - "absence": ignores crop_score entirely (comparing a crop to a
    negated sentence is meaningless) and instead reports the GAPS in
    time where the anchor was never detected, as long as that gap is
    wide enough to not just be an ordinary detector miss.

Writes/updates outputs/submission.csv, merging with any existing
object-category predictions already there rather than overwriting them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

QUERIES_PATH = DATA_DIR / "queries.csv"
STATE_SCORES_PATH = OUTPUT_DIR / "state_frame_scores.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

MIN_CROP_SCORE = 0.13
MAX_GAP_SEC = 5.0

# Wider gap tolerance for ordinary presence queries only (NOT the
# stationary ones -- see below). State anchors are longer/rarer noun
# phrases than object queries (e.g. "a tractor on its side" vs "a
# car") and Grounding DINO fires on them far less reliably per-frame --
# confirmed against a real known event (q059: tractor tips at t~52s,
# righted at t~180s, per user-confirmed ground truth): the raw evidence
# gate only fires on 48/281 sampled frames in that span, with gaps up
# to 15.5s despite the tractor lying there the whole time, fragmenting
# one real ~128s event into 10 pieces at the old 5.0s gap.
#
# Deliberately NOT applied to STATIONARY_QUERY_IDS: merging q004/q006/
# q033's fragments at this wider gap reveals box-position std of
# 100-227px -- these were never one continuously-tracked stationary
# object, just several short low-variance fragments that individually
# slipped past the (5.0s-calibrated) stationarity check. Widening their
# gap too would launder that inconsistency into one big interval that
# then fails filter_stationary and flips to NONE -- a real behavior
# change with no ground truth backing it, unlike q059. Left at the
# original 5.0s for those six query_ids until independently verified.
MAX_GAP_SEC_NON_STATIONARY = 20.0

MIN_DURATION_SEC = 0.1
BOUNDARY_PAD_SEC = 0.25
MIN_ABSENCE_DURATION_SEC = 10.0
# Recalibrated from 3.0 on the full 8-video rerun: checked all 5 absence
# queries' actual detector_score>0 gap distributions. q014/q024 are
# unaffected either way (never-detected / always-detected). q032 and
# q067 have their own detector reliably firing 83-96% of frames with
# just 1-3 short gaps that don't survive past ~5-8s. q068 ("shopping
# carts in the frame") is the outlier: only 35% of frames detect the
# anchor at all (occluded/cluttered checkout scene), producing 17
# gaps >3s -- clearly detector noise, not real intermittent absence.
# 10.0s keeps every genuine long gap while dropping most of that noise
# (17 -> 5 for q068) without touching the queries that were already fine.

# Queries whose text asserts the subject is NOT MOVING -- a single
# frame cannot show this, only box position across multiple frames
# can, so these get an extra stationarity check on top of the normal
# presence gate.
STATIONARY_QUERY_IDS = {"q004", "q005", "q006", "q033", "q042", "q049"}
# 15.0 (original guess, never checked against real box-position data) was
# rejecting almost every real detection window -- Grounding DINO's own box
# jitter on a genuinely parked vehicle runs well past 15px per axis even
# frame-to-frame, before the object ever moves. Recalibrated against the
# user's ground truth (outputs/ground_truth_manual_reference.csv, all 6
# STATIONARY_QUERY_IDS): the true positive/negative boundary sits on a flat
# plateau from ~130 to ~155px (q004: F1 0.000 -> 0.476; q033: 0.167 -> 0.222;
# both plateau exactly across that range, nothing regresses -- q005/q049,
# the two queries whose real answer is NONE, never pass the evidence gate at
# all regardless of this constant, so they're unaffected either way). 140 is
# the plateau midpoint, not an edge value.
MAX_STATIONARY_STD_PX = 140.0  # box-center std (pixels) allowed to still call it "not moving"


def presence_intervals(group, video_end, max_gap_sec=MAX_GAP_SEC):
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
        if times[idx] - times[merged[-1][1]] <= max_gap_sec:
            merged[-1][1] = idx
        else:
            merged.append([idx, idx])

    results = []
    for s, e in merged:
        start = max(0.0, times[s] - BOUNDARY_PAD_SEC)
        end = min(video_end, times[e] + BOUNDARY_PAD_SEC)
        if end - start < MIN_DURATION_SEC:
            continue
        results.append((start, end, s, e))
    return results


def filter_stationary(group, intervals):
    """Drop segments where the box moves too much to plausibly be
    described as 'stationary' / 'parked'."""
    if not intervals:
        return intervals

    x0, y0, x1, y1 = group["x0"].to_numpy(), group["y0"].to_numpy(), group["x1"].to_numpy(), group["y1"].to_numpy()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    kept = []
    for start, end, s_idx, e_idx in intervals:
        window_cx = cx[s_idx : e_idx + 1]
        window_cy = cy[s_idx : e_idx + 1]
        valid = np.isfinite(window_cx) & np.isfinite(window_cy)
        if valid.sum() < 2:
            continue
        std = max(window_cx[valid].std(), window_cy[valid].std())
        if std <= MAX_STATIONARY_STD_PX:
            kept.append((start, end))
    return kept


def absence_intervals(group, video_end):
    times = group["time"].to_numpy()
    detector_scores = group["detector_score"].to_numpy()
    present = detector_scores > 0

    if not present.any():
        # Anchor never detected at all -> "absent" the whole video.
        return [(0.0, video_end)]

    present_idxs = np.where(present)[0]

    gaps = []
    if times[present_idxs[0]] - 0.0 >= MIN_ABSENCE_DURATION_SEC:
        gaps.append((0.0, times[present_idxs[0]]))

    for a, b in zip(present_idxs[:-1], present_idxs[1:]):
        gap_start, gap_end = times[a], times[b]
        if gap_end - gap_start >= MIN_ABSENCE_DURATION_SEC:
            gaps.append((gap_start, gap_end))

    if video_end - times[present_idxs[-1]] >= MIN_ABSENCE_DURATION_SEC:
        gaps.append((times[present_idxs[-1]], video_end))

    return gaps


def format_prediction(intervals):
    if not intervals:
        return "NONE"
    return ";".join(f"{s:.2f} {e:.2f}" for s, e in intervals)


def main():
    queries = pd.read_csv(QUERIES_PATH)
    scores_df = pd.read_csv(STATE_SCORES_PATH)

    video_end_by_id = {vid: g.time.max() for vid, g in scores_df.groupby("video_id")}
    predictions = {}
    diag = {}

    for qid, g in scores_df.groupby("query_id"):
        g = g.sort_values("time").reset_index(drop=True)
        video_id = g["video_id"].iloc[0]
        mode = g["mode"].iloc[0]
        video_end = video_end_by_id[video_id]

        if mode == "absence":
            intervals = absence_intervals(g, video_end)
        else:
            if qid in STATIONARY_QUERY_IDS:
                raw_intervals = presence_intervals(g, video_end, max_gap_sec=MAX_GAP_SEC)
                intervals = filter_stationary(g, raw_intervals)
            else:
                raw_intervals = presence_intervals(g, video_end, max_gap_sec=MAX_GAP_SEC_NON_STATIONARY)
                intervals = [(s, e) for s, e, _, _ in raw_intervals]

        predictions[qid] = format_prediction(intervals)

        valid_crop = g["crop_score"].dropna()
        diag[qid] = {
            "mode": mode,
            "max_det": g["detector_score"].max(),
            "max_crop": valid_crop.max() if len(valid_crop) else float("nan"),
        }

    # Merge into submission.csv, keeping any existing (object) predictions.
    if SUBMISSION_PATH.exists():
        existing = pd.read_csv(SUBMISSION_PATH).set_index("query_id")["prediction"].to_dict()
    else:
        existing = {}

    rows = []
    for _, row in queries.iterrows():
        qid = row["query_id"]
        if qid in predictions:
            pred = predictions[qid]
        else:
            pred = existing.get(qid, "NONE")
        rows.append({"query_id": qid, "prediction": pred})

    pd.DataFrame(rows).to_csv(SUBMISSION_PATH, index=False)

    print(f"Wrote {len(rows)} rows -> {SUBMISSION_PATH}")
    non_none = sum(1 for qid in predictions if predictions[qid] != "NONE")
    print(f"State queries with a predicted interval: {non_none}/{len(predictions)}")

    print("\nPer-query state predictions:")
    for qid in sorted(predictions, key=lambda x: int(x[1:])):
        text = queries.loc[queries.query_id == qid, "query_text"].iloc[0]
        d = diag[qid]
        print(f"  {qid} [{d['mode']:8s}] {text!r:55s} (max_det={d['max_det']:.2f}, max_crop={d['max_crop']:.3f}) -> {predictions[qid]}")


if __name__ == "__main__":
    main()
