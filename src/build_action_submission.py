"""
outputs/action_candidate_windows.csv (Qwen3-VL's per-window yes/no +
parsed start/end, produced by action_pipeline.py) -> final [start,end]
intervals per action query_id -> outputs/submission.csv.

Kept separate from action_pipeline.py (same reasoning as
build_submission.py / build_state_submission.py): re-running the
merge/threshold decision should never require paying for GPU inference
again.

Merging rule: a query_id can have MULTIPLE candidate windows (Stage 1
found several separate bursts of subject presence). Each window that
Qwen3-VL confirmed ("YES start-end", successfully parsed) contributes
one sub-interval in ABSOLUTE video time. Sub-intervals separated by a
short gap are merged (same MAX_GAP_SEC spirit as the other categories);
distinct, far-apart confirmations stay as separate intervals in the
prediction (a real action can plausibly happen more than once).

Writes/updates outputs/submission.csv, merging with any existing
object/state predictions already there rather than overwriting them
(same pattern as build_state_submission.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
QUERIES_PATH = DATA_DIR / "queries.csv"
WINDOWS_PATH = OUTPUT_DIR / "action_candidate_windows.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"


# 5.0s was copied from the object/state categories, where it makes sense
# (0.5s frame sampling, small dropouts). Action's Stage-1 windows are
# spaced ~16s apart (MAX_CLIP_SEC=20 - WINDOW_OVERLAP_SEC=4, in
# action_pipeline.py), so a single window Qwen answers NO on -- while its
# neighbors on both sides answer YES for what is really one continuous
# action -- creates a 16-20s hole, which 5.0s cannot bridge. Confirmed on
# real data (2026-09-15, two independent full-24-query GPU runs, scored
# against user-supplied ground truth with tIoU-F1): a query with one long
# continuous true interval (q007, "a lawn mower cutting the grass",
# true=145-220s) was fragmented into 4 disconnected pieces at 5.0s
# (F1 as low as 0.000-0.133) but became a single correct near-perfect
# match at 16.0s (F1=1.000) in both runs. 16.0s was the empirical optimum
# (or tied for it) on both runs' full gap sweeps (5-100s tested), and it
# is not just a curve-fit: it equals the Stage-1 window step itself, i.e.
# "bridge across exactly one skipped window". Net effect across all 24
# action queries was positive in both runs (mean tIoU-F1 improved), with
# the same trade-off pattern each time: it can over-merge queries whose
# ground truth has multiple genuinely separate short events within ~16s
# of each other (observed on q072, "a cashier is bagging groceries") --
# an acceptable trade given the net gain, but worth knowing about if a
# specific query regresses after this change.
MAX_GAP_SEC = 16.0
BOUNDARY_PAD_SEC = 0.25


def merge_intervals(intervals, max_gap_sec=MAX_GAP_SEC, pad_sec=BOUNDARY_PAD_SEC):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= max_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(max(0.0, s - pad_sec), e + pad_sec) for s, e in merged]


def format_prediction(intervals):
    if not intervals:
        return "NONE"
    return ";".join(f"{s:.2f} {e:.2f}" for s, e in intervals)


def main():
    queries = pd.read_csv(QUERIES_PATH)
    action_query_ids = set(queries[queries.query_type == "action"].query_id)

    if not WINDOWS_PATH.exists():
        raise SystemExit(
            f"{WINDOWS_PATH} not found -- run action_pipeline.py first (on the GPU "
            f"instance with SAM3 + Qwen3-VL), then copy it here."
        )

    windows_df = pd.read_csv(WINDOWS_PATH)

    predictions = {}
    diag = {}
    for qid, g in windows_df.groupby("query_id"):
        confirmed = g.dropna(subset=["parsed_start", "parsed_end"])
        intervals = list(zip(confirmed.parsed_start, confirmed.parsed_end))
        merged = merge_intervals(intervals)
        predictions[qid] = format_prediction(merged)
        diag[qid] = {
            "n_windows": len(g),
            "n_confirmed": len(confirmed),
        }

    if SUBMISSION_PATH.exists():
        existing = pd.read_csv(SUBMISSION_PATH).set_index("query_id")["prediction"].to_dict()
    else:
        existing = {}

    rows = []
    for _, row in queries.iterrows():
        qid = row["query_id"]
        if qid in action_query_ids:
            pred = predictions.get(qid, "NONE")
        else:
            pred = existing.get(qid, "NONE")
        rows.append({"query_id": qid, "prediction": pred})

    pd.DataFrame(rows).to_csv(SUBMISSION_PATH, index=False)

    print(f"Wrote {len(rows)} rows -> {SUBMISSION_PATH}")
    non_none = sum(1 for qid in action_query_ids if predictions.get(qid, "NONE") != "NONE")
    print(f"Action queries with a predicted interval: {non_none}/{len(action_query_ids)}")

    print("\nPer-query action predictions:")
    for qid in sorted(action_query_ids, key=lambda x: int(x[1:])):
        text = queries.loc[queries.query_id == qid, "query_text"].iloc[0]
        d = diag.get(qid, {"n_windows": 0, "n_confirmed": 0})
        print(
            f"  {qid} | {text!r:55} "
            f"(windows={d['n_windows']}, confirmed={d['n_confirmed']}) "
            f"-> {predictions.get(qid, 'NONE')}"
        )


if __name__ == "__main__":
    main()
