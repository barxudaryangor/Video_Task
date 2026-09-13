"""
For every CONFIRMED action window in an action_candidate_windows.csv
(parsed_start/parsed_end not NaN), extract 3 frames -- at the predicted
start, middle, and end timestamps -- so a human can visually confirm
Qwen3-VL's answer actually matches what's happening in the video.

Saved to outputs/action_frame_checks/<query_id>/<query_id>_<start>-<end>_<tag>.jpg
(tag is one of "start", "mid", "end"), with the timestamp burned into the
top-left corner of each frame.

This is a read-only diagnostic step -- it does not touch submission.csv
or any threshold/merge logic.
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FRAME_CHECK_DIR = OUTPUT_DIR / "action_frame_checks"


def extract_frame(video_path, timestamp_sec):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def save_labeled_frame(frame, timestamp_sec, out_path):
    label = f"t={timestamp_sec:.1f}s"
    cv2.putText(frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-csv", default=str(OUTPUT_DIR / "action_candidate_windows.csv"))
    parser.add_argument("--video-id", default=None, help="Only process this video_id (e.g. video_01)")
    args = parser.parse_args()

    windows_path = Path(args.windows_csv)
    if not windows_path.exists():
        raise SystemExit(f"{windows_path} not found")

    df = pd.read_csv(windows_path)
    confirmed = df.dropna(subset=["parsed_start", "parsed_end"])
    if args.video_id:
        confirmed = confirmed[confirmed.video_id == args.video_id]

    if confirmed.empty:
        print("No confirmed (YES) windows to extract frames for.")
        return

    FRAME_CHECK_DIR.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for _, row in confirmed.iterrows():
        qid = row["query_id"]
        video_id = row["video_id"]
        start = float(row["parsed_start"])
        end = float(row["parsed_end"])
        mid = (start + end) / 2

        video_path = DATA_DIR / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"  [skip] {video_path} not found")
            continue

        query_dir = FRAME_CHECK_DIR / qid
        query_dir.mkdir(parents=True, exist_ok=True)

        for tag, ts in (("start", start), ("mid", mid), ("end", end)):
            frame = extract_frame(video_path, ts)
            if frame is None:
                print(f"  [warn] {qid}: could not read frame at t={ts:.1f}s")
                continue
            out_path = query_dir / f"{qid}_{start:.1f}-{end:.1f}_{tag}.jpg"
            save_labeled_frame(frame, ts, out_path)
            n_saved += 1

        print(f"  {qid} ({video_id}) [{start:.1f},{end:.1f}] -> frames saved in {query_dir}")

    print(f"\nSaved {n_saved} frame(s) across {confirmed.query_id.nunique()} confirmed window(s) "
          f"-> {FRAME_CHECK_DIR}")


if __name__ == "__main__":
    main()
