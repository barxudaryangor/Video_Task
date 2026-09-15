"""
Action category pipeline (24 "action" queries).

Two stages, deliberately kept separate so the expensive stage only ever
runs on a small, pre-filtered slice of each video:

  Stage 1 (SAM3, cheap, runs on every sampled frame):
    For each query, extract a short subject noun phrase (see
    action_utils.extract_action_subject -- e.g. "a car drives through
    the parking lot" -> "a car") and ask SAM3 whether that subject is
    present in the frame (SAM3's own presence_logits, sigmoid-scored).
    Frames where the subject is present are merged (bridging short
    gaps) into candidate time windows, padded on both sides so Qwen3-VL
    gets surrounding context rather than a razor-thin clip.

  Stage 2 (Qwen3-VL-8B, expensive, runs ONLY on candidate windows):
    For each candidate window, ask Qwen3-VL the actual query text
    ("does this action happen here, and if so, when") and parse out a
    start/end second *within the clip*, converting back to absolute
    video time.

This file only collects evidence -- outputs/action_candidate_windows.csv
(one row per candidate window per query, with SAM3 presence stats and
Qwen3-VL's raw+parsed answer). It does not touch submission.csv; see
build_action_submission.py for that (kept separate, same pattern as
object/state, so the decision step can be re-run without re-paying for
GPU inference).
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import action_utils

# decord 0.6.0's default multi-threaded decoder has a known race
# ("Check failed: avcodec_send_packet(...) >= 0 (-11 vs. 0)") on certain
# H.264 streams (seen on video_04.mp4). Forcing single-threaded decoding
# is the standard workaround. This must happen before qwen_vl_utils's
# _read_video_decord (which does `decord.VideoReader(video_path)` with no
# kwargs) ever constructs a VideoReader, so patch the class at import time.
try:
    import decord

    _OrigDecordVideoReader = decord.VideoReader

    class _SingleThreadedVideoReader(_OrigDecordVideoReader):
        def __init__(self, uri, *args, **kwargs):
            kwargs.setdefault("num_threads", 1)
            super().__init__(uri, *args, **kwargs)

    decord.VideoReader = _SingleThreadedVideoReader
except ImportError:
    pass

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
QUERIES_PATH = DATA_DIR / "queries.csv"

# ------------------------------------------------------------
# Stage 1 (SAM3) config
# ------------------------------------------------------------
SAMPLE_INTERVAL_SEC = 1.0     # coarser than object/state (0.5s): Stage 1 only
                              # needs to find candidate windows, not pin exact
                              # frames -- Stage 2 does the precise work.
PRESENCE_THRESHOLD = 0.5      # sigmoid(presence_logits) >= this -> "subject present"
MAX_GAP_SEC = 10.0            # bridge short dropouts when merging presence frames
                              # into windows (subjects flicker in/out of a coarse
                              # detector more than a clean "is it there" signal would)
WINDOW_PAD_SEC = 5.0          # context padding added to each candidate window
MAX_CLIP_SEC = 20.0           # cap a single Qwen3-VL clip length (cost control) --
                              # long real events still get located, just via
                              # multiple sub-windows rather than one huge clip
WINDOW_OVERLAP_SEC = 4.0      # overlap between consecutive sub-windows when a raw
                              # presence span exceeds MAX_CLIP_SEC, so an action
                              # straddling a chunk boundary isn't cut in half and
                              # missed by both neighboring chunks

# ------------------------------------------------------------
# Stage 2 (Qwen3-VL) config
# ------------------------------------------------------------
QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_SAMPLE_FPS = 2.0          # frames/sec Qwen3-VL samples from each clip (configurable)

# IMPORTANT: Qwen3-VL's processor embeds each sampled frame's own ABSOLUTE
# timestamp (from the source video) into the prompt as "<12.3 seconds>"
# tokens (see transformers Qwen3VLProcessor.replace_video_token, which
# calls _calculate_timestamps using the real frame indices / source fps --
# this is only correct when real video_metadata is supplied, which is
# exactly the bug being fixed here). Since the model is shown ABSOLUTE
# timestamps per frame, we ask it to answer in that SAME absolute
# timescale rather than "seconds since clip start" -- asking for a
# different convention than what it's shown risks the model just
# echoing back the timestamps it saw, mislabeled.
QWEN_PROMPT_TEMPLATE = (
    "You are given a short video clip from a static surveillance camera. "
    "Each frame shown to you is labeled with its own timestamp in seconds, e.g. "
    "\"<123.4 seconds>\". "
    "Question: does the following happen at any point in this clip: \"{query_text}\"? "
    "If yes, respond in EXACTLY this format: \"YES start-end\" where start and end "
    "are the ACTUAL TIMESTAMPS IN SECONDS matching the frame labels you were shown "
    "(do NOT measure from the start of the clip -- use the same absolute timestamps "
    "printed on the frames), marking when the action happens. "
    "If it does not happen anywhere in this clip, respond with EXACTLY: \"NO\"."
)
ANSWER_RE = re.compile(r"YES\s+([\d.]+)\s*-\s*([\d.]+)", re.IGNORECASE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Stage 1: SAM3 presence scoring
# ============================================================

def load_sam3():
    from transformers import Sam3Model, Sam3Processor
    print(f"Loading SAM3...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(DEVICE).eval()
    print("SAM3 loaded.")
    return processor, model


def sam3_presence_score(image, text, processor, model):
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE, torch.float16)
    with torch.inference_mode():
        outputs = model(**inputs)
    return torch.sigmoid(outputs.presence_logits).item()


def _chunk_span(start, end):
    """
    Split [start, end] into <=MAX_CLIP_SEC windows with WINDOW_OVERLAP_SEC
    overlap between consecutive pieces (never a single window wider than
    MAX_CLIP_SEC, so a long span straddling the limit isn't silently
    truncated). Shared by the normal presence-merge path and the
    zero-window fallback below, which both need to turn one long span
    into Qwen-sized clips the same way.
    """
    span = end - start
    if span <= MAX_CLIP_SEC:
        return [(start, end)]
    step = MAX_CLIP_SEC - WINDOW_OVERLAP_SEC
    assert step > 0, "WINDOW_OVERLAP_SEC must be smaller than MAX_CLIP_SEC"
    result = []
    c_start = start
    while True:
        c_end = min(end, c_start + MAX_CLIP_SEC)
        result.append((c_start, c_end))
        if c_end >= end:
            break
        c_start += step
    return result


def stage1_candidate_windows(video_path, query_ids, query_texts, subjects, processor, model, debug_query=None):
    """
    Samples the video every SAMPLE_INTERVAL_SEC and scores SAM3 presence
    for each query's subject phrase. Returns:
      windows: {query_id: [(start, end), ...]}
      presence_rows: list of dicts (one per query per sampled frame) for
        the debug CSV.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    times = []
    presence = {qid: [] for qid in query_ids}

    frame_index = 0
    next_sample_time = 0.0
    progress = tqdm(total=frame_count, desc=f"{video_path.stem} SAM3")

    while True:
        success, frame = cap.read()
        if not success:
            break
        target_frame = round(next_sample_time * fps)
        if frame_index >= target_frame:
            timestamp = frame_index / fps
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            times.append(timestamp)
            for qid, subject in zip(query_ids, subjects):
                if debug_query and qid != debug_query:
                    presence[qid].append(0.0)
                    continue
                score = sam3_presence_score(image, subject, processor, model)
                presence[qid].append(score)
            next_sample_time += SAMPLE_INTERVAL_SEC
        frame_index += 1
        progress.update(1)
    progress.close()
    cap.release()

    times = np.asarray(times)
    video_end = float(times[-1]) if len(times) else 0.0

    windows = {}
    presence_rows = []
    for qid in query_ids:
        scores = np.asarray(presence[qid])
        for t, s in zip(times, scores):
            presence_rows.append({"query_id": qid, "time": float(t), "sam3_presence": float(s)})

        above = scores >= PRESENCE_THRESHOLD
        idxs = np.where(above)[0]
        if len(idxs) == 0:
            # SAM3 presence never crossed the threshold anywhere in the
            # whole video. Confirmed (2026-09-15, real data) that this is
            # NOT always genuine absence: for some subject phrases (e.g.
            # "robbers", "a tractor") the score stays low across frames
            # where the object is plausibly/confirmedly on screen -- a
            # SAM3 vocabulary/grounding gap, not evidence of nothing
            # happening. Silently emitting zero windows here means an
            # automatic NONE with no Stage-2 check at all, which is a
            # costly wrong guess if the query is actually non-NONE.
            # Falling back to a full-video scan costs exactly what the
            # "saturated presence" case below already pays per query, so
            # this isn't a new expense class -- it just stops the weakest
            # Stage-1 signal from being the sole word on these queries.
            print(f"    [Stage1 fallback] {qid}: SAM3 presence never reached "
                  f"{PRESENCE_THRESHOLD} (max={scores.max():.3f}) -- falling back "
                  f"to full-video scan instead of auto-NONE")
            windows[qid] = _chunk_span(0.0, video_end)
            continue

        merged = [[idxs[0], idxs[0]]]
        for idx in idxs[1:]:
            if times[idx] - times[merged[-1][1]] <= MAX_GAP_SEC:
                merged[-1][1] = idx
            else:
                merged.append([idx, idx])

        result = []
        for s_idx, e_idx in merged:
            start = max(0.0, float(times[s_idx]) - WINDOW_PAD_SEC)
            end = min(video_end, float(times[e_idx]) + WINDOW_PAD_SEC)
            # A long continuous presence stretch (e.g. a parked car visible
            # for two minutes) must NOT be collapsed to one centered
            # MAX_CLIP_SEC slice -- that silently throws away most of the
            # span, and the real action could be anywhere in it.
            result.extend(_chunk_span(start, end))
        windows[qid] = result

    return windows, presence_rows


# ============================================================
# Stage 2: Qwen3-VL temporal grounding
# ============================================================

def load_qwen():
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    print("Loading Qwen3-VL-8B-Instruct...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        QWEN_MODEL_ID, quantization_config=quant_config, device_map=DEVICE
    ).eval()
    print("Qwen3-VL loaded.")
    return processor, model


def parse_and_validate_answer(answer, window_start, window_end, tolerance_sec=1.0):
    """
    'YES 151.6-159.8' -> (abs_start, abs_end), where the numbers are
    ALREADY absolute video seconds (matching the per-frame timestamp
    labels Qwen3-VL was shown -- see QWEN_PROMPT_TEMPLATE). No offset
    arithmetic is needed; this only validates and clamps.

    Returns None for 'NO', anything unparseable, or a malformed range
    (end <= start, or wildly outside the window) -- never silently
    accepts garbage.
    """
    match = ANSWER_RE.search(answer)
    if not match:
        return None
    a, b = float(match.group(1)), float(match.group(2))
    if b <= a:
        return None
    # Allow a small tolerance for the model rounding to a frame's own
    # timestamp that lands just outside the requested window, but
    # reject anything further off as a malformed/hallucinated answer.
    if a < window_start - tolerance_sec or b > window_end + tolerance_sec:
        return None
    a = max(a, window_start)
    b = min(b, window_end)
    if b <= a:
        return None
    return (a, b)


def qwen_ground_window(video_path, start, end, query_text, processor, model, sample_fps=QWEN_SAMPLE_FPS):
    from qwen_vl_utils import process_vision_info
    from transformers.video_utils import VideoMetadata

    clip_duration = end - start
    question = QWEN_PROMPT_TEMPLATE.format(query_text=query_text)
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": str(video_path),
                "video_start": start,
                "video_end": end,
                "fps": sample_fps,
            },
            {"type": "text", "text": question},
        ],
    }]

    # return_video_metadata=True is the fix: without it, qwen_vl_utils
    # still samples the correct frames internally but THROWS AWAY the
    # fps/frame-index metadata describing them, so the processor can't
    # tell how much real time those frames span and silently falls back
    # to assuming 24fps (the bug this whole function exists to fix).
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True
    )
    video_tensor, meta_dict = video_inputs[0]
    metadata = VideoMetadata(**meta_dict)

    n_frames = int(video_tensor.shape[0])
    frames_indices = list(metadata.frames_indices) if metadata.frames_indices is not None else []
    src_fps = metadata.fps
    first_ts = frames_indices[0] / src_fps if frames_indices and src_fps else None
    last_ts = frames_indices[-1] / src_fps if frames_indices and src_fps else None
    perceived_duration = (last_ts - first_ts) if (first_ts is not None and last_ts is not None) else None
    effective_sample_fps = (n_frames / clip_duration) if clip_duration > 0 else float("nan")

    print(f"      [Qwen diag] window abs=({start:.2f},{end:.2f}) expected_duration={clip_duration:.2f}s")
    print(f"      [Qwen diag] source_video_fps={src_fps} sampled_frames={n_frames} "
          f"requested_sample_fps={sample_fps} effective_sample_fps={effective_sample_fps:.3f}")
    print(f"      [Qwen diag] metadata.total_num_frames={metadata.total_num_frames} "
          f"frames_indices[:3]={frames_indices[:3]} frames_indices[-3:]={frames_indices[-3:]}")
    print(f"      [Qwen diag] first_frame_ts={first_ts} last_frame_ts={last_ts} "
          f"perceived_duration~={perceived_duration}")

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=image_inputs, videos=[video_tensor],
        video_metadata=[metadata], do_sample_frames=False,
        padding=True, return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=64)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    parsed_abs = parse_and_validate_answer(answer, start, end)

    diag = {
        "source_video_fps": src_fps,
        "sampled_frames": n_frames,
        "requested_sample_fps": sample_fps,
        "effective_sample_fps": effective_sample_fps,
        "metadata_total_num_frames": metadata.total_num_frames,
        "first_frame_ts": first_ts,
        "last_frame_ts": last_ts,
        "perceived_duration": perceived_duration,
    }
    return answer, parsed_abs, diag


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-query", type=str, default=None, help="Restrict to one query_id (its video only).")
    args = parser.parse_args()

    queries = pd.read_csv(QUERIES_PATH)
    action_queries = queries[queries.query_type == "action"].copy()
    if args.debug_query:
        action_queries = action_queries[action_queries.query_id == args.debug_query].copy()
        if action_queries.empty:
            raise SystemExit(f"--debug-query {args.debug_query!r} is not an action query_id")

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.debug_query:
        presence_path = OUTPUT_DIR / f"action_presence_debug_{args.debug_query}.csv"
        windows_path = OUTPUT_DIR / f"action_candidate_windows_debug_{args.debug_query}.csv"
    else:
        presence_path = OUTPUT_DIR / "action_presence.csv"
        windows_path = OUTPUT_DIR / "action_candidate_windows.csv"

    # Write results incrementally, one video at a time, instead of only once
    # at the very end -- a crash partway through (e.g. a bad video decode on
    # a later video) must not throw away GPU work already done on earlier
    # videos. Each video's rows are appended to the CSV as soon as that
    # video finishes.
    presence_path.write_text("")
    windows_path.write_text("")
    presence_header_written = False
    windows_header_written = False

    sam3_processor, sam3_model = load_sam3()
    qwen_processor, qwen_model = load_qwen()

    total_presence_rows = 0
    total_window_rows = 0

    video_ids = sorted(action_queries.video_id.unique())
    for video_id in video_ids:
        video_path = DATA_DIR / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        video_queries = action_queries[action_queries.video_id == video_id].reset_index(drop=True)
        query_ids = video_queries.query_id.tolist()
        query_texts = video_queries.query_text.tolist()
        subjects = [action_utils.extract_action_subject(t) for t in query_texts]

        print(f"\n=== {video_id} ===")
        for qid, text, subj in zip(query_ids, query_texts, subjects):
            print(f"  {qid} | {text!r} | Stage-1 subject: {subj!r}")

        windows, presence_rows = stage1_candidate_windows(
            video_path, query_ids, query_texts, subjects,
            sam3_processor, sam3_model,
            debug_query=args.debug_query,
        )
        for row in presence_rows:
            row["video_id"] = video_id

        video_window_rows = []
        for qid, text in zip(query_ids, query_texts):
            qid_windows = windows.get(qid, [])
            print(f"  {qid}: {len(qid_windows)} candidate window(s) from Stage 1: "
                  f"{[(round(s,1), round(e,1)) for s,e in qid_windows]}")
            for w_start, w_end in qid_windows:
                try:
                    answer, parsed_abs, diag = qwen_ground_window(
                        video_path, w_start, w_end, text, qwen_processor, qwen_model
                    )
                except Exception as e:
                    # A single window failing to decode/infer (e.g. a corrupt
                    # packet in this stretch of video) must not take down the
                    # whole run -- log it, record it as unconfirmed, move on.
                    print(f"    window [{w_start:.1f},{w_end:.1f}] -> ERROR ({type(e).__name__}: {e}) "
                          f"-- treating as unconfirmed, continuing")
                    answer, parsed_abs, diag = f"ERROR: {e}", None, {}
                else:
                    print(f"    window [{w_start:.1f},{w_end:.1f}] -> Qwen3-VL raw: {answer!r} "
                          f"-> validated absolute interval={parsed_abs}")
                video_window_rows.append({
                    "video_id": video_id,
                    "query_id": qid,
                    "query_text": text,
                    "window_start": w_start,
                    "window_end": w_end,
                    "qwen_raw_answer": answer,
                    "parsed_start": parsed_abs[0] if parsed_abs else np.nan,
                    "parsed_end": parsed_abs[1] if parsed_abs else np.nan,
                    **{f"diag_{k}": v for k, v in diag.items()},
                })

        presence_df = pd.DataFrame(presence_rows)
        presence_df.to_csv(presence_path, mode="a", header=not presence_header_written, index=False)
        presence_header_written = True
        total_presence_rows += len(presence_df)

        windows_df = pd.DataFrame(video_window_rows)
        windows_df.to_csv(windows_path, mode="a", header=not windows_header_written, index=False)
        windows_header_written = True
        total_window_rows += len(windows_df)
        print(f"  -> appended {len(presence_df)} presence row(s) and {len(windows_df)} window row(s) "
              f"for {video_id} to disk")

        torch.cuda.empty_cache()

    print(f"\nSaved SAM3 presence -> {presence_path} ({total_presence_rows} rows)")
    print(f"Saved candidate windows + Qwen3-VL answers -> {windows_path} ({total_window_rows} rows)")


if __name__ == "__main__":
    main()
