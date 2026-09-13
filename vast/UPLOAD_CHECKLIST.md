# Vast.ai rental + upload checklist

## 1. What to rent (do this yourself on vast.ai -- I can't pay/authenticate for you)

- GPU: **RTX 3090 (24GB VRAM)** -- currently ~$0.06-0.10/hr on Vast.ai, cheaper
  and roomier than a 16GB card for running SAM3 + Qwen3-VL-8B together.
- Template: **PyTorch** (comes with CUDA/cuDNN preinstalled).
- Disk: **80GB+** (SAM3 + Qwen3-VL-8B weights + the 8 source videos + working space).
- Given the project's ~$12 GPU budget note in PROJECT_STATUS.md: at $0.08/hr
  that's ~150 hours of runtime available -- plenty for iterating, as long as
  you remember to **stop the instance** when not actively using it (billing
  is per-hour while running, on-demand instances bill even when idle).

## 2. Once the instance is up

1. SSH in (Vast.ai gives you the exact command on the instance page).
2. Upload this project's `src/`, `data/` (the 8 videos + queries.csv), and
   `vast/` directories, e.g. from your local machine:
   ```
   scp -r -P <port> "src" "data" "vast" root@<instance-ip>:/workspace/
   ```
   (videos are the bulk of the transfer -- check total size first with
   `du -sh data/*.mp4` locally; if it's large, `rsync -avz --progress` resumes
   better than `scp` on a flaky connection.)
3. On the instance: `bash vast/vast_setup.sh`
4. Smoke test BEFORE writing more pipeline code:
   ```
   python3 vast/verify_models.py --video data/video_02.mp4 --time 10.0 --text "a car"
   ```
   Confirm both models load, produce a sane answer, and check the printed
   VRAM numbers actually fit the card.

## 3. After the smoke test passes

Come back and we design `src/action_pipeline.py` together against
whatever verify_models.py actually showed (real API behavior, real VRAM
numbers, real answer quality) rather than against assumptions.

## 4. When done for the session

**Stop (not just disconnect from) the instance from the Vast.ai console** --
an on-demand instance keeps billing while running even if you're not
connected to it.
