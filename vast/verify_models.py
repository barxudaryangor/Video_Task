"""
Smoke test -- run this FIRST on the rented instance, before writing any
real pipeline code. Confirms both models actually load, run once, and
reports peak VRAM, so we don't burn rented GPU time debugging import/API
issues inside a half-built pipeline.

Usage (from the project root on the Vast.ai instance):
    python3 vast/verify_models.py --video data/video_02.mp4 --time 10.0 --text "a car"

Expected to print:
  - SAM3: box(es)/mask found for the text prompt on one frame
  - Qwen3-VL-8B: an answer probing its timestamp-grounding ability on a
    short clip (it's specifically trained for text-timestamp alignment,
    so this checks whether it can actually point at seconds, not just
    answer yes/no)
  - Peak VRAM (MB) for each stage, and combined if loaded together
"""

import argparse

import cv2
import torch
from PIL import Image


def vram_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def test_sam3(video_path, timestamp, text):
    from transformers import Sam3Model, Sam3Processor

    print("\n=== SAM3 ===")
    torch.cuda.reset_peak_memory_stats()

    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to("cuda").eval()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at t={timestamp} from {video_path}")

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # NOTE: SAM3's promptable-concept-segmentation call signature is new
    # (added to transformers 2025-11-19) -- if this exact call shape has
    # since changed, check `help(Sam3Processor.__call__)` /
    # https://huggingface.co/docs/transformers/model_doc/sam3 and adjust.
    inputs = processor(images=image, text=text, return_tensors="pt").to("cuda", torch.float16)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]

    n_found = len(results.get("masks", results.get("boxes", [])))
    print(f"text prompt: {text!r} -> {n_found} instance(s) found on frame at t={timestamp}s")
    print(f"SAM3 peak VRAM: {vram_mb():.0f} MB")

    del model
    torch.cuda.empty_cache()
    return n_found


def test_qwen_vl(video_path, start, end, question):
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    print("\n=== Qwen3-VL-8B-Instruct (4-bit) ===")
    torch.cuda.reset_peak_memory_stats()

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model_id = "Qwen/Qwen3-VL-8B-Instruct"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=quant_config, device_map="cuda"
    ).eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "video_start": start,
                    "video_end": end,
                    "fps": 1.0,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=128)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    print(f"question: {question!r}")
    print(f"answer: {answer!r}")
    print(f"Qwen3-VL peak VRAM: {vram_mb():.0f} MB")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--time", type=float, default=10.0, help="Frame timestamp for the SAM3 test.")
    parser.add_argument("--text", default="a car", help="Text prompt for the SAM3 test.")
    parser.add_argument("--clip-start", type=float, default=0.0)
    parser.add_argument("--clip-end", type=float, default=10.0)
    parser.add_argument(
        "--question",
        default=(
            "At what point in this clip (in seconds from the start of the "
            "clip) does a car drive through the parking lot, if at all? "
            "Answer with a start and end second, or 'none' if it never happens."
        ),
        help="Default probes Qwen3-VL's timestamp-grounding ability, not just yes/no.",
    )
    args = parser.parse_args()

    test_sam3(args.video, args.time, args.text)
    test_qwen_vl(args.video, args.clip_start, args.clip_end, args.question)

    print(f"\nCombined peak VRAM this run: {vram_mb():.0f} MB (models were unloaded between stages)")
