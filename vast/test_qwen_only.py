import argparse
import torch


def vram_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--clip-start", type=float, default=0.0)
    parser.add_argument("--clip-end", type=float, default=15.0)
    parser.add_argument(
        "--question",
        default=(
            "At what point in this clip (in seconds from the start of the "
            "clip) does a car drive through the parking lot, if at all? "
            "Answer with a start and end second, or 'none' if it never happens."
        ),
    )
    args = parser.parse_args()
    test_qwen_vl(args.video, args.clip_start, args.clip_end, args.question)
