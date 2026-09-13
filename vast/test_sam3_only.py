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

    inputs = processor(images=image, text=text, return_tensors="pt").to("cuda", torch.float16)

    with torch.inference_mode():
        outputs = model(**inputs)

    print("Output keys:", list(outputs.keys()) if hasattr(outputs, "keys") else type(outputs))
    print("pred_logits shape:", outputs.pred_logits.shape, "max:", outputs.pred_logits.max().item(), "min:", outputs.pred_logits.min().item())
    print("presence_logits:", outputs.presence_logits)
    import torch as _t
    print("pred_logits sigmoid top5:", _t.sigmoid(outputs.pred_logits.flatten()).topk(5))

    try:
        results = processor.post_process_instance_segmentation(
            outputs, target_sizes=[(image.height, image.width)], threshold=0.0
        )[0]
        n_found = len(results.get("masks", results.get("boxes", [])))
        print(f"text prompt: {text!r} -> {n_found} instance(s) found on frame at t={timestamp}s")
        if n_found:
            print("scores:", results.get("scores"))
            print("boxes:", results.get("boxes"))
    except Exception as e:
        print("post_process_instance_segmentation failed:", repr(e))
        print("Available processor methods:", [m for m in dir(processor) if "process" in m.lower()])

    print(f"SAM3 peak VRAM: {vram_mb():.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--text", default="a car")
    args = parser.parse_args()
    test_sam3(args.video, args.time, args.text)
