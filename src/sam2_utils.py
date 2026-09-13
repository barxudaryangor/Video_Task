"""
SAM2 image-mode mask prediction, prompted by a Grounding DINO box.

Deliberately IMAGE mode (Sam2Model + Sam2Processor), not video mode
(Sam2VideoModel). Both object_pipeline.py and state_pipeline.py already
process sampled frames one at a time in a simple loop -- there is no
persistent per-video inference session to hook a video-tracking API into
without a much larger restructure of that loop. Image-mode, prompted fresh
with the current frame's Grounding DINO box each time, gets the actual fix
this step needs (a precise object mask instead of a rectangle that mixes in
background pixels) without that restructure. Cross-frame continuity (does
this look like the same instance as last time, is it moving) is handled
separately, from the resulting per-frame boxes/masks, not by SAM2 itself.

This module only knows how to turn one (image, box) pair into one mask. It
does not decide when to call SAM2, what to do with the mask, or what
"detection" means -- that stays in object_pipeline.py / state_pipeline.py,
unchanged for the existing rectangular-crop path.
"""

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

SAM2_MODEL_NAME = "facebook/sam2-hiera-base-plus"

# Gray, not black: a black background can itself read as a dark "color"
# to a pixel-level color check or shift a SigLIP embedding, exactly the
# kind of contamination this whole SAM2 step exists to remove.
NEUTRAL_BG_COLOR = (128, 128, 128)

MASK_CROP_MARGIN = 0.10


def load_sam2(device, dtype=torch.float32):
    """Loads the SAM2 image processor + model once; reuse the returned
    pair across all frames/videos rather than reloading per call."""
    processor = Sam2Processor.from_pretrained(SAM2_MODEL_NAME)
    model = Sam2Model.from_pretrained(SAM2_MODEL_NAME, dtype=dtype).to(device).eval()
    return processor, model


def predict_mask(image, box, processor, model, device):
    """
    image: PIL.Image, full frame, RGB.
    box:   [x0, y0, x1, y1] in image pixel coordinates -- the Grounding
           DINO detection box, used as SAM2's box prompt.

    Returns:
        mask:      np.ndarray[bool], shape (H, W), full image size.
        mask_bbox: (x0, y0, x1, y1) tight box around the True pixels,
                   or None if SAM2 returned an empty mask.
        mask_area: int, number of True pixels (0 if empty).
    """
    input_boxes = [[[float(v) for v in box]]]

    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
    )
    mask = masks[0][0, 0].numpy().astype(bool)  # (H, W), original image size

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask, None, 0

    mask_bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    mask_area = int(mask.sum())
    return mask, mask_bbox, mask_area


def masked_crop_with_margin(image, mask, mask_bbox, margin_ratio=MASK_CROP_MARGIN):
    """
    Crops the region around mask_bbox (+margin, so the crop isn't cut
    exactly at the mask edge) and replaces every pixel OUTSIDE the mask
    with a neutral gray fill -- this is what a color check or SigLIP
    comparison should see instead of the raw rectangular crop, since it
    excludes whatever background the bounding box happened to include.

    Returns None if mask_bbox is None (SAM2 found nothing) or the
    resulting crop region is degenerate.
    """
    if mask_bbox is None:
        return None

    width, height = image.size
    x0, y0, x1, y1 = mask_bbox
    bw, bh = x1 - x0, y1 - y0
    mx, my = bw * margin_ratio, bh * margin_ratio

    cx0 = max(0, int(x0 - mx))
    cy0 = max(0, int(y0 - my))
    cx1 = min(width, int(x1 + mx))
    cy1 = min(height, int(y1 + my))

    if cx1 <= cx0 or cy1 <= cy0:
        return None

    img_arr = np.array(image.convert("RGB"))
    bg = np.full_like(img_arr, NEUTRAL_BG_COLOR, dtype=np.uint8)
    composited = np.where(mask[..., None], img_arr, bg)

    composited_img = Image.fromarray(composited)
    return composited_img.crop((cx0, cy0, cx1, cy1))
