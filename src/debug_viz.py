"""
Debug visualization for the object/state pipelines' SAM2 integration.

Produces, per saved detection, four individual PNGs plus one combined
contact sheet, all under:

    outputs/frame_checks/{category}/{query_id}/

named:

    {query_id}_t{timestamp}_det{NN}_context.png
    {query_id}_t{timestamp}_det{NN}_bbox_crop.png
    {query_id}_t{timestamp}_det{NN}_sam_mask.png
    {query_id}_t{timestamp}_det{NN}_masked_crop.png
    {query_id}_t{timestamp}_det{NN}_debug.png          (combined 2x2)

plus one row appended to outputs/frame_checks/debug_index.csv per saved
detection, so filenames alone never have to carry every detail.

This module only draws and writes files -- it does not decide WHICH
frames are worth saving (that policy lives in the calling pipeline,
see DebugPolicy below) and does not compute any of the scores/masks it
draws (those come from sam2_utils.py and the pipeline's own SigLIP calls).
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_ROOT = Path("outputs") / "frame_checks"
INDEX_PATH = OUT_ROOT / "debug_index.csv"

MASK_OVERLAY_COLOR = (255, 40, 40)
BBOX_COLOR = (255, 210, 0)
CENTER_COLOR = (0, 220, 255)

INDEX_COLUMNS = [
    "category", "query_id", "video_id", "timestamp", "frame_idx",
    "detection_id", "track_id", "full_query", "automatic_anchor",
    "detector_prompt", "detector_score", "detector_bbox", "sam_bbox",
    "sam_mask_area", "siglip_bbox_score", "siglip_masked_score",
    "expected_color", "color_score", "center_x_norm", "center_y_norm",
    "speed", "context_png", "bbox_crop_png", "sam_mask_png",
    "masked_crop_png", "combined_debug_png",
]


def _font(size=14):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def format_timestamp(t):
    """Filename-safe, fixed-width: 12.5 -> '0012.50'."""
    return f"{t:07.2f}"


def _detection_dir(category, query_id):
    d = OUT_ROOT / category / query_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _filename(query_id, timestamp, detection_id, suffix):
    return f"{query_id}_t{format_timestamp(timestamp)}_det{detection_id:02d}_{suffix}.png"


def _metadata_lines(meta):
    """meta is a dict of label -> value; None/missing values are omitted
    rather than shown as 'None' or invented."""
    lines = []
    for label, value in meta.items():
        if value is None or value == "":
            continue
        lines.append(f"{label}: {value}")
    return lines


def _draw_text_block(draw, xy, lines, font, fill=(255, 255, 255), bg=(0, 0, 0)):
    x, y = xy
    for line in lines:
        bbox = draw.textbbox((x, y), line, font=font)
        pad = 2
        draw.rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            fill=bg,
        )
        draw.text((x, y), line, font=font, fill=fill)
        y = bbox[3] + pad + 2


def draw_mask_overlay(image, mask, box, margin=0.15):
    """Crops around box (+margin) and paints a translucent mask overlay
    on top, so it's immediately visible which pixels SAM2 kept."""
    width, height = image.size
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    mx, my = bw * margin, bh * margin
    cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    cx1, cy1 = min(width, int(x1 + mx)), min(height, int(y1 + my))
    if cx1 <= cx0 or cy1 <= cy0:
        cx0, cy0, cx1, cy1 = 0, 0, width, height

    img_arr = np.array(image.convert("RGB")).astype(np.float32)
    overlay = img_arr.copy()
    if mask is not None:
        color = np.array(MASK_OVERLAY_COLOR, dtype=np.float32)
        overlay[mask] = overlay[mask] * 0.45 + color * 0.55
    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    return overlay_img.crop((cx0, cy0, cx1, cy1))


SPEED_ARROW_COLOR = (255, 120, 255)
# Per-frame dx/dy (normalized, fraction of frame width/height) are tiny at
# typical object speeds -- this purely-visual multiplier makes the arrow
# legible; it exaggerates length, never direction, and is not used for
# anything but drawing.
SPEED_ARROW_SCALE = 10.0


def draw_context_image(image, box, mask, meta, velocity=None):
    """Full frame + DINO bbox + mask overlay + center + optional speed
    arrow + a readable metadata text block (which may include a Track
    line). `meta` values that don't apply should be left out of the dict
    entirely (see _metadata_lines).

    velocity: optional (dx, dy) in NORMALIZED units (see
    tracking_utils.QueryTracker) -- drawn as an arrow from the box center
    in the direction of travel. None (or either component None) draws no
    arrow, e.g. on the first frame of a track when there's no prior
    position to compare against."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")

    if mask is not None:
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_rgba[mask] = (*MASK_OVERLAY_COLOR, 110)
        img.paste(Image.fromarray(mask_rgba, "RGBA"), (0, 0), Image.fromarray(mask_rgba, "RGBA"))

    if box is not None and all(np.isfinite(v) for v in box):
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1, y1), outline=BBOX_COLOR, width=3)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        if velocity is not None and velocity[0] is not None and velocity[1] is not None:
            dx, dy = velocity
            ex = cx + dx * img.width * SPEED_ARROW_SCALE
            ey = cy + dy * img.height * SPEED_ARROW_SCALE
            if abs(ex - cx) > 1 or abs(ey - cy) > 1:
                draw.line((cx, cy, ex, ey), fill=SPEED_ARROW_COLOR, width=3)
                draw.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), fill=SPEED_ARROW_COLOR)

        r = 5
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=CENTER_COLOR)

    _draw_text_block(draw, (8, 8), _metadata_lines(meta), _font(15))
    return img


def make_combined_sheet(context_img, bbox_crop_img, mask_overlay_img, masked_crop_img, header_lines):
    """2x2: [frame+bbox | bbox_crop] / [mask overlay | masked_crop], with
    a text header. Cells keep their own aspect ratio, only downscaled if
    larger than the panel budget, to avoid making small objects illegible."""
    panel_w, panel_h = 480, 420
    header_h = 22 * (len(header_lines) + 1)

    MAX_UPSCALE = 6.0

    def fit(img):
        img = img.convert("RGB")
        scale = min(panel_w / img.width, panel_h / img.height)
        # Downscale smoothly (LANCZOS); upscale with NEAREST, capped, so a
        # small crop actually fills its panel instead of sitting tiny in a
        # sea of background -- without inventing smoothed-in fake detail
        # that could mislead a color/pixel inspection.
        if scale >= 1.0:
            scale = min(scale, MAX_UPSCALE)
            resample = Image.NEAREST
        else:
            resample = Image.LANCZOS
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        resized = img.resize(new_size, resample)
        canvas = Image.new("RGB", (panel_w, panel_h), (32, 32, 32))
        off = ((panel_w - new_size[0]) // 2, (panel_h - new_size[1]) // 2)
        canvas.paste(resized, off)
        return canvas

    sheet = Image.new("RGB", (panel_w * 2, panel_h * 2 + header_h), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    _draw_text_block(draw, (8, 6), header_lines, _font(16), bg=(16, 16, 16))

    sheet.paste(fit(context_img), (0, header_h))
    sheet.paste(fit(bbox_crop_img), (panel_w, header_h))
    sheet.paste(fit(mask_overlay_img), (0, header_h + panel_h))
    sheet.paste(fit(masked_crop_img), (panel_w, header_h + panel_h))
    return sheet


def _append_index_row(row):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    is_new = not INDEX_PATH.exists()
    with open(INDEX_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in INDEX_COLUMNS})


def save_detection_debug(
    category,
    query_id,
    video_id,
    timestamp,
    frame_idx,
    detection_id,
    image,
    box,
    mask,
    bbox_crop,
    masked_crop,
    fields,
):
    """
    Saves all 5 PNGs for one detection and appends a row to debug_index.csv.

    fields: dict that may include any of detector_prompt, full_query,
    automatic_anchor, detector_score, sam_bbox, sam_mask_area,
    siglip_bbox_score, siglip_masked_score, expected_color, color_score,
    track_id, center_x_norm, center_y_norm, speed. Missing keys are left
    blank in the CSV and omitted from the on-image text -- never invented.

    Returns the debug_index.csv row (dict) that was written.
    """
    out_dir = _detection_dir(category, query_id)

    meta_for_image = {
        "Query": query_id,
        "Text": fields.get("full_query"),
        "Anchor": fields.get("automatic_anchor") or fields.get("detector_prompt"),
        "DINO": _fmt(fields.get("detector_score")),
        "SigLIP bbox": _fmt(fields.get("siglip_bbox_score")),
        "SigLIP mask": _fmt(fields.get("siglip_masked_score")),
        "Color": _color_label(fields),
        "Track": fields.get("track_id"),
        "Speed": _fmt(fields.get("speed")),
    }

    velocity = (fields.get("dx"), fields.get("dy"))
    context_img = draw_context_image(image, box, mask, meta_for_image, velocity=velocity)
    mask_overlay_img = draw_mask_overlay(image, mask, box if box is not None else (0, 0, image.width, image.height))

    context_name = _filename(query_id, timestamp, detection_id, "context")
    bbox_crop_name = _filename(query_id, timestamp, detection_id, "bbox_crop")
    sam_mask_name = _filename(query_id, timestamp, detection_id, "sam_mask")
    masked_crop_name = _filename(query_id, timestamp, detection_id, "masked_crop")
    combined_name = _filename(query_id, timestamp, detection_id, "debug")

    context_img.save(out_dir / context_name)
    if bbox_crop is not None:
        bbox_crop.convert("RGB").save(out_dir / bbox_crop_name)
    mask_overlay_img.save(out_dir / sam_mask_name)
    if masked_crop is not None:
        masked_crop.convert("RGB").save(out_dir / masked_crop_name)

    header = [
        f"{query_id} @ t={timestamp:.2f}s  det{detection_id:02d}  video={video_id}",
    ] + _metadata_lines(meta_for_image)

    combined = make_combined_sheet(
        context_img,
        bbox_crop if bbox_crop is not None else Image.new("RGB", (10, 10), (64, 64, 64)),
        mask_overlay_img,
        masked_crop if masked_crop is not None else Image.new("RGB", (10, 10), (64, 64, 64)),
        header,
    )
    combined.save(out_dir / combined_name)

    rel = lambda p: str((out_dir / p).as_posix())

    row = {
        "category": category,
        "query_id": query_id,
        "video_id": video_id,
        "timestamp": f"{timestamp:.3f}",
        "frame_idx": frame_idx,
        "detection_id": detection_id,
        "track_id": fields.get("track_id", ""),
        "full_query": fields.get("full_query", ""),
        "automatic_anchor": fields.get("automatic_anchor", ""),
        "detector_prompt": fields.get("detector_prompt", ""),
        "detector_score": fields.get("detector_score", ""),
        "detector_bbox": _bbox_str(box),
        "sam_bbox": _bbox_str(fields.get("sam_bbox")),
        "sam_mask_area": fields.get("sam_mask_area", ""),
        "siglip_bbox_score": fields.get("siglip_bbox_score", ""),
        "siglip_masked_score": fields.get("siglip_masked_score", ""),
        "expected_color": fields.get("expected_color", ""),
        "color_score": fields.get("color_score", ""),
        "center_x_norm": fields.get("center_x_norm", ""),
        "center_y_norm": fields.get("center_y_norm", ""),
        "speed": fields.get("speed", ""),
        "context_png": rel(context_name),
        "bbox_crop_png": rel(bbox_crop_name) if bbox_crop is not None else "",
        "sam_mask_png": rel(sam_mask_name),
        "masked_crop_png": rel(masked_crop_name) if masked_crop is not None else "",
        "combined_debug_png": rel(combined_name),
    }
    _append_index_row(row)
    return row


def _fmt(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and np.isnan(v):
            return None
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return v


def _color_label(fields):
    color = fields.get("expected_color")
    score = fields.get("color_score")
    if not color:
        return None
    score_str = _fmt(score)
    return f"{color} {score_str}" if score_str is not None else color


def _bbox_str(box):
    if box is None:
        return ""
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return ""
    if any(np.isnan(v) for v in vals):
        return ""
    return ",".join(f"{v:.1f}" for v in vals)


class DebugPolicy:
    """
    Generic, non-per-query-id selection of which frames are "interesting"
    enough to render debug images for, so a full run doesn't dump
    thousands of PNGs. The same criteria apply to every query:

      - the first valid detection seen
      - the highest Grounding DINO confidence seen so far
      - the highest SigLIP (masked, falling back to bbox) score seen so far
      - the largest |masked_score - bbox_score| mismatch seen so far
        (exactly the "DINO says yes, SigLIP on the clean mask disagrees"
        case this whole debug setup exists to catch)
      - up to a few representative frames spread evenly through the video

    Each criterion can only ever hold ONE current "champion" frame; when a
    later frame beats it, the caller re-renders that slot (cheap PNG I/O)
    and the previous file for that slot is simply overwritten. This needs
    no second pass over the video and no buffering of raw frames.
    """

    REPRESENTATIVE_SLOTS = 3

    def __init__(self, max_images=20, video_duration=None):
        self.max_images = max_images
        self.video_duration = video_duration
        self._best_dino = {}
        self._best_siglip = {}
        self._best_mismatch = {}
        self._seen_first = set()
        self._representative_next_t = {}
        self._saved_count = {}

    def _budget_ok(self, query_id):
        return self._saved_count.get(query_id, 0) < self.max_images

    def _mark_saved(self, query_id):
        self._saved_count[query_id] = self._saved_count.get(query_id, 0) + 1

    def slots_for(self, query_id, detector_score, siglip_score, bbox_score, timestamp):
        """Returns a list of slot-name strings this frame currently wins
        (possibly empty, possibly more than one)."""
        if not self._budget_ok(query_id):
            return []

        slots = []

        if query_id not in self._seen_first:
            self._seen_first.add(query_id)
            slots.append("first_valid")

        if detector_score is not None:
            if detector_score > self._best_dino.get(query_id, float("-inf")):
                self._best_dino[query_id] = detector_score
                slots.append("highest_dino")

        best_siglip = siglip_score if siglip_score is not None else bbox_score
        if best_siglip is not None:
            if best_siglip > self._best_siglip.get(query_id, float("-inf")):
                self._best_siglip[query_id] = best_siglip
                slots.append("highest_siglip")

        if siglip_score is not None and bbox_score is not None:
            mismatch = abs(siglip_score - bbox_score)
            if mismatch > self._best_mismatch.get(query_id, float("-inf")):
                self._best_mismatch[query_id] = mismatch
                slots.append("suspicious_mismatch")

        if self.video_duration:
            next_t = self._representative_next_t.get(
                query_id, self.video_duration / (self.REPRESENTATIVE_SLOTS + 1)
            )
            slot_index = len([s for s in slots if s == "representative"])
            saved_reps = getattr(self, "_reps_saved", {}).get(query_id, 0)
            if timestamp >= next_t and saved_reps < self.REPRESENTATIVE_SLOTS:
                slots.append("representative")
                self._representative_next_t[query_id] = next_t + self.video_duration / (
                    self.REPRESENTATIVE_SLOTS + 1
                )
                if not hasattr(self, "_reps_saved"):
                    self._reps_saved = {}
                self._reps_saved[query_id] = saved_reps + 1

        if slots:
            self._mark_saved(query_id)
        return slots
