"""
Generic (not per-query) color-word extraction and mask-restricted color
scoring.

The whole point is fixing the "yellow object" = beige object + yellow/sandy
BACKGROUND problem found earlier by hand: color must be judged from pixels
INSIDE the SAM2 mask, never the rectangular crop, since the box routinely
includes background that has nothing to do with the object.

Nothing here knows about query_id. extract_color_words() works on
whatever text it's given; masked_color_score() works on whatever
image+mask it's given.
"""

import re

import cv2
import numpy as np

COLOR_WORDS = {
    "red": "red", "orange": "orange", "yellow": "yellow", "green": "green",
    "blue": "blue", "purple": "purple", "white": "white", "black": "black",
    "gray": "gray", "grey": "gray", "brown": "brown", "pink": "pink",
}

# OpenCV HSV: H in [0,179], S,V in [0,255]. Gaps between ranges are left
# unclassified on purpose (a pixel we're not confident about should not
# be forced into the nearest bucket).
_HUE_RANGES = {
    "red": [(0, 9), (160, 179)],
    "orange": [(10, 19)],
    "brown": [(10, 24)],       # overlaps orange in hue; brown differs mainly
                               # by lower value, see _pixel_color_buckets
    "yellow": [(20, 34)],
    "green": [(35, 84)],
    "blue": [(85, 129)],
    "purple": [(130, 154)],
    "pink": [(155, 169)],
}

_NEUTRAL_SAT_MAX = 40   # below this saturation, hue is unreliable
_WHITE_VAL_MIN = 190
_BLACK_VAL_MAX = 50
_BROWN_VAL_MAX = 150    # orange-hued but darker/duller reads as brown, not orange


def extract_color_words(text):
    """Every known color word found in text, in first-seen order, e.g.
    'a yellow-and-white car' -> ['yellow', 'white']. Empty list if none."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    colors = []
    for t in tokens:
        c = COLOR_WORDS.get(t)
        if c and c not in colors:
            colors.append(c)
    return colors


def _pixel_color_buckets(hsv_pixels):
    """hsv_pixels: (N,3) uint8 array of H,S,V. Returns an (N,) array of
    bucket-name strings; '' means "not confidently any known color"."""
    h = hsv_pixels[:, 0].astype(np.int16)
    s = hsv_pixels[:, 1].astype(np.int16)
    v = hsv_pixels[:, 2].astype(np.int16)

    buckets = np.full(len(h), "", dtype=object)

    neutral = s < _NEUTRAL_SAT_MAX
    buckets[neutral & (v < _BLACK_VAL_MAX)] = "black"
    buckets[neutral & (v >= _BLACK_VAL_MAX) & (v < _WHITE_VAL_MIN)] = "gray"
    buckets[neutral & (v >= _WHITE_VAL_MIN)] = "white"

    chromatic = ~neutral
    for name, ranges in _HUE_RANGES.items():
        in_range = np.zeros(len(h), dtype=bool)
        for lo, hi in ranges:
            in_range |= (h >= lo) & (h <= hi)

        candidate = chromatic & in_range & (buckets == "")
        if name == "brown":
            candidate &= v < _BROWN_VAL_MAX
        elif name == "orange":
            candidate &= v >= _BROWN_VAL_MAX
        buckets[candidate] = name

    return buckets


def masked_color_score(image, mask, expected_colors):
    """
    image: PIL.Image (any mode; converted to RGB).
    mask:  np.ndarray[bool], full-image size (from sam2_utils.predict_mask).
    expected_colors: list[str] from extract_color_words(); may be empty.

    Returns (color_score, dominant_color):
      - (None, None) if expected_colors is empty or the mask is empty --
        i.e. this genuinely does not apply, not "0% match".
      - Otherwise color_score in [0, 1]: the fraction of confidently-
        classified in-mask pixels whose color is one of expected_colors.
        dominant_color is the single most common classified bucket in
        the mask (for display/diagnostics; not itself a pass/fail call).
    """
    if not expected_colors or mask is None or not mask.any():
        return None, None

    rgb = np.array(image.convert("RGB"))
    pixels_rgb = rgb[mask]
    if len(pixels_rgb) == 0:
        return None, None

    hsv = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    buckets = _pixel_color_buckets(hsv)

    valid = buckets != ""
    if not valid.any():
        return 0.0, None

    match = np.isin(buckets[valid], expected_colors)
    score = float(match.mean())

    values, counts = np.unique(buckets[valid], return_counts=True)
    dominant = str(values[np.argmax(counts)])

    return score, dominant
