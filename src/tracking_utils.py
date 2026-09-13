"""
Generic (not per-query-id) pseudo-tracking built on top of the pipelines'
existing independent per-frame detections.

IMPORTANT -- what this is NOT:
  - NOT a SAM2-video-memory tracker (no persistent inference session,
    no learned re-identification across occlusion).
  - NOT multi-object tracking (no Hungarian/IoU assignment across several
    simultaneous candidates -- each query already keeps only its single
    best-scoring detection per frame upstream of this module).
  - NOT action recognition. It only exposes motion/stability numbers as
    auxiliary evidence; nothing here accepts or rejects a detection.

What it IS: each query's frames are already processed one at a time and
independently (see object_pipeline.py / state_pipeline.py). This module
just stitches that already-independent sequence of (timestamp, box) pairs
into "tracks" after the fact, using two simple generic rules -- a large
time gap since the last detection, or a large jump in normalized
position -- to guess when a query's detector most likely jumped from one
physical instance to a different one. It is a lightweight heuristic for
producing evidence, not a claim of guaranteed identity continuity.
"""

import math

# A gap this long since the last detection of the same query is treated as
# "the object was gone/undetected long enough that this is probably a new
# instance", not a continuation of the same track.
MAX_TIME_GAP_SEC = 3.0

# A one-step jump in normalized center position larger than this fraction
# of the frame is treated the same way (an object simply cannot teleport
# across a big chunk of a static-camera frame between two nearby samples).
MAX_JUMP_NORM = 0.25


class QueryTracker:
    """
    One instance per query, scoped to a single video (queries never span
    videos in this dataset, and each pipeline processes one video's worth
    of frames per call).

    Call update(timestamp, box, frame_width, frame_height) once per
    sampled frame, in increasing timestamp order, for every frame --
    including frames where nothing was detected (box=None) -- so gaps are
    recorded honestly rather than silently skipped.
    """

    def __init__(self):
        self.track_id = 0
        self._last = None  # {"t", "cx", "cy", "area"}
        self._speed_sum = 0.0
        self._speed_count = 0

    def update(self, timestamp, box, frame_width, frame_height):
        """
        box: [x0, y0, x1, y1] in pixel coordinates, or None if this query
        had no detection on this frame.

        Returns a dict of the new evidence columns for this frame. All
        values are None when box is None (tracking evidence genuinely
        does not apply to a frame with no detection) or when there isn't
        yet enough history to compute a given value (e.g. dx/dy/speed on
        the very first detection of a track) -- never invented as 0.
        """
        if box is None:
            return self._empty()

        x0, y0, x1, y1 = [float(v) for v in box]
        width_px = x1 - x0
        height_px = y1 - y0
        area_px = width_px * height_px
        cx = (x0 + x1) / 2.0 / frame_width
        cy = (y0 + y1) / 2.0 / frame_height

        dx = dy = speed = delta_area = stability = None

        if self._last is None:
            self.track_id = 1
        else:
            dt = timestamp - self._last["t"]
            raw_dx = cx - self._last["cx"]
            raw_dy = cy - self._last["cy"]
            jump = math.hypot(raw_dx, raw_dy)

            if dt > MAX_TIME_GAP_SEC or jump > MAX_JUMP_NORM:
                # Most likely a different physical instance -- start a new
                # track and reset the running speed stats that feed
                # track_stability, rather than carrying old-track history
                # into the new one.
                self.track_id += 1
                self._speed_sum = 0.0
                self._speed_count = 0
            else:
                dx, dy = raw_dx, raw_dy
                delta_area = area_px - self._last["area"]
                if dt > 0:
                    speed = jump / dt
                    self._speed_sum += speed
                    self._speed_count += 1
                if self._speed_count > 0:
                    avg_speed = self._speed_sum / self._speed_count
                    # In (0, 1]: near 1 for a track that has barely moved
                    # so far, decaying toward 0 as its average speed grows.
                    # This is descriptive evidence only -- nothing here
                    # decides a track is "too stable to be real".
                    stability = 1.0 / (1.0 + avg_speed)

        self._last = {"t": timestamp, "cx": cx, "cy": cy, "area": area_px}

        return {
            "track_id": self.track_id,
            "center_x_norm": cx,
            "center_y_norm": cy,
            "width": width_px,
            "height": height_px,
            "area": area_px,
            "dx": dx,
            "dy": dy,
            "speed": speed,
            "delta_area": delta_area,
            "track_stability": stability,
        }

    @staticmethod
    def _empty():
        return {
            "track_id": None,
            "center_x_norm": None,
            "center_y_norm": None,
            "width": None,
            "height": None,
            "area": None,
            "dx": None,
            "dy": None,
            "speed": None,
            "delta_area": None,
            "track_stability": None,
        }
