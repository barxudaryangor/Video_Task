# Project status — temporal video grounding competition

## Task

Kaggle-style competition: temporal video grounding. Given a natural-language
query and a video, predict the time interval(s) where the query holds true
(or `NONE` if it never does). Queries are one of three types: `object`,
`state`, `action`.

**Data**: `data/queries.csv` (72 queries) + 8 videos (`data/video_01.mp4` ...
`video_08.mp4`, 3-5 min each, resolutions ranging 320x240 to 1920x1080, all
static surveillance-style cameras — parking lots, stores, streets).

**Submission format**: `outputs/submission.csv` with columns `query_id,prediction`.
`prediction` is `"start end;start2 end2"` (seconds, semicolon-separated
multiple intervals) or `NONE`.

**Scoring**: temporal-IoU F1 averaged over thresholds {0.3, 0.5, 0.7},
matched one-to-one by max-cardinality bipartite matching, then macro-averaged
over the three query types. Critically: empty ground truth + `NONE`
prediction scores 1.0; empty ground truth + non-empty prediction scores 0
(a wrong guess is NOT neutral — it forfeits a free point on true-negative
queries).

**Competition also requires** (separate from this repo's current scope):
a Google Colab notebook that reproduces `submission.csv` end-to-end from a
clean restart (weights auto-download, no manual steps, no hard-coded
intervals), plus a written "approach & experiments" document, plus GPU
credits management on Vast.ai ($12 budget).

## Repo layout

```
data/                       queries.csv + 8 source videos
outputs/
  object_frame_scores.csv   per-frame scores for all 24 object queries
  state_frame_scores.csv    per-frame scores for all 24 state queries
  submission.csv            current best combined submission (object done,
                             state in progress, action not started)
  approach_report.pdf       written report (object category only so far)
  frame_checks/             manually-extracted verification frames/crops
  report_assets/            chart images embedded in the PDF report
src/
  object_pipeline.py        object-category: raw score collection (slow, GPU)
  build_submission.py       object-category: score(t) -> intervals -> submission.csv
  state_pipeline.py         state-category: raw score collection (slow, GPU)
  build_state_submission.py state-category: score(t) -> intervals, merges into submission.csv
  sahi_experiment.py        one-off tiled-detection experiment (see below)
  generate_report.py        builds outputs/approach_report.pdf
```

## Object category (24 queries) — done, calibrated

**Models**: `google/siglip2-giant-opt-patch16-384` (embedding/similarity) +
`IDEA-Research/grounding-dino-base` (zero-shot open-vocabulary detector).
Both are the largest publicly available checkpoints in their respective
families that fit in 8GB VRAM (peak observed ~5.9GB).

**Pipeline** (`object_pipeline.py`): sample every 0.5s. Per frame: run
Grounding DINO with ALL of that video's query texts batched as one call ->
candidate boxes with confidence scores. For each match, crop the box (+10%
margin) from the *original* full-resolution frame and re-embed with SigLIP,
comparing against the query's text embedding (`crop_score`, raw cosine
similarity). No whole-frame ("global") embedding signal — deliberately
removed after an ablation proved it only injects noise for small objects
(documented in the PDF report and in code comments) without providing any
independent fallback signal.

**Postprocessing** (`build_submission.py`): interval = frame runs where
`detector_score > 0 AND crop_score >= MIN_CROP_SCORE` (currently **0.13**),
gap-merged (bridge <=5.0s dropouts), padded ±0.25s at boundaries. This
replaced an earlier smoothed/z-score/hysteresis approach that had two
concrete failure modes (documented in code): it deleted a real single-frame
detection by over-smoothing, and it fragmented a query that was detected in
100% of its video's frames into 5 pieces because z-scoring is relative, not
absolute.

**Threshold calibration**: swept against a handful of *manually
video-verified* cases (extracted actual frames, looked at them) — not a
statistically rigorous validation set, just enough anchor points to place
the cutoff. 0.13 was chosen because it collapses two confirmed false
positives (a beige car scored against "a yellow car"; a red umbrella scored
against "a blue umbrella" — color-attribute binding failures, a known
CLIP/SigLIP-family limitation, NOT fixed by using a bigger model — bigger
model made the false matches *more* confident, not less) to `NONE`, while
keeping confirmed true positives (a verified cyclist, a continuously-present
dumpster/car) largely intact. Known unresolved residual: one confirmed false
positive (a static background object mistaken for "yellow shorts") survives
this threshold because its score sits in the same range as real matches —
crop_score alone cannot separate it; needs a different signal (box-position
stability over time — a query about a moving person shouldn't have a
pixel-static box for the length of the whole video).

**Result**: 14-16/24 object queries get a non-`NONE` prediction depending on
exact threshold; the rest are `NONE` (some high-confidence — literally zero
detections all video — some lower-confidence judgment calls).

**SAHI tiling experiment** (`sahi_experiment.py`): tested manually-written
2x2 overlapping-tile detection (since off-the-shelf `sahi` package doesn't
support Grounding DINO's text-conditioned calling convention) on video_01 +
video_06 at coarser 1.0s sampling, to see if it recovers small/distant
objects the untiled detector misses. Result was NOT a clear win: it made the
detector fire far more often (more candidate boxes) without improving
crop_score confidence on the true target — net effect closer to added noise
than added recall, at this specific tiling granularity/sampling rate. Not
adopted into the main pipeline. Left as a documented negative result.

## State category (24 queries) — data collection done, calibration pending

**Architecture**: one unified mechanism for all 24 queries (not 8 bespoke
solutions per sub-type), reusing the exact same detect+crop+SigLIP core as
object. Per query, at runtime:

1. **Anchor + mode extraction is now fully automatic** (via spaCy dependency
   parsing in `state_pipeline.py::extract_anchor_and_mode`), NOT a
   hand-written per-query lookup table (an earlier version WAS hand-written
   — read each of the 24 query texts and manually typed an anchor phrase —
   correctly flagged by the user as not generalizing to unseen queries, and
   replaced). The algorithm: find the sentence's grammatical subject
   (nsubj/nsubjpass/attr), detect negation structurally (a "no" determiner
   attached to the subject, not a keyword list), build the anchor from the
   subject's dependency subtree (this naturally captures noun-internal
   modifiers like "no cars IN THE PARKING LOT") plus any prepositional
   phrases attached to the main verb (disambiguates e.g. "a person NEAR A
   TABLE" from "a person ON THE STAIRS" so two queries in the same video
   sharing a subject noun don't collide on an identical anchor string —
   this exact collision was a real bug found and fixed: Grounding DINO given
   two *identical* text prompts in one batched call can't tell which
   detection belongs to which query, so one starves).
2. **presence mode** (most queries): detect anchor, crop, score crop against
   the query's FULL text via SigLIP (same evidence-gate segmentation as
   object).
3. **absence mode** (5 queries whose text negates something, e.g. "there are
   no cars in the parking lot"): skip SigLIP entirely (comparing a crop to a
   negated sentence is meaningless) — the query is true during the GAPS in
   time where the anchor was never detected (with a minimum gap duration so
   an ordinary detector miss isn't mistaken for real absence).

This does NOT specially handle spatial relations, pose/orientation, or
counting queries with any bespoke geometry/counting logic — those ~10-12 of
24 queries get the same generic treatment and are expected to be less
reliable; this was a deliberate scope decision (one honest uniform attempt
across all 24 rather than hand-picking easy ones) rather than an oversight.

**Status**: `state_pipeline.py` has been run to completion 3 times as data
collection was fixed up — 1st run had hand-written per-query anchors, 2nd
run patched a duplicate-anchor collision bug for 2 specific queries, 3rd
(current, final) run regenerated everything with the fully-automatic spaCy
extraction described above. `build_state_submission.py` has been rerun
against this final data and `outputs/submission.csv` is up to date: 14/24
state queries get a non-`NONE` prediction.

`MIN_CROP_SCORE=0.13` and other thresholds were carried over unchanged from
object category — NOT yet independently recalibrated for state's own score
distribution (a real gap: state's crop_score values could plausibly have a
different natural scale, same lesson as when the object-category threshold
had to be redone after the SigLIP model upgrade).

**Known regression from automatic anchor extraction, deliberately left
as-is**: for `q022` ("a person is standing near a table under an umbrella"),
the auto-extracted anchor is the full phrase "a person near a table under an
umbrella" — Grounding DINO given this as a single detection target finds
*zero* matches all video, whereas the earlier hand-written short anchor
("a person" alone) matched 602/603 frames. This mirrors the same general
pattern seen elsewhere: Grounding DINO's phrase-grounding is more reliable
with short, concrete noun phrases than with long compound descriptive
phrases. The richer auto-extracted phrase is what fixed a real duplicate-
anchor collision bug (q022 vs q024 both reducing to bare "a person" in the
hand-written version), so it is a genuine trade-off, not a pure regression
— explicitly decided (by the user) to leave as a documented limitation
rather than special-case it back. A cleaner fix, not yet implemented: feed
the *short* subject-only phrase to the detector for localization, and use
the fuller phrase (with prepositional context) only for SigLIP scoring and
for disambiguating a same-video anchor collision when one actually occurs.
Also visually unverified but suspicious: `q059` ("a tractor is lying on its
side") and `q068` ("there are no shopping carts in the frame") both produce
heavily fragmented interval lists (10 and ~25 pieces respectively) that look
more like detector noise than a real intermittent state — not yet
investigated.

## Action category (24 queries) — not started

No code written yet. Discussed direction (not implemented): actions need
motion/temporal evidence a single frame can't provide. Planned architecture
(validated conceptually against a paper the user found, Souček et al.
"Look for the Change" / ChangeIt, and another on hour-scale video grounding
as a search problem):

- **Person-subject queries**: Grounding DINO (find person) + SAM2 (stable
  per-frame mask/position, video-tracking so detection doesn't need to
  rerun every frame) + ViTPose (body keypoints — wrists/elbows/knees — to
  read pose) + optical flow within the person's region (real pixel motion,
  to distinguish e.g. "standing still" from "raising an arm").
- **Rigid-object-subject queries** (car, ball, box): SAM2 mask per frame,
  no ViTPose (doesn't apply to non-human objects). Track mask center/size
  over time — center delta = velocity/direction, area growth = approaching
  the (static) camera. Good fit for "a car drives through the lot", "a
  tractor tips over", etc.
- **State-change-with-little-motion queries** (a door opens, trunk opens —
  really a state/action hybrid): SAM2 isolates the object; LOCAL optical
  flow inside the mask detects *when* a change is happening even though the
  whole object doesn't move much; a visual-state encoder (DINOv2/CLIP/
  SigLIP, comparing the crop against competing captions "X open" vs "X
  closed" rather than one absolute threshold — this also applies to the two
  open/closed STATE queries, q013/q041) classifies what state resulted.
  Mirrors the ChangeIt paper's causal-ordering framing: initial state ->
  action -> end state, in that temporal order.

**Also discussed, not yet implemented**: using Grounded-SAM (Grounding DINO
box -> SAM2 precise pixel mask, not just a rectangle) specifically to fix
the object-category's unresolved color-attribute problem — a rectangular
crop mixes in background pixels (concretely observed: a "red backpack" box
that actually captured hair instead of the backpack; a "yellow dumpster"
box mostly filled with sand, not the dumpster body), corrupting any
pixel-level color check. SAM2's video-tracking would also let color be
aggregated/voted across many clean per-frame masks of the same tracked
instance rather than trusting one frame. This was the immediate next
planned step before the state-pipeline rerun took priority.

## Known open problems / honest limitations

1. **Color-attribute binding**: SigLIP/CLIP-family embeddings do not
   reliably distinguish "right object, wrong color" from a genuine match
   when colors are visually adjacent (beige vs yellow, red vs blue in a
   red-dominant scene). A larger SigLIP model made this *worse* (more
   confident wrong matches), not better. An earlier pixel-level HSV
   color-check attempt was implemented and reverted — it needed the
   Grounded-SAM fix above to be reliable (rectangular boxes were too often
   capturing the wrong sub-region or mixed content).
2. **Small/distant objects**: Grounding DINO (even the -base checkpoint)
   can miss genuinely small objects (confirmed case: a small child with a
   red backpack, ~20x25px in a 1280x720 frame, at t~1.5-5s of video_01 —
   found only after upgrading from -tiny to -base). Manually-tested SAHI
   tiling did not clearly help at the tested granularity.
3. **Threshold calibration is small-sample**: `MIN_CROP_SCORE=0.13` (and
   related constants) were tuned by looking at a handful of manually
   video-verified queries, not the full 72. Real risk that it under- or
   over-fits queries never checked by eye.
4. **State's harder sub-types** (spatial relations, pose distinctions,
   counting, "inside vs outside" containment — roughly half of the 24 state
   queries) get only the generic detect+compare treatment, with no
   specialized geometry/counting/pose logic yet.
5. **Colab reproducibility**: the videos need an automatic (no manual
   upload) way to reach a fresh Colab runtime — not yet decided (Google
   Drive + gdown? Kaggle API? direct URL?). Model weights themselves
   auto-download fine (public HF checkpoints, no auth needed).

## Everything is real inference, not per-query hard-coding

Anchor extraction (state) is a generic syntactic rule applied uniformly via
spaCy, not a per-query_id lookup table (an earlier hand-written version was
explicitly replaced for this reason). Thresholds are global constants
shared across all queries in a category, not tuned per individual query_id.
No ground-truth intervals are known or used anywhere in this pipeline —
all manual "verification" was actual frame-by-frame visual inspection of
the videos by extracting real frames, not consulting any answer key.
