"""
Builds outputs/approach_report.pdf -- the written "approach, experiments,
ideas explored, reasoning" deliverable for the object-category pipeline.

This does not touch object_pipeline.py or build_submission.py; it only
narrates the design decisions already made there, reusing the frame
screenshots saved under outputs/frame_checks/ during manual verification
and the ablation numbers computed earlier from outputs/object_frame_scores.csv.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    HRFlowable,
    KeepTogether,
)

OUTPUT_DIR = Path("outputs")
ASSETS_DIR = OUTPUT_DIR / "report_assets"
FRAMES_DIR = OUTPUT_DIR / "frame_checks"
REPORT_PATH = OUTPUT_DIR / "approach_report.pdf"

INK = colors.HexColor("#201d1a")
MUTED = colors.HexColor("#6b6459")
BORDER = colors.HexColor("#ddd4c3")
KEPT = colors.HexColor("#1f6f63")
KEPT_SOFT = colors.HexColor("#e2efe9")
REMOVED = colors.HexColor("#a23b2e")
REMOVED_SOFT = colors.HexColor("#f3e2dc")


# ============================================================
# CHART: q046 fused score, with vs. without the global signal
# ============================================================

def build_ablation_chart():
    times = [165.5, 166.0, 166.5, 167.0, 167.5, 168.0, 168.5,
             169.0, 169.5, 170.0, 170.5, 171.0, 171.5]
    with_global = [-0.995320, 0.080505, 0.344403, 0.881151, 1.340670,
                   1.474588, 2.039618, -0.071282, -0.449587, -0.257546,
                   -0.391355, -0.447881, -0.481407]
    without_global = [0.0] * len(times)

    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)

    ax.plot(times, with_global, color="#a23b2e", linewidth=2,
            marker="o", markersize=3.5, label="fused score — with global (run 1)")
    ax.plot(times, without_global, color="#1f6f63", linewidth=2.4,
            label="fused score — without global (current)")

    peak_i = with_global.index(max(with_global))
    ax.annotate(f"{with_global[peak_i]:.2f}",
                (times[peak_i], with_global[peak_i]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color="#a23b2e", fontweight="bold")

    ax.set_xlabel("time (s)", fontsize=9, color="#3a352f")
    ax.set_ylabel("fused_score (z-units)", fontsize=9, color="#3a352f")
    ax.set_title("q046 — \"a person wearing a red cap\" (video_06, detector_score = 0 all video)",
                 fontsize=9.5, color="#3a352f", loc="left")
    ax.axhline(0, color="#ddd4c3", linewidth=1, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ddd4c3")
    ax.tick_params(colors="#6b6459", labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()

    path = ASSETS_DIR / "q046_ablation_chart.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ============================================================
# STYLES
# ============================================================

def make_styles():
    styles = {}
    styles["Title"] = ParagraphStyle(
        "Title", fontName="Times-Bold", fontSize=21, leading=25,
        textColor=INK, spaceAfter=6,
    )
    styles["Subtitle"] = ParagraphStyle(
        "Subtitle", fontName="Helvetica", fontSize=11, leading=15,
        textColor=MUTED, spaceAfter=4,
    )
    styles["Meta"] = ParagraphStyle(
        "Meta", fontName="Courier", fontSize=8.5, leading=12,
        textColor=MUTED, spaceBefore=10, spaceAfter=2,
    )
    styles["H1"] = ParagraphStyle(
        "H1", fontName="Times-Bold", fontSize=15, leading=18,
        textColor=INK, spaceBefore=22, spaceAfter=8,
    )
    styles["H2"] = ParagraphStyle(
        "H2", fontName="Helvetica-Bold", fontSize=11.5, leading=14,
        textColor=INK, spaceBefore=14, spaceAfter=5,
    )
    styles["Body"] = ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=9.7, leading=14.5,
        textColor=INK, spaceAfter=7, alignment=4,
    )
    styles["Caption"] = ParagraphStyle(
        "Caption", fontName="Helvetica-Oblique", fontSize=8.5, leading=12,
        textColor=MUTED, spaceBefore=3, spaceAfter=12,
    )
    styles["Code"] = ParagraphStyle(
        "Code", fontName="Courier", fontSize=9, leading=13,
        textColor=INK, backColor=colors.HexColor("#f2ede2"),
        borderPadding=6, spaceAfter=8,
    )
    styles["CalloutLabel"] = ParagraphStyle(
        "CalloutLabel", fontName="Courier-Bold", fontSize=8.5, leading=11,
        textColor=REMOVED, spaceAfter=4,
    )
    styles["TableCell"] = ParagraphStyle(
        "TableCell", fontName="Helvetica", fontSize=8, leading=10.5, textColor=INK,
    )
    styles["TableCellMono"] = ParagraphStyle(
        "TableCellMono", fontName="Courier", fontSize=8, leading=10.5, textColor=INK,
    )
    styles["TableHead"] = ParagraphStyle(
        "TableHead", fontName="Helvetica-Bold", fontSize=7.6, leading=9.5,
        textColor=MUTED,
    )
    return styles


def rule():
    return HRFlowable(width="100%", thickness=0.75, color=BORDER, spaceAfter=10)


# ============================================================
# BUILD DOCUMENT
# ============================================================

def build_report():
    S = make_styles()
    story = []

    story.append(Paragraph(
        "Temporal Video Grounding — Object Category", S["Title"]))
    story.append(Paragraph(
        "Approach, experiments, and reasoning behind the design decisions",
        S["Subtitle"]))
    story.append(Paragraph(
        "Pipeline: object_pipeline.py (SigLIP + Grounding DINO frame scoring) "
        "&rarr; build_submission.py (score-to-interval postprocessing)　|　"
        "24 object queries, 8 surveillance videos", S["Meta"]))
    story.append(rule())

    # ------------------------------------------------------------
    story.append(Paragraph("1. Task and category scope", S["H1"]))
    story.append(Paragraph(
        "The task is temporal grounding: given a natural-language query and a video, predict "
        "the time interval(s), if any, where the query holds. Scoring is temporal-IoU F1 "
        "averaged over thresholds 0.3/0.5/0.7, macro-averaged over three query types "
        "(object, state, action). This report covers the <b>object</b> category only "
        "(24 of 72 queries) &mdash; the category attempted first because it reduces to a "
        "per-frame appearance question, unlike state (compound scene predicates) or action "
        "(requires temporal/motion reasoning).", S["Body"]))

    # ------------------------------------------------------------
    story.append(Paragraph("2. Approach", S["H1"]))
    story.append(Paragraph("2.1 Model choice", S["H2"]))
    story.append(Paragraph(
        "<b>SigLIP</b> (google/siglip-base-patch16-224) was chosen over CLIP because its "
        "sigmoid loss produces a calibrated image-text similarity rather than a purely "
        "rank-based one, making an absolute cutoff ("
        "“above X counts as present”) more meaningful than with plain CLIP cosine "
        "similarity. <b>Grounding DINO</b> (IDEA-Research/grounding-dino-tiny) is a zero-shot "
        "phrase-grounding detector: given an image and a batch of query phrases for that "
        "video, it returns candidate boxes with a confidence score, letting us localize "
        "<i>where</i> in the frame the query might be, not just whether the whole frame "
        "resembles it.", S["Body"]))
    story.append(Paragraph("2.2 Why not rely on the whole frame alone", S["H2"]))
    story.append(Paragraph(
        "Object queries typically name a small part of the frame (a person, a backpack, a "
        "single car among many). A single whole-frame embedding is dominated by background, "
        "so the pipeline additionally crops each detected box (with a 10% margin) and "
        "re-embeds <i>only that crop</i> with SigLIP, comparing it again to the query text. "
        "Three signals are produced per sampled frame (every 0.5s): <font face='Courier'>global_score</font> "
        "(whole frame), <font face='Courier'>detector_score</font> (Grounding DINO confidence), "
        "<font face='Courier'>crop_score</font> (SigLIP on the cropped box). Section 3.5 revisits "
        "and removes the first of these.", S["Body"]))
    story.append(Paragraph("2.3 Fusion", S["H2"]))
    story.append(Paragraph(
        "The three signals live on incomparable scales (detector confidence is roughly 0–1; "
        "SigLIP cosine similarity sits in a narrow 0–0.2 band). Each is <b>z-score "
        "normalized per query, across that query's own timeline</b> "
        "(<font face='Courier'>z = (x-mean)/std</font>) before being combined in a weighted sum. "
        "This is necessary to fuse the signals at all, but it has a structural side effect that "
        "drives most of Section 3: a z-score is always relative to that query's own noise floor, "
        "so it will report a &ldquo;peak&rdquo; even when the underlying signal never contains real "
        "evidence.", S["Body"]))

    # ------------------------------------------------------------
    story.append(Paragraph("3. Experiments and design iterations", S["H1"]))

    story.append(Paragraph("3.1 Fixed threshold &rarr; over-fragmentation &rarr; hysteresis", S["H2"]))
    story.append(Paragraph(
        "The first version thresholded the smoothed fused score at a single fixed z-value. "
        "For queries where the object is visible almost continuously, frame-to-frame detector "
        "noise made the score oscillate across that single cutoff dozens of times, producing "
        "up to 15 tiny predicted intervals for one real, continuous appearance &mdash; costly under "
        "IoU-matched F1, where every extra interval is an unmatched false positive. "
        "Fix: replace the single threshold with <b>hysteresis</b> &mdash; a frame only "
        "<i>enters</i> an event above an ENTER bar (0.75) and only <i>leaves</i> it below a lower "
        "EXIT bar (0.15), so noise straddling one boundary can no longer flip the state back "
        "and forth. Combined with a rolling-mean smoothing pass and merging of same-event "
        "segments separated by short gaps, this took queries like “a blue umbrella” from "
        "15 fragments down to 1–4 plausible ones.", S["Body"]))

    story.append(Paragraph("3.2 The pipeline could not say “absent”", S["H2"]))
    story.append(Paragraph(
        "The evaluation formula explicitly scores empty-ground-truth queries (a query whose "
        "object never appears, correctly predicted as NONE, scores 1.0). But per-query "
        "z-scoring means <i>any</i> curve has a relative peak, even pure noise. Three queries "
        "confirmed this directly: <font face='Courier'>q011</font>, <font face='Courier'>q030</font>, "
        "<font face='Courier'>q046</font> never produced a single Grounding DINO detection across "
        "their entire video (detector_score = 0 for thousands of sampled frames), yet the fused "
        "z-score still produced confident-looking peaks (1.07, 1.18, 2.04). Fix: an "
        "<b>absolute evidence gate</b> on top of the relative hysteresis segments &mdash; a "
        "candidate interval is only kept if it contains at least one frame with a real detection "
        "(detector_score &gt; 0, which itself already means Grounding DINO's own confidence "
        "threshold was cleared) <i>and</i> a crop SigLIP similarity of at least 0.085 "
        "(confident true matches in this dataset cluster around 0.11–0.17; unsupported ones "
        "sit at 0.05–0.09).", S["Body"]))

    story.append(Paragraph("3.3 Manual verification against real frames", S["H2"]))
    story.append(Paragraph(
        "Numeric thresholds were sanity-checked by extracting the actual video frame at each "
        "candidate peak. This caught a real design bug and one dataset-level limitation.",
        S["Body"]))

    verify_table = Table(
        [
            [Paragraph("<b>Query</b>", S["TableCell"]), Paragraph("<b>Claim checked</b>", S["TableCell"]),
             Paragraph("<b>Frame evidence</b>", S["TableCell"]), Paragraph("<b>Outcome</b>", S["TableCell"])],
            [Paragraph("q002", S["TableCellMono"]), Paragraph("“a cyclist”", S["TableCell"]),
             Paragraph("Cyclist clearly visible at the scored peak time.", S["TableCell"]),
             Paragraph("Confirmed true positive.", S["TableCell"])],
            [Paragraph("q003", S["TableCellMono"]), Paragraph("“a yellow car”", S["TableCell"]),
             Paragraph("Peak box is a beige/tan sedan, not yellow.", S["TableCell"]),
             Paragraph("Likely false positive — color-attribute confusion.", S["TableCell"])],
            [Paragraph("q021", S["TableCellMono"]), Paragraph("“a blue umbrella”", S["TableCell"]),
             Paragraph("Every umbrella in the scene is red/terracotta.", S["TableCell"]),
             Paragraph("Likely false positive — same failure mode.", S["TableCell"])],
        ],
        colWidths=[0.5 * inch, 1.0 * inch, 2.6 * inch, 1.9 * inch],
    )
    verify_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2ede2")),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(verify_table)
    story.append(Paragraph(
        "Frames referenced above are saved under outputs/frame_checks/ "
        "(q002_cyclist_top.png, q003_exact_box.png, q021_box_86.png).", S["Caption"]))

    story.append(Image(str(FRAMES_DIR / "q002_cyclist_top.png"), width=4.6 * inch, height=2.6 * inch))
    story.append(Paragraph(
        "Figure 1. q002 top-scoring frame (video_01, t=114.0s) &mdash; a cyclist is plainly visible "
        "on the sidewalk, confirming the detector+crop signal is grounded, not noise.", S["Caption"]))

    img1 = Image(str(FRAMES_DIR / "q003_exact_box.png"), width=2.6 * inch, height=2.03 * inch)
    img2 = Image(str(FRAMES_DIR / "q021_box_86.png"), width=2.6 * inch, height=2.5 * inch)
    side_by_side = Table([[img1, img2]], colWidths=[3.1 * inch, 3.1 * inch])
    side_by_side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(side_by_side)
    story.append(Paragraph(
        "Figure 2. Left: the exact detector box scored highest for q003 (“a yellow car”) is a "
        "beige sedan (crop_score=0.076, notably below the confident-match range). Right: the exact "
        "box for q021 (“a blue umbrella”) is a red umbrella (crop_score=0.091). Both sit right at "
        "the evidence-gate boundary, which cosine similarity alone does not resolve cleanly.",
        S["Caption"]))

    story.append(Paragraph("3.4 Attempted fix: pixel-level color verification (reverted)", S["H2"]))
    story.append(Paragraph(
        "Since SigLIP cosine similarity does not cleanly separate “right object, wrong color” "
        "from a genuine match (both land around 0.08–0.10), the next idea was to verify color "
        "directly from pixels: read the median HSV hue/saturation/value inside the detected box "
        "and classify it into a color bucket (red/yellow/blue/...), then require it to match a "
        "color word extracted from the query text. This was implemented and immediately reverted "
        "after it rejected <font face='Courier'>q001</font> (“a man with a red backpack”) &mdash; a query "
        "manually confirmed present in the video. Debugging showed why: the highest-crop-score box "
        "for q001 was 18×22 pixels and captured hair/shadow, not the backpack, giving a "
        "misleading median hue; separately, a box for “a yellow dumpster filled with sand” "
        "captured mostly the sand rather than the dumpster's own color. A single whole-box median "
        "color is too easily dominated by background, occlusion, or the wrong sub-region to gate a "
        "hard pass/fail on. This is left as a limitation rather than solved.", S["Body"]))

    img3 = Image(str(FRAMES_DIR / "q001_debug_box.png"), width=1.0 * inch, height=1.25 * inch)
    img4 = Image(str(FRAMES_DIR / "q055_debug_box.png"), width=2.4 * inch, height=1.87 * inch)
    debug_row = Table([[img3, Spacer(1, 1), img4]], colWidths=[1.3 * inch, 0.3 * inch, 2.7 * inch])
    debug_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(debug_row)
    story.append(Paragraph(
        "Figure 3. Left: the entire 18×22px box used to judge q001's color &mdash; mostly hair, not "
        "backpack. Right: the box for the “yellow dumpster” query, mostly filled with sand rather "
        "than the dumpster body. Both make a single median-pixel color check unreliable.",
        S["Caption"]))

    story.append(Paragraph("3.5 Ablation: removing the global whole-frame signal", S["H2"]))
    story.append(Paragraph(
        "A further look at the fusion design raised a structural question: the whole-frame "
        "(&ldquo;global&rdquo;) SigLIP signal was originally included as a fallback for frames the "
        "detector misses. But for a genuinely small object, the whole-frame embedding is diluted "
        "by background for the <i>same underlying reason</i> the detector misses it &mdash; so it is "
        "not an independent fallback, and could inject noise exactly where nothing else provides "
        "evidence.", S["Body"]))
    story.append(Paragraph(
        "This was tested empirically rather than assumed: the z-score fusion was recomputed from "
        "the already-cached raw <font face='Courier'>detector_score</font>/<font face='Courier'>crop_score</font> "
        "columns, once with the original weights (global 0.20 / detector 0.30 / crop 0.50) and "
        "once with global dropped (detector 0.375 / crop 0.625, preserving their 3:5 ratio) "
        "&mdash; no re-inference required.", S["Body"]))

    zero_table_data = [
        [Paragraph("<b>Query</b>", S["TableHead"]), Paragraph("<b>Text</b>", S["TableHead"]),
         Paragraph("<b>Max detector</b>", S["TableHead"]), Paragraph("<b>Max fused, with global</b>", S["TableHead"]),
         Paragraph("<b>Max fused, without</b>", S["TableHead"])],
        [Paragraph("q011", S["TableCellMono"]), Paragraph("a person wearing a yellow shirt", S["TableCell"]),
         Paragraph("0.00", S["TableCellMono"]), Paragraph("1.066", S["TableCellMono"]), Paragraph("0.000", S["TableCellMono"])],
        [Paragraph("q030", S["TableCellMono"]), Paragraph("a man wearing a blue jacket and purple pants", S["TableCell"]),
         Paragraph("0.00", S["TableCellMono"]), Paragraph("1.184", S["TableCellMono"]), Paragraph("0.000", S["TableCellMono"])],
        [Paragraph("q046", S["TableCellMono"]), Paragraph("a person wearing a red cap", S["TableCell"]),
         Paragraph("0.00", S["TableCellMono"]), Paragraph("2.040", S["TableCellMono"]), Paragraph("0.000", S["TableCellMono"])],
    ]
    zero_table = Table(zero_table_data, colWidths=[0.55 * inch, 2.35 * inch, 1.0 * inch, 1.2 * inch, 1.15 * inch])
    zero_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2ede2")),
        ("BACKGROUND", (0, 1), (-1, -1), REMOVED_SOFT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, BORDER),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(Paragraph("Table 1. Queries with zero Grounding DINO detections across their entire video.", S["CalloutLabel"]))
    story.append(zero_table)
    story.append(Paragraph(
        "Without the global term, the fused score for all three is a flat 0.000 &mdash; there is "
        "nothing else left to produce a peak from. With it, every one of those “peaks” (1.07, "
        "1.18, 2.04) was manufactured entirely by whole-frame embedding drift, unrelated to the "
        "query.", S["Caption"]))

    chart_path = build_ablation_chart()
    story.append(Image(str(chart_path), width=5.6 * inch, height=2.6 * inch))
    story.append(Paragraph(
        "Figure 4. Real fused_score values for q046 in a representative 6-second window. The "
        "“with global” curve is what run 1 actually produced (climbing to 2.04 with no "
        "detection nearby); “without global” is flat because detector_score is a constant "
        "zero for this query for the full video.", S["Caption"]))

    story.append(Paragraph(
        "Across all 24 object queries, the two fusion variants were compared by Pearson "
        "correlation of their time series. Most queries correlate above 0.9 (detector+crop "
        "already carry the signal; global changes little either way); the three zero-detection "
        "queries above cannot even be correlated, since one side is a constant. Re-running the "
        "full postprocessing pipeline (hysteresis + evidence gate) on both variants produced the "
        "<i>same</i> final kept/NONE verdict for all 24 queries &mdash; the evidence gate from Section "
        "3.2 already independently protects against this failure mode. The practical effect of "
        "removing global was therefore mainly a loss of incidental curve-smoothing for a few "
        "already-detected queries (more fragmented candidate segments), which was compensated by "
        "widening the smoothing window (5&rarr;7 samples) and the segment-merge gap (3.0s&rarr;5.0s) "
        "in build_submission.py, rather than by keeping a signal whose justification did not "
        "hold up.", S["Body"]))

    # ------------------------------------------------------------
    story.append(Paragraph("4. Final parameters (object_pipeline.py / build_submission.py)", S["H1"]))
    params_data = [
        [Paragraph("<b>Parameter</b>", S["TableHead"]), Paragraph("<b>Value</b>", S["TableHead"]), Paragraph("<b>Role</b>", S["TableHead"])],
        [Paragraph("Fusion weights", S["TableCell"]), Paragraph("detector 0.375 / crop 0.625", S["TableCellMono"]), Paragraph("No global term (Section 3.5).", S["TableCell"])],
        [Paragraph("Sample interval", S["TableCell"]), Paragraph("0.5 s", S["TableCellMono"]), Paragraph("Frame sampling rate for both SigLIP and Grounding DINO passes.", S["TableCell"])],
        [Paragraph("Grounding DINO thresholds", S["TableCell"]), Paragraph("box 0.25 / text 0.20", S["TableCellMono"]), Paragraph("Detector's own confidence cutoffs.", S["TableCell"])],
        [Paragraph("Crop margin", S["TableCell"]), Paragraph("10%", S["TableCellMono"]), Paragraph("Padding around a detected box before re-cropping for SigLIP.", S["TableCell"])],
        [Paragraph("Smoothing window", S["TableCell"]), Paragraph("7 samples (~3.5s)", S["TableCellMono"]), Paragraph("Rolling mean on fused_score before segmentation.", S["TableCell"])],
        [Paragraph("Hysteresis ENTER / EXIT", S["TableCell"]), Paragraph("0.75 / 0.15 (z-units)", S["TableCellMono"]), Paragraph("Prevents threshold-flapping from noise (Section 3.1).", S["TableCell"])],
        [Paragraph("Max merge gap", S["TableCell"]), Paragraph("5.0 s", S["TableCellMono"]), Paragraph("Bridges short dropouts inside one continuous event.", S["TableCell"])],
        [Paragraph("Min interval duration", S["TableCell"]), Paragraph("1.0 s", S["TableCellMono"]), Paragraph("Drops single-sample noise spikes.", S["TableCell"])],
        [Paragraph("Evidence gate", S["TableCell"]), Paragraph("detector_score &gt; 0 and crop_score &ge; 0.085", S["TableCellMono"]), Paragraph("Requires real detection, not just a relative peak (Section 3.2).", S["TableCell"])],
        [Paragraph("Boundary padding", S["TableCell"]), Paragraph("&plusmn;0.25 s", S["TableCellMono"]), Paragraph("Half a sample step, since the true edge lies between samples.", S["TableCell"])],
    ]
    params_table = Table(params_data, colWidths=[1.5 * inch, 1.85 * inch, 2.9 * inch])
    params_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2ede2")),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(params_table)

    # ------------------------------------------------------------
    story.append(Paragraph("5. Results and known limitations", S["H1"]))
    story.append(Paragraph(
        "16 of 24 object queries clear the evidence gate and receive one or more predicted "
        "intervals; 8 (q003, q011, q020, q030, q046, q047, q048, q056) are predicted NONE. "
        "For q011/q030/q046 this is high-confidence (zero detections all video). For q003 "
        "(color mismatch, Section 3.3), q020/q047/q048/q056 (weak but nonzero crop evidence, "
        "0.052–0.077) it is a lower-confidence call the pipeline cannot fully verify without "
        "ground truth &mdash; these may be real true negatives, or real objects our detector/crop "
        "combination is simply too weak to confirm confidently.", S["Body"]))
    story.append(Paragraph(
        "The main known limitation is <b>color-attribute binding</b>: SigLIP/CLIP-style cosine "
        "similarity does not reliably distinguish “right object, wrong color” from a genuine "
        "match when the color is visually adjacent (beige vs. yellow, red vs. blue in a "
        "red-dominant scene). A pixel-level fix was attempted and reverted (Section 3.4) after "
        "it proved less reliable than the embedding itself on small or mixed-content boxes; this "
        "remains open for the object category, where almost every query is distinguished "
        "primarily by color.", S["Body"]))
    story.append(Paragraph(
        "State and action queries (48 of 72) are not yet implemented in this pipeline and are "
        "currently written as NONE placeholders in submission.csv so the file matches the "
        "required format.", S["Body"]))

    doc = SimpleDocTemplate(
        str(REPORT_PATH), pagesize=LETTER,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title="Temporal Video Grounding - Object Category Approach Report",
    )
    doc.build(story)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    build_report()
