"""
Generic (not per-query-id) subject-noun-phrase extraction for ACTION
queries, used only to pick a short, DINO-friendly text prompt for
Stage 1 (SAM2-based candidate-window proposal). This is deliberately
simpler than state_pipeline.py's extract_anchor_and_mode -- it does not
need presence/absence detection or anchor-collision disambiguation
(those solve problems specific to state queries), and it never falls
back to returning the whole query text, since a long fallback phrase is
exactly what makes Grounding DINO's phrase-grounding unreliable
(documented in PROJECT_STATUS.md).

The actual ACTION semantics (what the subject is DOING) are deliberately
NOT extracted here -- that is Qwen3-VL's job in Stage 2. This function
only answers "what noun should Stage 1 point a detector at".
"""

import spacy

_NLP = spacy.load("en_core_web_sm")

# TRIED AND REVERTED (2026-09-15): a "tractor" -> "a construction vehicle"
# Stage-1 vocabulary override. Isolated single-frame SAM3 probes on
# video_07 looked promising (0.60-0.96 for "a construction vehicle" vs.
# a noisy 0.32-0.86 for "a tractor", with "a tractor" dipping below the
# 0.5 presence threshold at some frames inside a real ground-truth
# window) -- but a full pipeline run on q063 ("a tractor is operating")
# showed the opposite in aggregate: the override made SAM3 presence
# saturate (>85% of frames), collapsing Stage 1 back into an unfiltered
# 19-window whole-video scan, and Qwen's resulting window boundaries
# shifted enough to MISS a true interval (167-175s) that the un-overridden
# "a tractor" pipeline had correctly covered (162.95-183.25). Net effect
# on real data was a regression, not an improvement -- do not reintroduce
# this override without re-validating on a full run, not just isolated
# frame probes (same "looks great in isolation, regresses in aggregate"
# lesson as the object/state pipeline's anchor-shortening attempts, see
# PROJECT_STATUS.md).


def extract_action_subject(query_text):
    """
    Returns a short noun phrase: the sentence's grammatical subject
    (nsubj/nsubjpass) plus only its direct modifiers (determiner,
    compound nouns, adjectives) -- never prepositional phrases or
    clausal attachments, and never the whole sentence as a fallback.

    If no clear subject is found, returns the single root-most noun
    chunk in the sentence (spaCy's noun_chunks), so the result is
    always a short phrase, not a full clause.

    Examples:
      "a car drives through the parking lot" -> "a car"
      "several men lift the tractor back up" -> "several men"
      "a lawn mower cutting the grass" -> "a lawn mower"
      "a tractor tips over" -> "a tractor"
      "people are running away" -> "people"
    """
    doc = _NLP(query_text)

    subject = next(
        (t for t in doc if t.dep_ in ("nsubj", "nsubjpass")),
        None,
    )

    if subject is not None:
        keep_deps = {"det", "compound", "amod", "nummod", "poss"}
        children = [c for c in subject.children if c.dep_ in keep_deps]
        span_tokens = sorted(children + [subject], key=lambda t: t.i)
        lo, hi = span_tokens[0].i, span_tokens[-1].i
        return doc[lo : hi + 1].text

    if doc.noun_chunks:
        chunk = next(iter(doc.noun_chunks))
        # spaCy occasionally mistags a short intransitive phrasal verb as a
        # NOUN when no nsubj is found (confirmed 2026-09-15 on real data:
        # "a tractor tips over" has no finite-verb parse spaCy recognizes,
        # so "tips" gets tagged NOUN/ROOT and folded into this fallback
        # noun chunk as "a tractor tips" instead of being left out as the
        # verb -- SAM3 then searches for the nonsense phrase "a tractor
        # tips" and scores near zero everywhere, even where a tractor is
        # clearly on screen). Detect this narrow case: if what's left
        # after the chunk is a single bare intransitive particle (no
        # object following it), the chunk's last token is almost
        # certainly that mistagged verb, not part of the subject phrase.
        PARTICLES = {"over", "up", "down", "off", "back", "away", "out"}
        after = doc[chunk.end:].text.lower().strip()
        if len(chunk) > 1 and after in PARTICLES:
            return chunk[:-1].text
        return chunk.text

    return query_text
