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
        return next(iter(doc.noun_chunks)).text

    return query_text
