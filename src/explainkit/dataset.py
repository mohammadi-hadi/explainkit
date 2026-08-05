"""A synthetic corpus with implanted rationales — validation by construction.

Real moderation data cannot be bundled, and it would not come with ground
truth about *which words carry the label*.  This corpus does: every text is
neutral workplace filler with one to three signal words implanted from a
hostile or civil lexicon, and the implant positions are recorded.  That gives
the demo and the test suite something no real dataset offers — known-true
rationales, so plausibility metrics and token selection can be checked for
recovering exactly what was planted.  All of it derives from one seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOSTILE_LEXICON = (
    "useless", "pathetic", "incompetent", "idiotic", "clueless", "worthless",
    "lazy", "sloppy", "hopeless", "dreadful", "embarrassing", "stupid",
)

CIVIL_LEXICON = (
    "helpful", "thoughtful", "careful", "creative", "attentive", "reliable",
    "diligent", "precise", "patient", "generous", "organized", "thorough",
)

# Hostile word -> civil counterpart, used by the demo's replacement probe:
# swapping the evidence should swap the verdict.
CIVIL_SYNONYMS: dict[str, str] = dict(zip(HOSTILE_LEXICON, CIVIL_LEXICON, strict=True))

_FILLER = (
    "the", "this", "that", "your", "our", "their", "was", "is", "felt",
    "seemed", "looked", "again", "today", "yesterday", "quite", "rather",
    "honestly", "frankly", "overall", "mostly", "report", "meeting", "draft",
    "plan", "review", "update", "deadline", "team", "project", "idea",
    "comment", "email", "agenda", "schedule", "budget", "client", "design",
    "feature", "summary", "feedback", "proposal", "document", "section",
    "figure", "analysis", "result", "method", "presentation", "handover",
    "notes",
)


@dataclass(frozen=True)
class Corpus:
    """Texts, binary labels (1 = hostile), and implanted rationale indices."""

    texts: list[str]
    labels: list[int]
    rationales: list[list[int]]

    def __len__(self) -> int:
        return len(self.texts)


def generate(n: int = 240, seed: int = 0) -> Corpus:
    """Generate ``n`` texts, half hostile, half civil, shuffled by ``seed``.

    Texts are 8 to 14 space-joined lowercase words, so the default word
    tokenizer's indices coincide with word positions and the recorded
    rationale indices are valid token indices.
    """
    rng = np.random.default_rng(seed)
    texts: list[str] = []
    labels: list[int] = []
    rationales: list[list[int]] = []
    for i in range(n):
        label = 1 if i < n // 2 else 0
        lexicon = HOSTILE_LEXICON if label else CIVIL_LEXICON
        length = int(rng.integers(8, 15))
        words = [_FILLER[int(rng.integers(len(_FILLER)))] for _ in range(length)]
        n_signals = int(rng.integers(1, 4))
        positions = sorted(int(p) for p in rng.choice(length, size=n_signals, replace=False))
        for position in positions:
            words[position] = lexicon[int(rng.integers(len(lexicon)))]
        texts.append(" ".join(words))
        labels.append(label)
        rationales.append(positions)
    order = [int(i) for i in rng.permutation(n)]
    return Corpus(
        texts=[texts[i] for i in order],
        labels=[labels[i] for i in order],
        rationales=[rationales[i] for i in order],
    )


def train_test_split(corpus: Corpus, n_test: int = 40) -> tuple[Corpus, Corpus]:
    """Split an already-shuffled corpus into head-train and tail-test."""
    if not 0 < n_test < len(corpus):
        raise ValueError(f"n_test {n_test} outside corpus of {len(corpus)}")
    cut = len(corpus) - n_test
    return (
        Corpus(corpus.texts[:cut], corpus.labels[:cut], corpus.rationales[:cut]),
        Corpus(corpus.texts[cut:], corpus.labels[cut:], corpus.rationales[cut:]),
    )
