"""Explanations: what an attribution method claims about one prediction.

An explanation is a set of word-level scores over one input text, aimed at one
class of one classifier.  Everything downstream — selection, steering,
replacement, faithfulness — consumes this record, so it carries the alignment
data needed to reconstruct perturbed inputs exactly: character spans.  The
model validates that every span really is the token it claims to be, which is
what lets the rest of the package splice text instead of string-replacing it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

MetaValue = str | int | float | bool


class Span(NamedTuple):
    """One token and where it lives in the source text."""

    text: str
    start: int
    end: int


class PredictFn(Protocol):
    """Any text classifier: a batch of texts in, class probabilities out.

    The returned array has shape ``(len(texts), n_classes)``; each row should
    sum to 1.  This is the package's only contract with the model — sklearn
    pipelines, HF pipelines (via ``explainkit.hf``), or a hand-rolled closure
    all qualify.
    """

    def __call__(self, texts: Sequence[str]) -> NDArray[np.float64]: ...


class Tokenizer(Protocol):
    """Split text into word spans; ``text[start:end]`` must equal ``text``."""

    def __call__(self, text: str) -> list[Span]: ...


class Explanation(BaseModel):
    """Word-level attribution scores for one text, one target class.

    ``scores[i]`` is evidence that token ``i`` pushes the prediction *toward*
    ``target_class``; the scale depends on ``method`` (occlusion scores are
    probability drops, SHAP values are additive contributions), so comparisons
    across methods should use ranks, never raw magnitudes.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    tokens: list[str]
    spans: list[tuple[int, int]]
    scores: list[float]
    # p(target_class) with every token dropped / masked out.
    base_value: float
    # p(target_class) on the untouched text.
    prediction: float
    predicted_class: int
    target_class: int
    method: str
    meta: dict[str, MetaValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _aligned(self) -> Explanation:
        if not len(self.tokens) == len(self.spans) == len(self.scores):
            raise ValueError(
                f"misaligned explanation: {len(self.tokens)} tokens, "
                f"{len(self.spans)} spans, {len(self.scores)} scores"
            )
        for token, (start, end) in zip(self.tokens, self.spans, strict=True):
            if not 0 <= start < end <= len(self.text):
                raise ValueError(f"span ({start}, {end}) outside text of length {len(self.text)}")
            if self.text[start:end] != token:
                raise ValueError(
                    f"span ({start}, {end}) reads {self.text[start:end]!r}, not {token!r}"
                )
        return self

    def ranking(self) -> list[int]:
        """Token indices by score descending; ties keep text order."""
        return sorted(range(len(self.scores)), key=lambda i: (-self.scores[i], i))

    def top_k(self, k: int) -> list[int]:
        """The ``k`` tokens with the strongest evidence for the target class."""
        return self.ranking()[: max(k, 0)]
