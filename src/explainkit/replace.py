"""Attribution-guided token replacement: edit what the explanation points at.

The move from arXiv:2506.04050 — find the tokens most influential to a
classifier, swap them, and watch whether the verdict survives.  A flip from a
handful of targeted edits is a robustness finding *and* a check on the
attribution itself: scores that cannot steer an intervention were not really
evidence.

A replacement strategy is any callable ``(span, full_text) -> str | None``:
return the replacement string, or ``None`` to leave the token alone.  The
paper's heavier strategies — Word2Vec synonyms from human text (HSR), the
POS-matched variant (PSR), dictionary APIs, an LLM — all fit this signature;
see ``examples/replace_hsr.py``.  The built-ins here need nothing beyond the
standard library.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from explainkit.explain import Explainer
from explainkit.schema import Explanation, Span

Strategy = Callable[[Span, str], str | None]


def mask_strategy(mask: str = "[MASK]") -> Strategy:
    """Replace every targeted token with a fixed mask string."""

    def strategy(span: Span, text: str) -> str | None:
        return mask

    return strategy


def delete_strategy() -> Strategy:
    """Delete every targeted token."""

    def strategy(span: Span, text: str) -> str | None:
        return ""

    return strategy


def synonym_strategy(
    mapping: Mapping[str, str | Sequence[str]],
    *,
    lowercase: bool = True,
    seed: int = 0,
) -> Strategy:
    """Replace tokens via a lookup table; unknown tokens are left alone.

    A value may be a single replacement or a sequence to sample from — the
    sampling is seeded once per strategy, so a full run is reproducible.
    """
    rng = np.random.default_rng(seed)

    def strategy(span: Span, text: str) -> str | None:
        key = span.text.lower() if lowercase else span.text
        replacement = mapping.get(key)
        if replacement is None or isinstance(replacement, str):
            return replacement
        return replacement[int(rng.integers(len(replacement)))]

    return strategy


@dataclass(frozen=True)
class Replacement:
    """One text after attribution-guided edits."""

    original: str
    text: str
    # (token index, original token, replacement) per edit, in text order.
    edits: list[tuple[int, str, str]]


def replace_tokens(
    explanation: Explanation,
    strategy: Strategy,
    *,
    top_k: int = 5,
    tokens: Sequence[str] | None = None,
) -> Replacement:
    """Rewrite the top-attributed tokens of one explained text.

    By default the ``top_k`` tokens of the explanation's own ranking are
    targeted; passing ``tokens`` (e.g. a corpus vocabulary from
    ``select_tokens``) targets every occurrence of those tokens instead.
    Edits splice the original character spans right to left, so replacing a
    token can never corrupt a longer word around it.
    """
    if tokens is not None:
        wanted = {t.lower() for t in tokens}
        indices = [i for i, t in enumerate(explanation.tokens) if t.lower() in wanted]
    else:
        indices = explanation.top_k(top_k)
    out = explanation.text
    edits: list[tuple[int, str, str]] = []
    for i in sorted(indices, reverse=True):
        start, end = explanation.spans[i]
        span = Span(explanation.tokens[i], start, end)
        replacement = strategy(span, explanation.text)
        if replacement is None or replacement == span.text:
            continue
        out = out[:start] + replacement + out[end:]
        edits.append((i, span.text, replacement))
    edits.reverse()
    return Replacement(original=explanation.text, text=out, edits=edits)


@dataclass(frozen=True)
class ProbeResult:
    """Before/after verdict for one text."""

    explanation: Explanation
    replacement: Replacement
    p_before: float
    p_after: float
    flipped: bool

    @property
    def drop(self) -> float:
        return self.p_before - self.p_after


@dataclass(frozen=True)
class RobustnessReport:
    """Corpus-level outcome of a replacement probe."""

    results: list[ProbeResult]

    @property
    def mean_drop(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.drop for r in self.results) / len(self.results)

    @property
    def flip_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.flipped for r in self.results) / len(self.results)


def robustness_probe(
    explainer: Explainer,
    texts: Sequence[str],
    strategy: Strategy,
    *,
    top_k: int = 5,
    method: str = "occlusion",
    target: int | None = None,
    n_samples: int = 256,
    seed: int = 0,
) -> RobustnessReport:
    """Explain each text, rewrite its top tokens, and re-classify the result.

    ``flipped`` records whether the argmax class moved off the original
    prediction — the paper's headline event for detector robustness.
    """
    results: list[ProbeResult] = []
    for text in texts:
        explanation = explainer.explain(
            text, method=method, target=target, n_samples=n_samples, seed=seed
        )
        replacement = replace_tokens(explanation, strategy, top_k=top_k)
        probs = np.asarray(explainer.predict_fn([replacement.text]), dtype=np.float64)
        p_after = float(probs[0, explanation.target_class])
        flipped = int(np.argmax(probs[0])) != explanation.predicted_class
        results.append(
            ProbeResult(
                explanation=explanation,
                replacement=replacement,
                p_before=explanation.prediction,
                p_after=p_after,
                flipped=flipped,
            )
        )
    return RobustnessReport(results=results)
