"""Corpus-level token selection, as specified in Mohammadi et al. (2025).

From a set of explanations, compute each token's importance and keep the
smallest set of tokens that carries a target share of the total:

- ``SI_t``: mean absolute score of token ``t``, where only instances the
  model classified *correctly* contribute to the sum — a wrong prediction's
  attribution is noise about the decision boundary, not evidence — while the
  divisor ``N_t`` counts every occurrence (GeBNLP @ ACL 2025, §3.3).
- ``IR_t = SI_t / Σ_k SI_k``: the importance ratio, summing to 1.
- Selection keeps tokens in IR order while the cumulative ratio stays within
  the threshold ``T_c`` (0.95 in the paper).

The selected vocabulary is what ``explainkit.steer`` highlights in prompts
and what ``explainkit.replace`` targets across a corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from explainkit.schema import Explanation


@dataclass(frozen=True)
class TokenImportance:
    """One token's corpus-level importance."""

    token: str
    si: float  # mean |score| over correctly-classified occurrences
    ir: float  # share of total importance, sums to 1 over the corpus
    n: int  # occurrences across all explanations


def token_importance(
    explanations: Sequence[Explanation],
    y_true: Sequence[int],
    *,
    correct_only: bool = True,
    lowercase: bool = True,
) -> list[TokenImportance]:
    """Aggregate per-token importance over a corpus of explanations.

    ``y_true`` aligns with ``explanations``; the correctness gate compares it
    against each explanation's ``predicted_class``.  Returns tokens sorted by
    importance ratio descending, ties alphabetical, so output order is stable.
    """
    if len(explanations) != len(y_true):
        raise ValueError(
            f"{len(explanations)} explanations but {len(y_true)} labels"
        )
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for explanation, label in zip(explanations, y_true, strict=True):
        credited = not correct_only or explanation.predicted_class == label
        for token, score in zip(explanation.tokens, explanation.scores, strict=True):
            key = token.lower() if lowercase else token
            counts[key] = counts.get(key, 0) + 1
            if credited:
                sums[key] = sums.get(key, 0.0) + abs(score)
    si = {token: sums.get(token, 0.0) / counts[token] for token in counts}
    total = sum(si.values())
    return sorted(
        (
            TokenImportance(
                token=token,
                si=si[token],
                ir=si[token] / total if total > 0 else 0.0,
                n=counts[token],
            )
            for token in counts
        ),
        key=lambda item: (-item.ir, item.token),
    )


def select_tokens(
    importances: Sequence[TokenImportance], *, threshold: float = 0.95
) -> list[TokenImportance]:
    """The smallest importance-ordered prefix covering ``threshold`` of the total.

    Tokens are taken in importance-ratio order while the cumulative ratio
    stays within the threshold; the top token is always kept, even when it
    alone exceeds the threshold.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    ordered = sorted(importances, key=lambda item: (-item.ir, item.token))
    selected: list[TokenImportance] = []
    cumulative = 0.0
    for importance in ordered:
        if selected and cumulative + importance.ir > threshold + 1e-12:
            break
        selected.append(importance)
        cumulative += importance.ir
    return selected
