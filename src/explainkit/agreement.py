"""Do two attribution methods tell the same story about the same text?

Raw magnitudes are not comparable across methods — SHAP values are additive
contributions, occlusion scores are probability drops — so agreement is
measured on ranks and sets only.  Both correlations are implemented here in
plain numpy with full tie handling; the test suite checks them against scipy
to 1e-10 without the library depending on scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from explainkit.schema import Explanation


def _same_tokens(a: Explanation, b: Explanation) -> None:
    if a.text != b.text or a.tokens != b.tokens:
        raise ValueError(
            "agreement needs two explanations of the same text with the same "
            "tokenization; got different texts or token lists"
        )


def _average_ranks(values: Sequence[float]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    i = 0
    while i < len(array):
        j = i
        while j + 1 < len(array) and array[order[j + 1]] == array[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = _average_ranks(x), _average_ranks(y)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float(dx @ dx) * float(dy @ dy))
    if denominator == 0.0:
        return math.nan
    return float(dx @ dy) / denominator


def _kendall(x: Sequence[float], y: Sequence[float]) -> float:
    # Kendall's tau-b, the tie-corrected variant scipy computes by default.
    n = len(x)
    concordant = discordant = 0
    ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    pairs = n * (n - 1) // 2
    denominator = math.sqrt(
        (pairs - _tie_pairs(x)) * (pairs - _tie_pairs(y))
    )
    if denominator == 0.0:
        return math.nan
    return (concordant - discordant) / denominator


def _tie_pairs(values: Sequence[float]) -> int:
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sum(c * (c - 1) // 2 for c in counts.values())


def rank_correlation(
    a: Explanation,
    b: Explanation,
    *,
    method: Literal["spearman", "kendall"] = "spearman",
) -> float:
    """Rank correlation of two score vectors over the same tokens.

    Returns NaN when either vector is constant (correlation is undefined
    there, and pretending otherwise hides a degenerate explanation).
    """
    _same_tokens(a, b)
    if len(a.scores) < 2:
        return math.nan
    if method == "kendall":
        return _kendall(a.scores, b.scores)
    return _spearman(a.scores, b.scores)


def topk_jaccard(a: Explanation, b: Explanation, *, k: int) -> float:
    """Overlap of the two top-``k`` sets: intersection over union."""
    _same_tokens(a, b)
    top_a, top_b = set(a.top_k(k)), set(b.top_k(k))
    union = top_a | top_b
    if not union:
        return 1.0
    return len(top_a & top_b) / len(union)
