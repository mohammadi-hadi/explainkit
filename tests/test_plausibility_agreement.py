"""Hand-computed plausibility cases; correlations cross-checked against scipy.

scipy is a dev dependency only — the library computes its correlations in
plain numpy, and this file is where those implementations answer to the
reference ones.
"""

from __future__ import annotations

import math

import pytest
from scipy import stats

from explainkit import (
    Explanation,
    rank_correlation,
    rationale_f1,
    rationale_iou,
    topk_jaccard,
)


def explanation(scores: list[float], tokens: list[str] | None = None) -> Explanation:
    tokens = tokens or [f"w{i}" for i in range(len(scores))]
    text = " ".join(tokens)
    spans = []
    cursor = 0
    for token in tokens:
        start = text.index(token, cursor)
        spans.append((start, start + len(token)))
        cursor = start + len(token)
    return Explanation(
        text=text,
        tokens=tokens,
        spans=spans,
        scores=scores,
        base_value=0.5,
        prediction=0.9,
        predicted_class=1,
        target_class=1,
        method="occlusion",
    )


# ------------------------------------------------------------ plausibility ---


def test_f1_hand_computed() -> None:
    # top-2 = {w0, w1}; gold = {w1, w2}.  P = R = 0.5 -> F1 = 0.5.
    e = explanation([0.9, 0.8, 0.1])
    assert rationale_f1(e, [1, 2], k=2) == pytest.approx(0.5)
    assert rationale_iou(e, [1, 2], k=2) == pytest.approx(1 / 3)


def test_perfect_and_disjoint() -> None:
    e = explanation([0.9, 0.8, 0.1, 0.0])
    assert rationale_f1(e, [0, 1], k=2) == pytest.approx(1.0)
    assert rationale_iou(e, [0, 1], k=2) == pytest.approx(1.0)
    assert rationale_f1(e, [2, 3], k=2) == pytest.approx(0.0)
    assert rationale_iou(e, [2, 3], k=2) == pytest.approx(0.0)


def test_string_and_index_rationales_agree() -> None:
    e = explanation([0.9, 0.8, 0.1], tokens=["Useless", "plan", "today"])
    assert rationale_f1(e, ["useless"], k=1) == rationale_f1(e, [0], k=1) == 1.0


def test_empty_cases_and_bad_index() -> None:
    e = explanation([0.5, 0.4])
    assert rationale_f1(e, [], k=0) == 1.0
    assert rationale_f1(e, [0], k=0) == 0.0
    with pytest.raises(ValueError, match="rationale index"):
        rationale_f1(e, [9], k=1)


# --------------------------------------------------------------- agreement ---


def test_spearman_matches_scipy_with_ties() -> None:
    x = [0.3, 0.1, 0.1, 0.9, -0.2, 0.9, 0.0]
    y = [0.1, 0.2, 0.2, 0.7, -0.1, 0.3, 0.05]
    ours = rank_correlation(explanation(x), explanation(y), method="spearman")
    reference = float(stats.spearmanr(x, y).statistic)
    assert ours == pytest.approx(reference, abs=1e-10)


def test_kendall_matches_scipy_with_ties() -> None:
    x = [0.3, 0.1, 0.1, 0.9, -0.2, 0.9, 0.0]
    y = [0.1, 0.2, 0.2, 0.7, -0.1, 0.3, 0.05]
    ours = rank_correlation(explanation(x), explanation(y), method="kendall")
    reference = float(stats.kendalltau(x, y).statistic)
    assert ours == pytest.approx(reference, abs=1e-10)


def test_perfect_and_inverted() -> None:
    a = explanation([0.1, 0.2, 0.3, 0.4])
    assert rank_correlation(a, a) == pytest.approx(1.0)
    b = explanation([0.4, 0.3, 0.2, 0.1])
    assert rank_correlation(a, b) == pytest.approx(-1.0)
    assert rank_correlation(a, b, method="kendall") == pytest.approx(-1.0)


def test_constant_scores_give_nan() -> None:
    a = explanation([0.1, 0.2, 0.3])
    constant = explanation([0.5, 0.5, 0.5])
    assert math.isnan(rank_correlation(a, constant))
    assert math.isnan(rank_correlation(a, constant, method="kendall"))


def test_mismatched_tokens_rejected() -> None:
    a = explanation([0.1, 0.2])
    b = explanation([0.1, 0.2], tokens=["other", "words"])
    with pytest.raises(ValueError, match="same text"):
        rank_correlation(a, b)
    with pytest.raises(ValueError, match="same text"):
        topk_jaccard(a, b, k=1)


def test_topk_jaccard() -> None:
    a = explanation([0.9, 0.8, 0.1, 0.0])
    b = explanation([0.9, 0.0, 0.8, 0.1])
    # top-2 sets: {0,1} vs {0,2} -> intersection 1, union 3.
    assert topk_jaccard(a, b, k=2) == pytest.approx(1 / 3)
    assert topk_jaccard(a, a, k=2) == pytest.approx(1.0)
    assert topk_jaccard(a, b, k=0) == pytest.approx(1.0)
