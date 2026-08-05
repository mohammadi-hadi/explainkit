"""The linear surrogate must recover a linear model's signal, deterministically.

The model's zero-weight filler words get exactly 0 from occlusion but small
sampling noise from the surrogate, so whole-vector rank correlation is only a
sanity floor; the sharp assertions are on the words that carry weight.
"""

from __future__ import annotations

import math

from explainkit import Explainer, rank_correlation, topk_jaccard
from test_occlusion import CountingLinearModel

TEXT = "a good draft, one great idea, one bad chart and an awful ending"


def test_recovers_linear_signal() -> None:
    explainer = Explainer(CountingLinearModel())
    occlusion = explainer.explain(TEXT, target=1)
    surrogate = explainer.explain(TEXT, method="surrogate", n_samples=512, seed=0, target=1)
    assert rank_correlation(occlusion, surrogate, method="spearman") > 0.6
    # The two positive words are the top 2 for both methods, exactly.
    assert topk_jaccard(occlusion, surrogate, k=2) == 1.0


def test_signed_words_get_signed_scores() -> None:
    surrogate = Explainer(CountingLinearModel()).explain(
        TEXT, method="surrogate", n_samples=512, seed=0, target=1
    )
    by_token = dict(zip(surrogate.tokens, surrogate.scores, strict=True))
    assert by_token["good"] > 0
    assert by_token["great"] > 0
    assert by_token["bad"] < 0
    assert by_token["awful"] < 0
    assert by_token["awful"] < by_token["bad"]
    assert abs(by_token["great"]) > abs(by_token["draft"])


def test_same_seed_same_explanation() -> None:
    explainer = Explainer(CountingLinearModel())
    first = explainer.explain(TEXT, method="surrogate", n_samples=128, seed=7)
    second = explainer.explain(TEXT, method="surrogate", n_samples=128, seed=7)
    assert first == second


def test_different_seed_different_samples() -> None:
    explainer = Explainer(CountingLinearModel())
    first = explainer.explain(TEXT, method="surrogate", n_samples=128, seed=1)
    second = explainer.explain(TEXT, method="surrogate", n_samples=128, seed=2)
    assert first.scores != second.scores


def test_base_value_and_metadata() -> None:
    surrogate = Explainer(CountingLinearModel()).explain(
        TEXT, method="surrogate", n_samples=64, seed=0
    )
    assert math.isfinite(surrogate.base_value)
    assert surrogate.meta == {"n_samples": 64, "seed": 0}
