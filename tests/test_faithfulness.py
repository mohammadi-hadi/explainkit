"""Identities first, inequalities second — the metrics must obey both."""

from __future__ import annotations

import numpy as np
import pytest

from explainkit import (
    Explainer,
    aopc,
    comprehensiveness,
    deletion_curve,
    permute_scores,
    sufficiency,
)
from test_occlusion import CountingLinearModel

TEXT = "a good draft, one great idea, one bad chart and an awful ending"


def _model_and_explanation():
    model = CountingLinearModel()
    return model, Explainer(model).explain(TEXT, target=1)


def test_comprehensiveness_at_zero_is_zero() -> None:
    model, explanation = _model_and_explanation()
    assert comprehensiveness(explanation, model, k=0) == pytest.approx(0.0)


def test_sufficiency_with_everything_kept_is_zero() -> None:
    model, explanation = _model_and_explanation()
    assert sufficiency(explanation, model, k=len(explanation.tokens)) == pytest.approx(0.0)


def test_aopc_is_mean_of_curve_drops() -> None:
    model, explanation = _model_and_explanation()
    curve = deletion_curve(explanation, model)
    drops = [explanation.prediction - p for _, p in curve]
    assert aopc(explanation, model) == pytest.approx(float(np.mean(drops)))


def test_deleting_positive_evidence_lowers_probability() -> None:
    model, explanation = _model_and_explanation()
    assert comprehensiveness(explanation, model, k=2) > 0.1


def test_curve_fractions_are_monotonic_in_deleted_mass() -> None:
    model, explanation = _model_and_explanation()
    curve = deletion_curve(explanation, model, fractions=(0.1, 0.3, 0.5))
    assert [f for f, _ in curve] == [0.1, 0.3, 0.5]
    probabilities = [p for _, p in curve]
    # Deleting more of the *best* tokens first can only hurt this model.
    assert probabilities[0] >= probabilities[-1]


def test_permuted_scores_are_less_faithful_on_average() -> None:
    model, explanation = _model_and_explanation()
    informed = comprehensiveness(explanation, model, k=2)
    permuted = [
        comprehensiveness(permute_scores(explanation, seed=seed), model, k=2)
        for seed in range(8)
    ]
    assert informed > float(np.mean(permuted))


def test_permute_keeps_multiset_and_marks_method() -> None:
    _, explanation = _model_and_explanation()
    permuted = permute_scores(explanation, seed=1)
    assert sorted(permuted.scores) == sorted(explanation.scores)
    assert permuted.method == "occlusion, permuted"
    assert permuted.tokens == explanation.tokens
