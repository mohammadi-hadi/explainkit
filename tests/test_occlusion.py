"""Occlusion on a linear model must recover the model's own coefficients."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from explainkit import Explainer, word_tokenizer

# A transparent "model": logit = sum of per-word weights, so the effect of
# removing one word is known exactly.
WEIGHTS = {"good": 2.0, "great": 3.0, "bad": -2.5, "awful": -4.0}


class CountingLinearModel:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.calls.append(len(texts))
        out = np.empty((len(texts), 2), dtype=np.float64)
        for row, text in enumerate(texts):
            logit = sum(
                WEIGHTS.get(span.text.lower(), 0.0) for span in word_tokenizer(text)
            )
            p = 1.0 / (1.0 + np.exp(-logit))
            out[row] = (1.0 - p, p)
        return out


def test_scores_recover_coefficient_signs() -> None:
    model = CountingLinearModel()
    explanation = Explainer(model).explain(
        "a good day, an awful meeting, plain notes", target=1
    )
    by_token = dict(zip(explanation.tokens, explanation.scores, strict=True))
    assert by_token["good"] > 0
    assert by_token["awful"] < 0
    for token in ("a", "day", "plain", "notes"):
        assert by_token[token] == pytest.approx(0.0, abs=1e-12)


def test_one_batched_call_of_n_plus_2_rows() -> None:
    model = CountingLinearModel()
    text = "good bad great"
    Explainer(model).explain(text)
    assert model.calls == [3 + 2]


def test_target_override_flips_scores() -> None:
    model = CountingLinearModel()
    explainer = Explainer(model)
    toward_positive = explainer.explain("a good idea", target=1)
    toward_negative = explainer.explain("a good idea", target=0)
    for one, other in zip(toward_positive.scores, toward_negative.scores, strict=True):
        assert one == pytest.approx(-other)
    assert toward_positive.predicted_class == toward_negative.predicted_class == 1


def test_deterministic_across_runs() -> None:
    explainer = Explainer(CountingLinearModel())
    first = explainer.explain("good bad awful notes")
    second = explainer.explain("good bad awful notes")
    assert first == second


def test_empty_text_yields_empty_explanation() -> None:
    explanation = Explainer(CountingLinearModel()).explain("... !!!")
    assert explanation.tokens == []
    assert explanation.scores == []
    assert explanation.prediction == pytest.approx(explanation.base_value)


def test_bad_predict_shape_rejected() -> None:
    def broken(texts: Sequence[str]) -> NDArray[np.float64]:
        return np.zeros(len(texts), dtype=np.float64)

    with pytest.raises(ValueError, match="expected"):
        Explainer(broken).explain("some text")


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        Explainer(CountingLinearModel()).explain("text", method="gradients")
