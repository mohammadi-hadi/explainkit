"""Missing extras fail with instructions; installed extras align with our spans."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from explainkit import Explainer, drop_spans
from test_occlusion import CountingLinearModel


def _missing(module: str) -> bool:
    return importlib.util.find_spec(module) is None


@pytest.mark.skipif(not _missing("shap"), reason="shap installed")
def test_shap_error_names_the_extra() -> None:
    with pytest.raises(ImportError, match=r"explainkit\[shap\]"):
        Explainer(CountingLinearModel()).explain("good idea", method="shap")


@pytest.mark.skipif(not _missing("lime"), reason="lime installed")
def test_lime_error_names_the_extra() -> None:
    with pytest.raises(ImportError, match=r"explainkit\[lime\]"):
        Explainer(CountingLinearModel()).explain("good idea", method="lime")


@pytest.mark.skipif(not _missing("transformers"), reason="transformers installed")
def test_hf_error_names_the_extra() -> None:
    from explainkit.hf import pipeline_predict_fn

    with pytest.raises(ImportError, match=r"explainkit\[transformers\]"):
        pipeline_predict_fn("any-model")


def test_shap_smoke_if_installed() -> None:
    pytest.importorskip("shap")
    explanation = Explainer(CountingLinearModel()).explain(
        "a good day, an awful meeting", method="shap", n_samples=64, seed=0, target=1
    )
    tokens = ["a", "good", "day", "an", "awful", "meeting"]
    assert explanation.tokens == tokens
    by_token = dict(zip(explanation.tokens, explanation.scores, strict=True))
    assert by_token["good"] > 0
    assert by_token["awful"] < 0


def test_shap_repeats_itself_with_the_same_seed() -> None:
    pytest.importorskip("shap")
    explainer = Explainer(CountingLinearModel())
    text = "a good day, an awful meeting, plain notes"
    runs = [
        explainer.explain(text, method="shap", n_samples=64, seed=0, target=1).scores
        for _ in range(5)
    ]
    assert all(run == runs[0] for run in runs)


class RecordingLinearModel(CountingLinearModel):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def __call__(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.seen += list(texts)
        return super().__call__(texts)


@pytest.mark.parametrize("mask", ["", "<x>", "two words"])
def test_shap_changes_only_the_dropped_words(mask: str) -> None:
    pytest.importorskip("shap")
    model = RecordingLinearModel()
    text = "\"A good day\", an awful meeting!"
    explanation = Explainer(model, mask=mask).explain(text, method="shap", n_samples=64, target=1)
    spans = explanation.spans
    allowed = {
        drop_spans(text, spans, [i for i in range(len(spans)) if not keep >> i & 1], mask)
        for keep in range(2 ** len(spans))
    }
    assert set(model.seen) <= allowed
    assert text in model.seen
    by_token = dict(zip(explanation.tokens, explanation.scores, strict=True))
    for token in ("A", "day", "an", "meeting"):
        assert by_token[token] == pytest.approx(0.0, abs=1e-12)
    assert explanation.base_value + sum(explanation.scores) == pytest.approx(
        explanation.prediction
    )


def test_shap_explains_a_one_word_text() -> None:
    pytest.importorskip("shap")
    model = CountingLinearModel()
    explanation = Explainer(model, mask="<x>").explain("good", method="shap", target=1)
    full, masked = model(["good", "<x>"])[:, 1]
    assert explanation.tokens == ["good"]
    assert explanation.scores == pytest.approx([full - masked])
    assert explanation.base_value == pytest.approx(masked)


def test_lime_smoke_if_installed() -> None:
    pytest.importorskip("lime")
    explanation = Explainer(CountingLinearModel()).explain(
        "a good day, an awful meeting", method="lime", n_samples=256, seed=0, target=1
    )
    assert explanation.tokens == ["a", "good", "day", "an", "awful", "meeting"]
    by_token = dict(zip(explanation.tokens, explanation.scores, strict=True))
    assert by_token["good"] > 0
    assert by_token["awful"] < 0
