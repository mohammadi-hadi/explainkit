"""Missing extras fail with instructions; installed extras align with our spans."""

from __future__ import annotations

import importlib.util

import pytest

from explainkit import Explainer
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


def test_lime_smoke_if_installed() -> None:
    pytest.importorskip("lime")
    explanation = Explainer(CountingLinearModel()).explain(
        "a good day, an awful meeting", method="lime", n_samples=256, seed=0, target=1
    )
    assert explanation.tokens == ["a", "good", "day", "an", "awful", "meeting"]
    by_token = dict(zip(explanation.tokens, explanation.scores, strict=True))
    assert by_token["good"] > 0
    assert by_token["awful"] < 0
