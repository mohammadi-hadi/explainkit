"""The Explanation model refuses misalignment, and JSONL round-trips exactly."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from explainkit import Explanation, dump_explanations, load_explanations


def make(**overrides: object) -> Explanation:
    fields: dict[str, object] = {
        "text": "a fine day",
        "tokens": ["a", "fine", "day"],
        "spans": [(0, 1), (2, 6), (7, 10)],
        "scores": [0.1, 0.5, -0.2],
        "base_value": 0.5,
        "prediction": 0.9,
        "predicted_class": 1,
        "target_class": 1,
        "method": "occlusion",
    }
    fields.update(overrides)
    return Explanation(**fields)  # type: ignore[arg-type]


def test_valid_explanation_builds() -> None:
    explanation = make()
    assert explanation.tokens == ["a", "fine", "day"]


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="misaligned"):
        make(scores=[0.1, 0.5])


def test_span_outside_text_rejected() -> None:
    with pytest.raises(ValidationError, match="outside text"):
        make(spans=[(0, 1), (2, 6), (7, 99)])


def test_span_reading_wrong_token_rejected() -> None:
    with pytest.raises(ValidationError, match="not"):
        make(spans=[(0, 1), (2, 6), (6, 9)])


def test_ranking_stable_under_ties() -> None:
    explanation = make(scores=[0.5, 0.5, 0.5])
    assert explanation.ranking() == [0, 1, 2]
    assert explanation.top_k(2) == [0, 1]
    assert explanation.top_k(0) == []


def test_jsonl_round_trip_is_byte_identical(tmp_path: Path) -> None:
    explanations = [make(), make(scores=[0.0, 0.0, 1.0], method="surrogate")]
    path = tmp_path / "explanations.jsonl"
    dump_explanations(explanations, path)
    first = path.read_bytes()
    loaded = load_explanations(path)
    assert loaded == explanations
    dump_explanations(loaded, path)
    assert path.read_bytes() == first


def test_load_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(make().model_dump_json() + '\n{"text": "broken"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_explanations(path)
