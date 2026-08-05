"""SI, IR and the cumulative cut, hand-computed against the paper's formulas."""

from __future__ import annotations

import pytest

from explainkit import Explanation, TokenImportance, select_tokens, token_importance


def explanation(text: str, scores: list[float], predicted: int) -> Explanation:
    tokens = text.split(" ")
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
        predicted_class=predicted,
        target_class=predicted,
        method="occlusion",
    )


def test_si_and_ir_hand_computed() -> None:
    # "bad" appears twice, once in a correctly-predicted text (|score| 0.6)
    # and once in a wrongly-predicted one (|score| 0.4).  The wrong instance
    # contributes nothing to the sum but still counts in N_t:
    # SI(bad) = 0.6 / 2 = 0.3.  For "fine": SI = 0.2 / 1 = 0.2.
    first = explanation("bad fine", [0.6, 0.2], predicted=1)
    second = explanation("bad news", [-0.4, 0.1], predicted=0)
    importances = {
        item.token: item for item in token_importance([first, second], y_true=[1, 1])
    }
    assert importances["bad"].si == pytest.approx(0.3)
    assert importances["bad"].n == 2
    assert importances["fine"].si == pytest.approx(0.2)
    # "news" occurs only in the wrong prediction: zero importance.
    assert importances["news"].si == pytest.approx(0.0)
    total = sum(item.si for item in importances.values())
    for item in importances.values():
        assert item.ir == pytest.approx(item.si / total)


def test_ir_sums_to_one() -> None:
    first = explanation("good plan today", [0.5, 0.3, 0.1], predicted=1)
    second = explanation("good idea again", [0.4, 0.2, 0.1], predicted=1)
    importances = token_importance([first, second], y_true=[1, 1])
    assert sum(item.ir for item in importances) == pytest.approx(1.0)


def test_correctness_gate_can_be_disabled() -> None:
    wrong = explanation("bad day", [0.8, 0.1], predicted=0)
    gated = token_importance([wrong], y_true=[1])
    ungated = token_importance([wrong], y_true=[1], correct_only=False)
    assert all(item.si == 0.0 for item in gated)
    assert {item.token: item.si for item in ungated}["bad"] == pytest.approx(0.8)


def test_lowercase_merging() -> None:
    first = explanation("Bad bad", [0.4, 0.2], predicted=1)
    merged = token_importance([first], y_true=[1])
    assert len(merged) == 1
    assert merged[0].token == "bad"
    assert merged[0].n == 2
    assert merged[0].si == pytest.approx(0.3)


def test_selection_respects_threshold() -> None:
    importances = [
        TokenImportance("a", si=0.5, ir=0.5, n=1),
        TokenImportance("b", si=0.3, ir=0.3, n=1),
        TokenImportance("c", si=0.15, ir=0.15, n=1),
        TokenImportance("d", si=0.05, ir=0.05, n=1),
    ]
    selected = select_tokens(importances, threshold=0.95)
    assert [item.token for item in selected] == ["a", "b", "c"]
    cumulative = sum(item.ir for item in selected)
    assert cumulative <= 0.95 + 1e-12
    assert cumulative + 0.05 > 0.95 + 1e-12


def test_top_token_always_kept() -> None:
    importances = [
        TokenImportance("dominant", si=0.99, ir=0.99, n=3),
        TokenImportance("rest", si=0.01, ir=0.01, n=1),
    ]
    selected = select_tokens(importances, threshold=0.5)
    assert [item.token for item in selected] == ["dominant"]


def test_threshold_validation_and_label_mismatch() -> None:
    with pytest.raises(ValueError, match="threshold"):
        select_tokens([], threshold=0.0)
    with pytest.raises(ValueError, match="labels"):
        token_importance([explanation("a b", [0.1, 0.2], predicted=1)], y_true=[1, 0])
