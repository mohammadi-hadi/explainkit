"""Highlighting and replacement must touch whole tokens and nothing else."""

from __future__ import annotations

import numpy as np
import pytest

from explainkit import (
    Explainer,
    delete_strategy,
    highlight,
    label_distribution,
    majority_vote,
    mask_strategy,
    replace_tokens,
    robustness_probe,
    synonym_strategy,
)
from test_occlusion import CountingLinearModel


def test_highlight_wraps_every_occurrence() -> None:
    out = highlight("good art, good party", ["good"])
    assert out == "**good** art, **good** party"


def test_highlight_never_marks_substrings() -> None:
    # "art" is a token of its own and a substring of "party"; only the token
    # may light up.  This is the regression test for the str.replace bug class.
    out = highlight("art in the party", ["art"])
    assert out == "**art** in the party"


def test_highlight_case_insensitive_by_default() -> None:
    assert highlight("Good idea", ["good"]) == "**Good** idea"
    assert highlight("Good idea", ["good"], lowercase=False) == "Good idea"


def test_majority_vote_and_tie_break() -> None:
    votes = [["yes", "no", "yes"], ["no", "yes"], ["tie", "tie", "other"]]
    assert majority_vote(votes) == ["yes", "no", "tie"]
    with pytest.raises(ValueError, match="no votes"):
        majority_vote([[]])


def test_label_distribution() -> None:
    (distribution,) = label_distribution([["a", "a", "b", "c"]])
    assert distribution == {"a": 0.5, "b": 0.25, "c": 0.25}


def _explanation(text: str = "an awful draft, a bad chart, one good note"):
    return Explainer(CountingLinearModel()).explain(text, target=1)


def test_replace_edits_only_selected_spans() -> None:
    explanation = _explanation()
    replaced = replace_tokens(explanation, mask_strategy("[X]"), top_k=1)
    assert replaced.text == "an awful draft, a bad chart, one [X] note"
    assert replaced.edits == [(explanation.top_k(1)[0], "good", "[X]")]


def test_replace_multiple_spans_right_to_left() -> None:
    explanation = _explanation()
    # Target the two negative words explicitly, whatever their rank.
    replaced = replace_tokens(explanation, delete_strategy(), tokens=["awful", "bad"])
    assert replaced.text == "an  draft, a  chart, one good note"
    assert [edit[1] for edit in replaced.edits] == ["awful", "bad"]


def test_replace_substring_token_cannot_corrupt_longer_word() -> None:
    explanation = _explanation("art in the party")
    replaced = replace_tokens(explanation, mask_strategy("craft"), tokens=["art"])
    assert replaced.text == "craft in the party"


def test_strategy_none_leaves_token_alone() -> None:
    explanation = _explanation()
    replaced = replace_tokens(explanation, synonym_strategy({"bad": "fine"}), top_k=99)
    assert replaced.text == "an awful draft, a fine chart, one good note"
    assert replaced.edits == [(4, "bad", "fine")]


def test_synonym_strategy_sequence_is_seeded() -> None:
    strategy_a = synonym_strategy({"bad": ("poor", "weak", "rough")}, seed=3)
    strategy_b = synonym_strategy({"bad": ("poor", "weak", "rough")}, seed=3)
    explanation = _explanation()
    first = replace_tokens(explanation, strategy_a, tokens=["bad"])
    second = replace_tokens(explanation, strategy_b, tokens=["bad"])
    assert first.text == second.text


def test_robustness_probe_flips_linear_model() -> None:
    report = robustness_probe(
        Explainer(CountingLinearModel()),
        ["a good great note", "plain notes today"],
        synonym_strategy({"good": "awful", "great": "bad"}),
        top_k=2,
    )
    assert report.results[0].flipped
    assert not report.results[1].flipped
    assert report.flip_rate == pytest.approx(0.5)
    assert report.mean_drop == pytest.approx(
        float(np.mean([r.p_before - r.p_after for r in report.results]))
    )
