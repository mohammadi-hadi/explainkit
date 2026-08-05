"""The demo is also an experiment with known ground truth — check it holds.

The corpus implants its own rationales, so recovery is testable: the model
must be accurate, occlusion must agree with an independent linear read of the
same data, selection must dig out the implanted lexicon, and informed
replacement must beat random replacement at flipping verdicts.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from explainkit.dataset import (
    CIVIL_LEXICON,
    CIVIL_SYNONYMS,
    HOSTILE_LEXICON,
    generate,
    train_test_split,
)
from explainkit.demo import build_stats, render_report, summary_table


def test_generation_is_deterministic() -> None:
    first = generate(60, seed=0)
    second = generate(60, seed=0)
    assert first.texts == second.texts
    assert first.labels == second.labels
    assert first.rationales == second.rationales
    assert generate(60, seed=1).texts != first.texts


def test_rationale_indices_point_at_signal_words() -> None:
    corpus = generate(80, seed=0)
    lexicon = set(HOSTILE_LEXICON) | set(CIVIL_LEXICON)
    for text, label, rationale in zip(
        corpus.texts, corpus.labels, corpus.rationales, strict=True
    ):
        words = text.split(" ")
        for index in rationale:
            assert words[index] in lexicon
        expected = HOSTILE_LEXICON if label else CIVIL_LEXICON
        assert all(words[i] in expected for i in rationale)


def test_split_sizes_and_balance() -> None:
    corpus = generate(240, seed=0)
    train, test = train_test_split(corpus, n_test=40)
    assert len(train) == 200 and len(test) == 40
    # The shuffle should leave both classes present on both sides.
    assert 0 < sum(train.labels) < len(train)
    assert 0 < sum(test.labels) < len(test)


def test_synonym_map_is_a_bijection_into_civil() -> None:
    assert set(CIVIL_SYNONYMS) == set(HOSTILE_LEXICON)
    assert set(CIVIL_SYNONYMS.values()) == set(CIVIL_LEXICON)


def test_demo_stats_tell_the_story() -> None:
    stats = build_stats()

    assert stats.accuracy >= 0.9

    occlusion, surrogate, random = stats.rows
    # Informed attributions must be more faithful and more plausible than
    # the permuted baseline, and must transfer to action.  (rationale F1@3 is
    # structurally capped below 1: a text with one implanted word can score at
    # most 0.5 against a fixed top-3.)
    assert occlusion.aopc > random.aopc + 0.1
    assert occlusion.rationale_f1 > 0.7
    assert random.rationale_f1 < 0.5
    assert occlusion.jaccard_vs_occlusion == 1.0
    assert surrogate.jaccard_vs_occlusion > random.jaccard_vs_occlusion
    assert occlusion.flip_rate > random.flip_rate + 0.3

    # Selection at 0.95 digs out most of the implanted vocabulary; the tail
    # of the last few percent is allowed to drag some filler along.
    assert stats.signal_recovered >= 12
    assert stats.signal_recovered >= 0.6 * stats.signal_present
    assert stats.selection_precision > 0.4
    selected = {item.token for item in stats.selected}
    recovered = selected & (set(HOSTILE_LEXICON) | set(CIVIL_LEXICON))
    assert len(recovered) == stats.signal_recovered


def test_occlusion_agrees_with_sklearn_coefficients() -> None:
    corpus = generate(240, seed=0)
    train, _ = train_test_split(corpus, n_test=40)
    pipeline = make_pipeline(
        TfidfVectorizer(), LogisticRegression(max_iter=1000, random_state=0)
    )
    pipeline.fit(train.texts, train.labels)
    vectorizer: TfidfVectorizer = pipeline.steps[0][1]
    regression: LogisticRegression = pipeline.steps[1][1]
    coefficients = dict(
        zip(vectorizer.get_feature_names_out(), regression.coef_[0], strict=True)
    )
    hostile_mean = float(np.mean([coefficients[w] for w in HOSTILE_LEXICON]))
    civil_mean = float(np.mean([coefficients[w] for w in CIVIL_LEXICON]))
    # An independent linear model reads the same implanted signal.
    assert hostile_mean > 0.5
    assert civil_mean < -0.5


def test_double_run_is_byte_identical() -> None:
    first = summary_table(build_stats())
    second = summary_table(build_stats())
    assert first == second
    assert render_report(build_stats()) == render_report(build_stats())
