"""LIME attribution mapped back onto this package's word spans.

LIME wants to do its own splitting, and its ``split_expression`` names the
*separators*, not the words — a pattern that cannot reproduce every
tokenizer.  So this backend does not fight it: LIME explains a
whitespace-joined rendering of the package's tokens, split on whitespace,
which makes its feature positions equal to token indices by construction,
for any tokenizer whose tokens contain no whitespace.  Punctuation between
tokens is not part of LIME's perturbed variants; scores still attach to the
original spans.  Requires the ``lime`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from explainkit._optional import require
from explainkit.schema import PredictFn, Span


def lime_scores(
    predict_fn: PredictFn,
    text: str,
    spans: Sequence[Span],
    target: int,
    n_samples: int,
    seed: int,
) -> tuple[list[float], float]:
    """LIME weights for one text, aligned to ``spans``; returns (scores, base)."""
    lime_text = require("lime.lime_text", extra="lime")

    tokens = [span.text for span in spans]
    if any(any(c.isspace() for c in token) for token in tokens):
        raise ValueError("the lime backend needs whitespace-free tokens")
    joined = " ".join(tokens)

    def f(batch: Sequence[str]) -> Any:
        return np.asarray(predict_fn(list(batch)), dtype=np.float64)

    explainer = lime_text.LimeTextExplainer(
        bow=False,
        split_expression=r"\s+",
        mask_string="",
        random_state=seed,
    )
    explanation = explainer.explain_instance(
        joined,
        f,
        labels=(target,),
        num_features=max(len(spans), 1),
        num_samples=n_samples,
    )
    indexed = explanation.domain_mapper.indexed_string
    if indexed.num_words() != len(spans):
        raise ValueError(
            f"lime split {indexed.num_words()} tokens where the package has "
            f"{len(spans)}; tokens must be whitespace-free"
        )
    scores = [0.0] * len(spans)
    for position, weight in explanation.as_map()[target]:
        scores[int(position)] = float(weight)
    return scores, float(explanation.intercept[target])
