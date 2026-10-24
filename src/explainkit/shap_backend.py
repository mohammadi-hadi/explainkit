"""SHAP attribution over this package's own word spans.

The one design rule here: shap never tokenizes.  Its Text masker is handed a
tokenizer built from the caller's word spans, so the returned values align
one-to-one with the same tokens every other method scores.

One mechanical difference from occlusion: the Text masker replaces dropped
words with a placeholder word (``MASK``) rather than deleting them, because
it requires a mask token that survives tokenization.  Cross-method
comparisons are rank-based, where this does not matter.  Requires the
``shap`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from explainkit._optional import require
from explainkit.schema import PredictFn, Span, Tokenizer

MASK_WORD = "MASK"


def _masker_tokenizer(tokenizer: Tokenizer) -> Any:
    def tokenize(text: str, return_offsets_mapping: bool = True) -> dict[str, object]:
        spans = tokenizer(text)
        return {
            "input_ids": [span.text for span in spans],
            "offset_mapping": [(span.start, span.end) for span in spans],
        }

    return tokenize


def shap_scores(
    predict_fn: PredictFn,
    tokenizer: Tokenizer,
    mask: str,
    text: str,
    spans: Sequence[Span],
    target: int,
    n_samples: int,
    seed: int,
) -> tuple[list[float], float]:
    """SHAP values for one text, aligned to ``spans``; returns (scores, base).

    ``mask`` is used when it tokenizes to exactly one word; otherwise the
    placeholder ``MASK`` stands in (the Text masker cannot delete outright).
    """
    shap = require("shap", extra="shap")

    def f(batch: Sequence[str]) -> Any:
        return np.asarray(predict_fn(list(batch)), dtype=np.float64)[:, target]

    mask_word = mask if len(tokenizer(mask)) == 1 else MASK_WORD
    if len(spans) == 1:
        # One word has one exact Shapley value, the full change from the
        # all-masked text.  shap's Text masker can't build its partition tree
        # over a single token and raises a zero-size array error instead.
        start, end = spans[0].start, spans[0].end
        both = f([text, text[:start] + mask_word + text[end:]])
        return [float(both[0] - both[1])], float(both[1])
    masker = shap.maskers.Text(
        _masker_tokenizer(tokenizer), mask_token=mask_word, collapse_mask_token=False
    )
    explainer = shap.Explainer(f, masker, algorithm="permutation", seed=seed)
    # The permutation explainer needs at least one full forward-and-back pass
    # over the tokens.
    max_evals = max(n_samples, 2 * len(spans) + 4)
    values = explainer([text], max_evals=max_evals, silent=True)
    scores = [float(v) for v in np.ravel(values.values[0])]
    if len(scores) != len(spans):
        raise ValueError(
            f"shap returned {len(scores)} values for {len(spans)} tokens; "
            "the masker did not honor the package tokenization"
        )
    base = float(np.ravel(values.base_values)[0])
    return scores, base
