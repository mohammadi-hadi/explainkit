"""SHAP attribution over this package's own word spans.

The one design rule here: shap never tokenizes.  Its Text masker is handed a
tokenizer built from the caller's word spans, so the returned values align
one-to-one with the same tokens every other method scores.

Masked texts are built the way occlusion builds them: each dropped word's
span is replaced with the explainer's ``mask``, which deletes it by default,
and nothing else in the string changes.  shap's stock Text masker writes a
placeholder that sticks to the next word ("a good day" with "good" dropped
reaches the model as "a MASKday"), so here it only supplies the shape and the
partition tree.  Requires the ``shap`` extra.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from explainkit._optional import require
from explainkit.explain import drop_spans
from explainkit.schema import PredictFn, Span, Tokenizer

# shap's Text masker needs a token to look up when it is built, though the
# splicing masker below never writes it.
_INIT_TOKEN = "MASK"


def _masker_tokenizer(tokenizer: Tokenizer) -> Any:
    def tokenize(text: str, return_offsets_mapping: bool = True) -> dict[str, object]:
        spans = tokenizer(text)
        return {
            "input_ids": [span.text for span in spans],
            "offset_mapping": [(span.start, span.end) for span in spans],
        }

    return tokenize


def _splicing_masker(shap: Any, tokenizer: Tokenizer, mask: str) -> Any:
    """shap's Text masker, with masked texts spliced from the word spans."""

    class SplicingText(shap.maskers.Text):  # type: ignore[misc]
        def __call__(self, keep: Any, s: str) -> tuple[Any, ...]:
            keep = np.asarray(self._standardize_mask(keep, s), dtype=bool)
            spans = [(span.start, span.end) for span in tokenizer(s)]
            dropped = [int(i) for i in np.flatnonzero(~keep)]
            return (np.array([drop_spans(s, spans, dropped, mask)]),)

    return SplicingText(
        _masker_tokenizer(tokenizer), mask_token=_INIT_TOKEN, collapse_mask_token=False
    )


def _warn_old_shap(version: str) -> None:
    if tuple(int(n) for n in re.findall(r"\d+", version)[:2]) >= (0, 53):
        return
    # Before 0.53 the partition-tree shuffle drew from numba's generator,
    # which np.random.seed never reaches.
    message = f"shap {version} ignores the seed for text; upgrade to shap>=0.53"
    warnings.warn(message, stacklevel=4)


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

    A dropped word is replaced with ``mask``; the empty string deletes it.
    """
    shap = require("shap", extra="shap")
    _warn_old_shap(shap.__version__)

    def f(batch: Sequence[str]) -> Any:
        return np.asarray(predict_fn(list(batch)), dtype=np.float64)[:, target]

    offsets = [(span.start, span.end) for span in spans]
    if len(spans) == 1:
        # One word has one exact Shapley value, the full change from the
        # masked text.  shap can't build its partition tree over a single
        # token and raises a zero-size array error instead.
        both = f([text, drop_spans(text, offsets, [0], mask)])
        return [float(both[0] - both[1])], float(both[1])
    masker = _splicing_masker(shap, tokenizer, mask)
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
