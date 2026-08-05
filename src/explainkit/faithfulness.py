"""Does the model actually care about the tokens the explanation points at?

Faithfulness metrics perturb the input according to the explanation's own
ranking and watch the target probability.  All of them rebuild variants with
the same span splicing the explainer used, so a SHAP explanation and an
occlusion explanation are scored under identical rules.  Ranking is by signed
score descending — the tokens claimed as evidence *for* the target class —
following the ERASER conventions (DeYoung et al., 2020).

The one honest caveat, stated here rather than hidden: deleting words produces
off-distribution inputs, and every deletion-based metric inherits that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from explainkit.explain import drop_spans
from explainkit.schema import Explanation, PredictFn

DEFAULT_FRACTIONS = (0.05, 0.1, 0.2, 0.33, 0.5)


def _p_target(predict_fn: PredictFn, texts: Sequence[str], target: int) -> list[float]:
    probs = np.asarray(predict_fn(texts), dtype=np.float64)
    return [float(p) for p in probs[:, target]]


def comprehensiveness(
    explanation: Explanation, predict_fn: PredictFn, *, k: int, mask: str = ""
) -> float:
    """Probability drop when the top-``k`` tokens are removed.

    High is good: the model misses the tokens the explanation called
    important.
    """
    variant = drop_spans(explanation.text, explanation.spans, explanation.top_k(k), mask)
    (p,) = _p_target(predict_fn, [variant], explanation.target_class)
    return explanation.prediction - p


def sufficiency(
    explanation: Explanation, predict_fn: PredictFn, *, k: int, mask: str = ""
) -> float:
    """Probability drop when *only* the top-``k`` tokens are kept.

    Low is good: the claimed evidence alone nearly reproduces the prediction.
    """
    keep = set(explanation.top_k(k))
    rest = [i for i in range(len(explanation.tokens)) if i not in keep]
    variant = drop_spans(explanation.text, explanation.spans, rest, mask)
    (p,) = _p_target(predict_fn, [variant], explanation.target_class)
    return explanation.prediction - p


def deletion_curve(
    explanation: Explanation,
    predict_fn: PredictFn,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    mask: str = "",
) -> list[tuple[float, float]]:
    """p(target) after deleting the top fraction of tokens, per fraction.

    All variants go to the model in one batch.
    """
    order = explanation.ranking()
    n = len(order)
    variants = []
    for fraction in fractions:
        k = math.ceil(fraction * n) if n else 0
        variants.append(drop_spans(explanation.text, explanation.spans, order[:k], mask))
    probabilities = _p_target(predict_fn, variants, explanation.target_class)
    return list(zip([float(f) for f in fractions], probabilities, strict=True))


def aopc(
    explanation: Explanation,
    predict_fn: PredictFn,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    mask: str = "",
) -> float:
    """Area over the perturbation curve: mean drop across deletion fractions."""
    curve = deletion_curve(explanation, predict_fn, fractions=fractions, mask=mask)
    return float(np.mean([explanation.prediction - p for _, p in curve]))


def permute_scores(explanation: Explanation, seed: int = 0) -> Explanation:
    """The same explanation with its scores shuffled — the random baseline.

    An informative attribution should beat its own permuted twin on every
    faithfulness metric; if it does not, the ranking carries no signal.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(explanation.scores))
    return explanation.model_copy(
        update={
            "scores": [explanation.scores[int(i)] for i in permutation],
            "method": f"{explanation.method}, permuted",
        }
    )
