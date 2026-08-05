"""Word-level attribution for any text classifier behind a predict function.

Two methods ship with the base install and need nothing beyond numpy:

- ``occlusion``: drop each word once and read the probability change.  One
  batched model call of n+2 rows per text.  Exact for near-linear models,
  honest everywhere else.
- ``surrogate``: a LIME-style local linear model fit on random keep/drop masks
  of the words, solved in closed form.  Seeded, so the same call returns the
  same explanation to the last digit.

``shap`` and ``lime`` dispatch to the optional backends, which are fed this
module's tokenization so scores always align with the same word spans.

The input is tokenized exactly once.  Every perturbed variant is built by
splicing the original string around the frozen spans — never by re-tokenizing,
never by ``str.replace`` — and handed to the model as a raw string.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
from numpy.typing import NDArray

from explainkit.schema import Explanation, PredictFn, Span, Tokenizer
from explainkit.tokenizers import word_tokenizer

_METHODS = ("occlusion", "surrogate", "shap", "lime")


def drop_spans(
    text: str,
    spans: Sequence[tuple[int, int]],
    indices: Collection[int],
    mask: str = "",
) -> str:
    """Replace the given token spans with ``mask``, splicing right to left.

    Right-to-left keeps earlier character offsets valid while later spans are
    being rewritten.  With the default empty mask this is word deletion.
    """
    out = text
    for i in sorted(indices, reverse=True):
        start, end = spans[i]
        out = out[:start] + mask + out[end:]
    return out


class Explainer:
    """Attribution methods over one predict function and one tokenizer.

    ``predict_fn`` takes a batch of texts and returns an array of class
    probabilities, shape ``(len(texts), n_classes)``.  ``mask`` is what a
    dropped word is replaced with when building perturbed variants; the empty
    default deletes the word outright.
    """

    def __init__(
        self,
        predict_fn: PredictFn,
        tokenizer: Tokenizer = word_tokenizer,
        *,
        mask: str = "",
    ) -> None:
        self.predict_fn = predict_fn
        self.tokenizer = tokenizer
        self.mask = mask

    def explain(
        self,
        text: str,
        *,
        method: str = "occlusion",
        target: int | None = None,
        n_samples: int = 256,
        seed: int = 0,
    ) -> Explanation:
        """Explain one prediction.

        ``target`` picks the class the scores argue for; the default is
        whatever class the model predicts for the untouched text.
        ``n_samples`` is the perturbation budget for the sampled methods
        (surrogate, shap, lime); occlusion ignores it.
        """
        if method not in _METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {_METHODS}")
        spans = self.tokenizer(text)

        if method in ("shap", "lime"):
            probs = self._predict([text])
            predicted, chosen, prediction = self._resolve_target(probs[0], target)
            scores: list[float]
            if not spans:
                scores, base_value = [], prediction
            elif method == "shap":
                from explainkit.shap_backend import shap_scores

                scores, base_value = shap_scores(
                    self.predict_fn, self.tokenizer, self.mask, text, spans, chosen,
                    n_samples, seed,
                )
            else:
                from explainkit.lime_backend import lime_scores

                scores, base_value = lime_scores(
                    self.predict_fn, text, spans, chosen, n_samples, seed
                )
            return Explanation(
                text=text,
                tokens=[s.text for s in spans],
                spans=[(s.start, s.end) for s in spans],
                scores=scores,
                base_value=base_value,
                prediction=prediction,
                predicted_class=predicted,
                target_class=chosen,
                method=method,
            )
        if method == "surrogate":
            return self._surrogate(text, spans, target, n_samples, seed)
        return self._occlusion(text, spans, target)

    def explain_batch(
        self,
        texts: Sequence[str],
        *,
        method: str = "occlusion",
        target: int | None = None,
        n_samples: int = 256,
        seed: int = 0,
    ) -> list[Explanation]:
        return [
            self.explain(text, method=method, target=target, n_samples=n_samples, seed=seed)
            for text in texts
        ]

    # ------------------------------------------------------------- methods ---

    def _predict(self, texts: Sequence[str]) -> NDArray[np.float64]:
        probs = np.asarray(self.predict_fn(texts), dtype=np.float64)
        if probs.ndim != 2 or probs.shape[0] != len(texts):
            raise ValueError(
                f"predict_fn returned shape {probs.shape} for {len(texts)} texts; "
                "expected (len(texts), n_classes)"
            )
        return probs

    def _resolve_target(
        self, probs_full: NDArray[np.float64], target: int | None
    ) -> tuple[int, int, float]:
        predicted = int(np.argmax(probs_full))
        chosen = predicted if target is None else target
        if not 0 <= chosen < probs_full.shape[0]:
            raise ValueError(f"target class {chosen} outside {probs_full.shape[0]} classes")
        return predicted, chosen, float(probs_full[chosen])

    def _occlusion(self, text: str, spans: list[Span], target: int | None) -> Explanation:
        offsets = [(s.start, s.end) for s in spans]
        variants = [text]
        variants += [drop_spans(text, offsets, [i], self.mask) for i in range(len(spans))]
        variants.append(drop_spans(text, offsets, range(len(spans)), self.mask))
        probs = self._predict(variants)
        predicted, chosen, prediction = self._resolve_target(probs[0], target)
        scores = [prediction - float(probs[1 + i, chosen]) for i in range(len(spans))]
        return Explanation(
            text=text,
            tokens=[s.text for s in spans],
            spans=offsets,
            scores=scores,
            base_value=float(probs[-1, chosen]),
            prediction=prediction,
            predicted_class=predicted,
            target_class=chosen,
            method="occlusion",
        )

    def _surrogate(
        self, text: str, spans: list[Span], target: int | None, n_samples: int, seed: int
    ) -> Explanation:
        if n_samples < 2:
            raise ValueError(f"surrogate needs n_samples >= 2, got {n_samples}")
        offsets = [(s.start, s.end) for s in spans]
        n = len(spans)
        rng = np.random.default_rng(seed)
        keep = rng.random((n_samples, n)) < 0.5
        if n:
            keep[0, :] = True  # anchor the fit at the untouched text
        variants = [
            drop_spans(text, offsets, [int(i) for i in np.flatnonzero(~row)], self.mask)
            for row in keep
        ]
        probs = self._predict([text, *variants])
        predicted, chosen, prediction = self._resolve_target(probs[0], target)
        y = probs[1:, chosen]

        # Weighted ridge regression of p(target) on the keep mask, in closed
        # form: variants closer to the full text weigh more (exponential
        # kernel on the dropped fraction), and a small L2 term keeps the
        # system solvable when masks collide.
        design = np.hstack([np.ones((n_samples, 1)), keep.astype(np.float64)])
        dropped_fraction = 1.0 - (keep.mean(axis=1) if n else np.ones(n_samples))
        weights = np.exp(-dropped_fraction / 0.25)
        lhs = design.T @ (design * weights[:, None]) + 1e-6 * np.eye(n + 1)
        rhs = design.T @ (weights * y)
        beta = np.linalg.solve(lhs, rhs)
        return Explanation(
            text=text,
            tokens=[s.text for s in spans],
            spans=offsets,
            scores=[float(b) for b in beta[1:]],
            base_value=float(beta[0]),
            prediction=prediction,
            predicted_class=predicted,
            target_class=chosen,
            method="surrogate",
            meta={"n_samples": n_samples, "seed": seed},
        )

