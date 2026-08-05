"""Turn a Hugging Face text-classification pipeline into a predict function.

The wrapper keeps subwords where they belong — inside the model.  Perturbed
texts reach the pipeline as raw strings; attribution happens at this
package's word level.  Requires the ``transformers`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from explainkit._optional import require
from explainkit.schema import PredictFn


def pipeline_predict_fn(model: str, *, batch_size: int = 32, device: Any = None) -> PredictFn:
    """Build a predict function from a model name on the Hub."""
    transformers = require("transformers", extra="transformers")
    pipe = transformers.pipeline(
        "text-classification", model=model, device=device, top_k=None
    )
    return from_pipeline(pipe, batch_size=batch_size)


def from_pipeline(pipe: Any, *, batch_size: int = 32) -> PredictFn:
    """Wrap an existing ``transformers`` text-classification pipeline.

    Class columns follow the model's ``label2id`` order, so ``target`` indices
    passed to :class:`explainkit.Explainer` mean what the model config says
    they mean.
    """
    label_to_id: dict[str, int] = dict(pipe.model.config.label2id)
    n_classes = len(label_to_id)

    def predict(texts: Sequence[str]) -> NDArray[np.float64]:
        raw = pipe(list(texts), top_k=None, batch_size=batch_size, truncation=True)
        out = np.zeros((len(texts), n_classes), dtype=np.float64)
        for row, entries in enumerate(raw):
            for entry in entries:
                out[row, label_to_id[entry["label"]]] = float(entry["score"])
        return out

    return predict
