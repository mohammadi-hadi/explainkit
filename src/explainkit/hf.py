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


def pipeline_predict_fn(
    model: str, *, batch_size: int = 32, device: Any = None
) -> PipelinePredict:
    """Build a predict function from a model name on the Hub."""
    transformers = require("transformers", extra="transformers")
    pipe = transformers.pipeline(
        "text-classification", model=model, device=device, top_k=None
    )
    return from_pipeline(pipe, batch_size=batch_size)


def from_pipeline(pipe: Any, *, batch_size: int = 32) -> PipelinePredict:
    """Wrap an existing ``transformers`` text-classification pipeline.

    Class columns follow the model's ``label2id`` order, so ``target`` indices
    passed to :class:`explainkit.Explainer` mean what the model config says
    they mean.
    """
    return PipelinePredict(pipe, batch_size=batch_size)


class PipelinePredict:
    """A predict function over a pipeline that also knows its mask token.

    ``mask_token`` is the tokenizer's own (``[MASK]`` for BERT models), or
    None when it has none.  ``Explainer(predict, mask=predict.mask_token)``
    replaces dropped words with it instead of deleting them.
    """

    def __init__(self, pipe: Any, *, batch_size: int = 32) -> None:
        self.pipe = pipe
        self.batch_size = batch_size
        self.label_to_id: dict[str, int] = dict(pipe.model.config.label2id)
        self.mask_token: str | None = getattr(pipe.tokenizer, "mask_token", None)

    def __call__(self, texts: Sequence[str]) -> NDArray[np.float64]:
        raw = self.pipe(list(texts), top_k=None, batch_size=self.batch_size, truncation=True)
        out = np.zeros((len(texts), len(self.label_to_id)), dtype=np.float64)
        for row, entries in enumerate(raw):
            for entry in entries:
                out[row, self.label_to_id[entry["label"]]] = float(entry["score"])
        return out
