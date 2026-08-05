"""Do the top tokens match what a human would point at?

Plausibility compares the explanation's top-``k`` tokens against a reference
rationale — token indices, or token strings matched case-insensitively.
Plausibility is not faithfulness: an explanation can flatter human intuition
while the model attends elsewhere, which is exactly why both families exist.
"""

from __future__ import annotations

from collections.abc import Collection

from explainkit.schema import Explanation


def _rationale_indices(explanation: Explanation, rationale: Collection[int | str]) -> set[int]:
    indices: set[int] = set()
    wanted: set[str] = set()
    for item in rationale:
        if isinstance(item, str):
            wanted.add(item.lower())
        else:
            if not 0 <= item < len(explanation.tokens):
                raise ValueError(
                    f"rationale index {item} outside {len(explanation.tokens)} tokens"
                )
            indices.add(item)
    if wanted:
        indices |= {i for i, t in enumerate(explanation.tokens) if t.lower() in wanted}
    return indices


def rationale_f1(
    explanation: Explanation, rationale: Collection[int | str], *, k: int
) -> float:
    """Token-level F1 between the top-``k`` tokens and the reference rationale."""
    predicted = set(explanation.top_k(k))
    gold = _rationale_indices(explanation, rationale)
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    overlap = len(predicted & gold)
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rationale_iou(
    explanation: Explanation, rationale: Collection[int | str], *, k: int
) -> float:
    """Intersection over union between the top-``k`` tokens and the rationale."""
    predicted = set(explanation.top_k(k))
    gold = _rationale_indices(explanation, rationale)
    union = predicted | gold
    if not union:
        return 1.0
    return len(predicted & gold) / len(union)
