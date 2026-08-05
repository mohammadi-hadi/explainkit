"""Point a generative annotator at the tokens that matter.

The GeBNLP @ ACL 2025 paper steers LLM annotators by bold-wrapping the
SHAP-selected tokens inside the prompt (the GenXAI condition), then simulates
an annotator pool by sampling several generations per item and aggregating —
majority vote for hard labels, the vote distribution for soft ones.  This
module is those two moves, model-agnostic: you bring the LLM call.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TypeVar

from explainkit.schema import Tokenizer
from explainkit.tokenizers import word_tokenizer

L = TypeVar("L")


def highlight(
    text: str,
    tokens: Collection[str],
    *,
    marker: str = "**",
    tokenizer: Tokenizer = word_tokenizer,
    lowercase: bool = True,
) -> str:
    """Wrap every occurrence of the given tokens in ``marker`` pairs.

    Matching is by whole token, never by substring — "art" does not light up
    inside "party" — because candidates come from the same tokenizer that
    produced the explanation.
    """
    wanted = {t.lower() for t in tokens} if lowercase else set(tokens)
    out = text
    for span in reversed(tokenizer(text)):
        token = span.text.lower() if lowercase else span.text
        if token in wanted:
            out = out[: span.start] + marker + span.text + marker + out[span.end :]
    return out


def majority_vote(votes: Sequence[Sequence[L]]) -> list[L]:
    """Hard label per item from several votes; ties go to the first-cast label."""
    winners: list[L] = []
    for item_votes in votes:
        if not item_votes:
            raise ValueError("majority_vote got an item with no votes")
        counts: dict[L, int] = {}
        for vote in item_votes:
            counts[vote] = counts.get(vote, 0) + 1
        best = max(counts.values())
        for vote in item_votes:
            if counts[vote] == best:
                winners.append(vote)
                break
    return winners


def label_distribution(votes: Sequence[Sequence[L]]) -> list[dict[L, float]]:
    """Soft label per item: each label's share of the votes."""
    distributions: list[dict[L, float]] = []
    for item_votes in votes:
        if not item_votes:
            raise ValueError("label_distribution got an item with no votes")
        counts: dict[L, int] = {}
        for vote in item_votes:
            counts[vote] = counts.get(vote, 0) + 1
        distributions.append({label: n / len(item_votes) for label, n in counts.items()})
    return distributions
