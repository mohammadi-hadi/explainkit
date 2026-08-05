"""Read and write explanations as JSON Lines.

Attribution runs against a real model can be expensive; storing explanations
lets selection, metrics and reports re-run without the model in the room.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from explainkit.schema import Explanation


def load_explanations(path: str | Path) -> list[Explanation]:
    """Load one explanation per line, failing with the line number on bad input."""
    records: list[Explanation] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(Explanation.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{path}, line {lineno}: {exc}") from exc
    return records


def dump_explanations(explanations: Iterable[Explanation], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for explanation in explanations:
            fh.write(explanation.model_dump_json(exclude_none=True) + "\n")
