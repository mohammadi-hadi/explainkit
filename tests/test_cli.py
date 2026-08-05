"""The CLI writes the artifacts, and README injection is exact and idempotent."""

from __future__ import annotations

from pathlib import Path

import pytest

from explainkit.cli import main
from explainkit.demo import inject

README = """# thing

<!-- explainkit:demo -->
stale table
<!-- /explainkit:demo -->

```
<!-- explainkit:demo -->
this pair is documentation and must survive
<!-- /explainkit:demo -->
```
"""


def test_demo_writes_report_summary_and_figures(tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert main(["demo", "--out", str(out)]) == 0
    assert (out / "report.md").is_file()
    assert (out / "summary.md").is_file()
    for figure in ("attribution.png", "selection_curve.png", "deletion_curve.png"):
        assert (out / "figures" / figure).stat().st_size > 0
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("| attribution |")
    assert (out / "report.md").read_text(encoding="utf-8").count(summary.splitlines()[2])


def test_inject_readme_is_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "results"
    main(["demo", "--out", str(out)])
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")

    assert main(["inject-readme", str(readme), "--results", str(out)]) == 0
    once = readme.read_text(encoding="utf-8")
    assert "stale table" not in once
    assert "this pair is documentation and must survive" in once

    assert main(["inject-readme", str(readme), "--results", str(out)]) == 0
    assert readme.read_text(encoding="utf-8") == once


def test_inject_requires_markers() -> None:
    with pytest.raises(ValueError, match="no marker pair"):
        inject("no markers here", "demo", "payload")


def test_inject_ignores_fenced_pairs_only() -> None:
    fenced_only = "```\n<!-- explainkit:demo -->\nx\n<!-- /explainkit:demo -->\n```\n"
    with pytest.raises(ValueError, match="no marker pair"):
        inject(fenced_only, "demo", "payload")
