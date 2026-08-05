"""Render explanations for humans: inline HTML and matplotlib figures.

Figures are for display and are not byte-stable across matplotlib versions,
so they get committed but never diffed — the deterministic artifacts are the
markdown tables (see ``explainkit.demo``).
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from explainkit.schema import Explanation
from explainkit.select import TokenImportance

ACCENT = "#2563eb"
NEGATIVE = "#dc2626"
MUTED = "#9ca3af"


def heatmap_html(explanation: Explanation) -> str:
    """The text with each token tinted by its score — blue for, red against.

    Returns an HTML fragment; drop it in a notebook cell or a report.
    """
    peak = max((abs(s) for s in explanation.scores), default=0.0)
    parts: list[str] = []
    cursor = 0
    for (start, end), score in zip(explanation.spans, explanation.scores, strict=True):
        parts.append(html.escape(explanation.text[cursor:start]))
        alpha = 0.0 if peak == 0 else 0.75 * abs(score) / peak
        channel = "37,99,235" if score >= 0 else "220,38,38"
        parts.append(
            f'<mark style="background: rgba({channel},{alpha:.2f}); padding: 0 1px;"'
            f' title="{score:+.4f}">{html.escape(explanation.text[start:end])}</mark>'
        )
        cursor = end
    parts.append(html.escape(explanation.text[cursor:]))
    return "".join(parts)


def token_bar_figure(explanation: Explanation, path: Path, *, top: int = 12) -> bool:
    """Horizontal bars for the strongest tokens of one explanation."""
    if not explanation.tokens:
        return False
    indices = sorted(
        range(len(explanation.scores)),
        key=lambda i: (-abs(explanation.scores[i]), i),
    )[:top]
    indices.reverse()
    labels = [explanation.tokens[i] for i in indices]
    values = [explanation.scores[i] for i in indices]
    colors = [ACCENT if v >= 0 else NEGATIVE for v in values]
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.barh(range(len(values)), values, color=colors)
    axis.set_yticks(range(len(values)), labels)
    axis.axvline(0, color=MUTED, linewidth=1)
    axis.set_xlabel(f"{explanation.method} score toward class {explanation.target_class}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def selection_curve_figure(
    importances: Sequence[TokenImportance],
    path: Path,
    *,
    threshold: float = 0.95,
) -> bool:
    """Cumulative importance ratio against token rank, with the T_c cut."""
    if not importances:
        return False
    ordered = sorted(importances, key=lambda item: (-item.ir, item.token))
    cumulative = []
    total = 0.0
    for item in ordered:
        total += item.ir
        cumulative.append(total)
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(range(1, len(cumulative) + 1), cumulative, color=ACCENT)
    axis.axhline(threshold, color=MUTED, linewidth=1, linestyle="--", label=f"T_c = {threshold:g}")
    axis.set_xlabel("tokens, by importance ratio")
    axis.set_ylabel("cumulative importance")
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def deletion_curve_figure(
    curves: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    path: Path,
) -> bool:
    """Mean p(target) as top-ranked tokens are deleted, one line per method."""
    if not curves:
        return False
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    palette = [ACCENT, "#059669", NEGATIVE, "#d97706"]
    for (label, curve), color in zip(curves, palette, strict=False):
        axis.plot(
            [f for f, _ in curve], [p for _, p in curve], "o-", color=color, label=label
        )
    axis.set_xlabel("fraction of tokens deleted, best first")
    axis.set_ylabel("mean p(target)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True
