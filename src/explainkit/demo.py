"""One classifier, three attributions, every claim tested.

The demo trains a small Naive Bayes on the synthetic corpus — chosen because
its fit is closed-form counting, reproducible to machine epsilon on any
platform — then walks the package's whole pipeline: explain each test text
with occlusion and the surrogate, add a permuted-scores baseline, select the
tokens that carry 95% of corpus importance, and probe robustness by swapping
top-attributed hostile words for civil ones.  Because the corpus has
implanted rationales, every number in the table has a right answer to be
checked against.  Seeds are fixed; CI regenerates the report and fails on
drift.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from explainkit.agreement import topk_jaccard
from explainkit.dataset import (
    CIVIL_LEXICON,
    CIVIL_SYNONYMS,
    HOSTILE_LEXICON,
    generate,
    train_test_split,
)
from explainkit.explain import Explainer
from explainkit.faithfulness import (
    aopc,
    comprehensiveness,
    deletion_curve,
    permute_scores,
    sufficiency,
)
from explainkit.plausibility import rationale_f1
from explainkit.replace import replace_tokens, synonym_strategy
from explainkit.schema import Explanation
from explainkit.select import TokenImportance, select_tokens, token_importance
from explainkit.tokenizers import word_tokenizer
from explainkit.viz import deletion_curve_figure, selection_curve_figure, token_bar_figure

K = 3
THRESHOLD = 0.95


class _NaiveBayes:
    """Multinomial Naive Bayes, fit by counting.

    No optimizer, no iteration order, no BLAS — the demo's numbers must be
    identical on every platform CI runs.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._vocabulary: dict[str, int] = {}
        self._log_prior: NDArray[np.float64] = np.zeros(2)
        self._log_theta: NDArray[np.float64] = np.zeros((2, 0))

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> _NaiveBayes:
        tokenized = [
            [span.text.lower() for span in word_tokenizer(text)] for text in texts
        ]
        self._vocabulary = {
            token: i
            for i, token in enumerate(sorted({t for tokens in tokenized for t in tokens}))
        }
        counts = np.zeros((2, len(self._vocabulary)), dtype=np.float64)
        class_counts = np.zeros(2, dtype=np.float64)
        for tokens, label in zip(tokenized, labels, strict=True):
            class_counts[label] += 1
            for token in tokens:
                counts[label, self._vocabulary[token]] += 1
        self._log_prior = np.log(class_counts / class_counts.sum())
        totals = counts.sum(axis=1, keepdims=True)
        self._log_theta = np.log(
            (counts + self.alpha) / (totals + self.alpha * len(self._vocabulary))
        )
        return self

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        out = np.empty((len(texts), 2), dtype=np.float64)
        for row, text in enumerate(texts):
            log_p = self._log_prior.copy()
            for span in word_tokenizer(text):
                index = self._vocabulary.get(span.text.lower())
                if index is not None:
                    log_p = log_p + self._log_theta[:, index]
            log_p -= log_p.max()
            p = np.exp(log_p)
            out[row] = p / p.sum()
        return out


@dataclass(frozen=True)
class MethodRow:
    """One attribution source's corpus-mean scorecard."""

    name: str
    aopc: float
    comprehensiveness: float
    sufficiency: float
    rationale_f1: float
    jaccard_vs_occlusion: float
    flip_rate: float


@dataclass(frozen=True)
class DemoStats:
    rows: list[MethodRow]
    accuracy: float
    vocabulary_size: int
    importances: list[TokenImportance]
    selected: list[TokenImportance]
    selection_precision: float
    signal_present: int
    signal_recovered: int
    curves: list[tuple[str, list[tuple[float, float]]]]
    sample: Explanation
    sample_surrogate: Explanation


def build_stats() -> DemoStats:
    corpus = generate(240, seed=0)
    train, test = train_test_split(corpus, n_test=40)
    model = _NaiveBayes().fit(train.texts, train.labels)
    explainer = Explainer(model.predict_proba)

    occlusion = explainer.explain_batch(test.texts)
    surrogate = explainer.explain_batch(test.texts, method="surrogate", n_samples=256, seed=0)
    random = [permute_scores(e, seed=100 + i) for i, e in enumerate(occlusion)]
    sources: list[tuple[str, list[Explanation]]] = [
        ("occlusion", occlusion),
        ("surrogate", surrogate),
        ("random scores", random),
    ]

    accuracy = float(
        np.mean([e.predicted_class == y for e, y in zip(occlusion, test.labels, strict=True)])
    )
    hostile = [i for i, e in enumerate(occlusion) if e.predicted_class == 1]
    strategy = synonym_strategy(CIVIL_SYNONYMS)

    rows: list[MethodRow] = []
    curves: list[tuple[str, list[tuple[float, float]]]] = []
    for name, explanations in sources:
        flips = []
        for i in hostile:
            replaced = replace_tokens(explanations[i], strategy, top_k=K)
            probs = model.predict_proba([replaced.text])
            flips.append(int(np.argmax(probs[0])) != explanations[i].predicted_class)
        per_text_curves = [
            deletion_curve(e, model.predict_proba) for e in explanations
        ]
        fractions = [f for f, _ in per_text_curves[0]]
        mean_curve = [
            (f, float(np.mean([curve[j][1] for curve in per_text_curves])))
            for j, f in enumerate(fractions)
        ]
        rows.append(
            MethodRow(
                name=name,
                aopc=float(np.mean([aopc(e, model.predict_proba) for e in explanations])),
                comprehensiveness=float(
                    np.mean(
                        [comprehensiveness(e, model.predict_proba, k=K) for e in explanations]
                    )
                ),
                sufficiency=float(
                    np.mean([sufficiency(e, model.predict_proba, k=K) for e in explanations])
                ),
                rationale_f1=float(
                    np.mean(
                        [
                            rationale_f1(e, rationale, k=K)
                            for e, rationale in zip(explanations, test.rationales, strict=True)
                        ]
                    )
                ),
                jaccard_vs_occlusion=float(
                    np.mean(
                        [
                            topk_jaccard(e, occlusion[i], k=K)
                            for i, e in enumerate(explanations)
                        ]
                    )
                ),
                flip_rate=float(np.mean(flips)),
            )
        )
        curves.append((name, mean_curve))

    importances = token_importance(occlusion, test.labels)
    selected = select_tokens(importances, threshold=THRESHOLD)
    signal = set(HOSTILE_LEXICON) | set(CIVIL_LEXICON)
    precision = float(np.mean([item.token in signal for item in selected]))
    signal_present = len(signal & {item.token for item in importances})
    signal_recovered = len(signal & {item.token for item in selected})

    # A one-signal text makes the sharpest showcase: the whole verdict hangs
    # on a single word, and occlusion shows exactly that.
    sample_index = next(
        i
        for i in range(len(occlusion))
        if test.labels[i] == 1 and len(test.rationales[i]) == 1
    )
    return DemoStats(
        rows=rows,
        accuracy=accuracy,
        vocabulary_size=len(importances),
        importances=importances,
        selected=selected,
        selection_precision=precision,
        signal_present=signal_present,
        signal_recovered=signal_recovered,
        curves=curves,
        sample=occlusion[sample_index],
        sample_surrogate=surrogate[sample_index],
    )


def summary_table(stats: DemoStats) -> str:
    lines = [
        "| attribution | AOPC | comp@3 | suff@3 | rationale F1@3 "
        "| top-3 jaccard vs occlusion | flip rate @3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in stats.rows:
        lines.append(
            f"| {row.name} | {row.aopc:.2f} | {row.comprehensiveness:.2f} "
            f"| {row.sufficiency:.2f} | {row.rationale_f1:.2f} "
            f"| {row.jaccard_vs_occlusion:.2f} | {row.flip_rate:.2f} |"
        )
    return "\n".join(lines)


def _selection_table(stats: DemoStats, limit: int = 15) -> str:
    lines = [
        "| rank | token | SI | IR | cumulative | n |",
        "|---|---|---|---|---|---|",
    ]
    cumulative = 0.0
    for rank, item in enumerate(stats.selected[:limit], start=1):
        cumulative += item.ir
        lines.append(
            f"| {rank} | {item.token} | {item.si:.4f} | {item.ir:.3f} "
            f"| {cumulative:.3f} | {item.n} |"
        )
    if len(stats.selected) > limit:
        lines.append(f"| … | {len(stats.selected) - limit} more selected tokens | | | | |")
    return "\n".join(lines)


def _sample_table(stats: DemoStats) -> str:
    lines = [
        "| token | occlusion | surrogate |",
        "|---|---|---|",
    ]
    for token, occ, sur in zip(
        stats.sample.tokens, stats.sample.scores, stats.sample_surrogate.scores, strict=True
    ):
        lines.append(f"| {token} | {occ:+.3f} | {sur:+.3f} |")
    return "\n".join(lines)


def render_report(stats: DemoStats) -> str:
    hostile_share = stats.rows[0].flip_rate
    random_share = stats.rows[-1].flip_rate
    lines = [
        "# explainkit demo report",
        "",
        "A Naive Bayes classifier is trained on 200 synthetic workplace comments "
        "(hostile vs civil) whose signal words — and only those — carry the label, "
        f"and reaches {stats.accuracy:.2f} accuracy on the 40 held-out texts.  "
        "Each held-out text is explained three ways, and every attribution is made "
        "to prove itself: deletion metrics for faithfulness, the implanted "
        "rationales for plausibility, and a targeted word swap for consequence.",
        "",
        "## Scorecard",
        "",
        summary_table(stats),
        "",
        "AOPC and comp@3 are probability drops after deleting top-ranked tokens "
        "(higher = more faithful); suff@3 is the drop when *only* the top 3 remain "
        "(lower = the evidence suffices); rationale F1@3 compares the top 3 against "
        "the implanted signal words; the flip rate swaps each hostile-predicted "
        "text's top 3 tokens for civil counterparts and counts changed verdicts.  "
        f"The row to sit with is the last one: random and informed attributions "
        f"describe the same classifier, but replacing three informed tokens flips "
        f"{hostile_share:.0%} of hostile verdicts while three random ones flip "
        f"{random_share:.0%} — an attribution you cannot act on was never evidence.",
        "",
        f"## Token selection at T_c = {THRESHOLD:g}",
        "",
        f"Of {stats.vocabulary_size} distinct test-corpus tokens, "
        f"{len(stats.selected)} carry 95% of the corpus importance "
        f"(correctness-gated mean |score|, occlusion).  The selection recovers "
        f"{stats.signal_recovered} of the {stats.signal_present} implanted signal "
        f"words that occur in the test slice; the remainder is the thin tail that "
        f"the last few percent of importance drags in.",
        "",
        _selection_table(stats),
        "",
        "## One text, explained",
        "",
        f"> {stats.sample.text}",
        "",
        f"True label hostile; p(hostile) = {stats.sample.prediction:.3f}.",
        "",
        _sample_table(stats),
        "",
        "![attribution](figures/attribution.png)",
        "",
        "![selection curve](figures/selection_curve.png)",
        "",
        "![deletion curve](figures/deletion_curve.png)",
    ]
    return "\n".join(lines) + "\n"


def write_demo(out_dir: Path) -> DemoStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    stats = build_stats()
    (out_dir / "report.md").write_text(render_report(stats), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_table(stats) + "\n", encoding="utf-8")
    token_bar_figure(stats.sample, figures / "attribution.png")
    selection_curve_figure(stats.importances, figures / "selection_curve.png", threshold=THRESHOLD)
    deletion_curve_figure(stats.curves, figures / "deletion_curve.png")
    return stats


def inject(text: str, key: str, payload: str) -> str:
    """Replace the block between ``<!-- explainkit:key -->`` markers.

    Marker pairs inside fenced code blocks are left alone, so documentation
    showing the marker syntax does not get real tables injected into it.
    """
    start = f"<!-- explainkit:{key} -->"
    end = f"<!-- /explainkit:{key} -->"
    fenced = [m.span() for m in re.finditer(r"^```.*?^```", text, re.MULTILINE | re.DOTALL)]
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    for match in pattern.finditer(text):
        if not any(a <= match.start() < b for a, b in fenced):
            replacement = f"{start}\n{payload.rstrip()}\n{end}"
            return text[: match.start()] + replacement + text[match.end() :]
    raise ValueError(f"no marker pair for key {key!r} outside code fences")


def inject_readme(readme_path: Path, results_dir: Path) -> None:
    summary = (results_dir / "summary.md").read_text(encoding="utf-8")
    text = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(inject(text, "demo", summary), encoding="utf-8")
