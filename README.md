<div align="center">

# explainkit

[![ci](https://github.com/mohammadi-hadi/explainkit/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadi-hadi/explainkit/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21803833.svg)](https://doi.org/10.5281/zenodo.21803833)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mohammadi-hadi/explainkit/blob/main/LICENSE)

*Explain the classifier, then test the explanation.*

</div>

A token heatmap is a claim about a model, and most heatmaps are never tested.
explainkit computes word-level attributions for any text classifier hiding
behind a `predict_proba`-style callable, then makes those attributions earn
their keep: it selects the tokens that carry 95% of the corpus importance,
steers an LLM annotator toward them, swaps them out to see whether the verdict
survives, and scores the whole chain for faithfulness, plausibility and
cross-method agreement.  Explain, select, steer, replace, verify: the
pipeline behind a set of published studies on transparent text classification
(citations at the bottom), packaged.

## The demo in one table

Three attributions of the same classifier (occlusion, a LIME-style
surrogate and a permuted-scores baseline), each made to prove itself on a
corpus whose ground-truth rationales are known by construction:

<!-- explainkit:demo -->
| attribution | AOPC | comp@3 | suff@3 | rationale F1@3 | top-3 jaccard vs occlusion | flip rate @3 |
|---|---|---|---|---|---|---|
| occlusion | 0.40 | 0.49 | -0.02 | 0.77 | 1.00 | 1.00 |
| surrogate | 0.39 | 0.49 | -0.02 | 0.77 | 0.90 | 1.00 |
| random scores | 0.04 | 0.03 | 0.24 | 0.20 | 0.20 | 0.20 |
<!-- /explainkit:demo -->

AOPC and comp@3 are the probability drops after deleting top-ranked words
(higher = more faithful).  suff@3 is the drop when *only* the top 3 remain
(near zero = the evidence suffices).  Rationale F1@3 compares top tokens
against the implanted signal words.  The flip rate replaces each
hostile-predicted text's top 3 words with civil counterparts and counts
changed verdicts.  The row to sit with is the last one: random and informed
scores describe the *same* classifier, but acting on the informed ones flips
essentially every verdict while acting on random ones barely moves it.  An
attribution you cannot act on was never evidence.

`make demo` regenerates the [full report](https://github.com/mohammadi-hadi/explainkit/blob/main/results/report.md) and the figures
from fixed seeds.  CI rebuilds them from pinned dependencies and fails if a
committed number differs from what the code produces.

![attribution](https://raw.githubusercontent.com/mohammadi-hadi/explainkit/main/results/figures/attribution.png)

![deletion curve](https://raw.githubusercontent.com/mohammadi-hadi/explainkit/main/results/figures/deletion_curve.png)

## Install

```
pip install git+https://github.com/mohammadi-hadi/explainkit
```

Python 3.11+.  Runtime dependencies: numpy, pydantic, matplotlib.  The core
methods (occlusion, surrogate) need nothing else.  Heavier backends are
extras:

| extra | adds | enables |
|---|---|---|
| `[shap]` | shap | `method="shap"` |
| `[lime]` | lime | `method="lime"` |
| `[transformers]` | transformers, torch | `explainkit.hf` pipeline wrapper |
| `[all]` | all of the above | everything |

## Quickstart

```python
from explainkit import Explainer, token_importance, select_tokens, highlight
from explainkit.replace import synonym_strategy, robustness_probe

explainer = Explainer(model.predict_proba)          # any (texts) -> (n, classes)

e = explainer.explain("your draft was useless and the meeting felt hopeless")
print(e.tokens, e.scores)                           # word-level attribution

importances = token_importance(explanations, y_true)   # corpus-level SI / IR
vocabulary = select_tokens(importances, threshold=0.95)

prompt = highlight(text, [t.token for t in vocabulary])  # **bold** the evidence

report = robustness_probe(explainer, texts,
                          synonym_strategy({"useless": "helpful"}), top_k=3)
print(report.flip_rate)                             # do the attributions act?
```

With a Hugging Face model:

```python
from explainkit.hf import pipeline_predict_fn      # pip install "explainkit[transformers]"

explainer = Explainer(pipeline_predict_fn("distilbert-base-uncased-finetuned-sst-2-english"))
e = explainer.explain("a quietly devastating film", method="shap")
```

## What it does

| module | the question it answers | needs |
|---|---|---|
| `explain` | which words pushed this prediction? (occlusion, surrogate, shap, lime) | core |
| `select` | which tokens carry the corpus? correctness-gated SI, IR, cumulative cut at T_c | core |
| `steer` | how do I point an LLM annotator at them? highlighting, virtual-annotator voting | core |
| `replace` | do the attributions transfer to action? strategies + robustness probes | core |
| `faithfulness` | does the model actually depend on them? comprehensiveness, sufficiency, AOPC | core |
| `plausibility` | do they match human rationales? token F1, IOU | core |
| `agreement` | do two methods tell one story? rank correlation, top-k overlap | core |
| `viz` | can I look at it? HTML heatmaps, attribution and selection figures | core |

## How the numbers are defended

- Occlusion must recover the coefficient signs of a known linear model, by
  test.  The surrogate must recover its ranking.
- Spearman and Kendall are computed in plain numpy and cross-checked against
  scipy to 1e-10 in the test suite.  The library depends on scipy for neither.
- The demo corpus implants its rationales, so selection and plausibility are
  checked against planted ground truth, not eyeballed.
- Every README number is regenerated by CI from pinned dependencies and
  diffed against the committed report.

## Design notes

- **Words, not subwords.**  Attribution, selection and replacement happen at
  the word level, the level humans read.  A model's subword vocabulary stays
  its own business: perturbed texts reach the model as raw strings.
  Subword-to-word pooling for transformer-native scores is on the roadmap.
- **Spans, not string replacement.**  Every perturbation splices validated
  character spans (the `Explanation` model rejects spans that do not read
  back as their token), so editing "art" can never corrupt "party".
- **Ranks across methods, never magnitudes.**  SHAP values are additive
  contributions, while occlusion scores are probability drops.  The agreement
  module compares rankings and top-k sets only, and the selection module
  normalizes within a corpus.
- **Deterministic to the digit.**  Every sampled method takes a seed,
  and the demo's model is closed-form counting, so its numbers cannot wobble
  across platforms.

## Limitations

- Occlusion costs n+2 model calls per text (batched into one) and reads
  near-independent token effects.  Strongly interacting features deserve the
  sampling methods.
- Deletion-based faithfulness metrics evaluate the model off-distribution.
  That caveat is known across the field, and stated here rather than hidden.
- The demo corpus is synthetic by design (real moderation data cannot ship
  with ground-truth rationales), so treat its numbers as a validation harness,
  not a benchmark.
- Word-level only.  Occlusion, the surrogate and the shap backend all build
  their variants by splicing the same word spans, so they see the same texts.
  The lime backend perturbs a whitespace-joined rendering of the tokens, so
  its variants are not byte-identical to the others.

## Roadmap

Subword-to-word attribution pooling, a POS-matched replacement strategy in
core, bootstrap confidence intervals on corpus metrics, an
integrated-gradients backend, and a `report` CLI over stored explanation JSONL.

## Related work

[SHAP](https://github.com/shap/shap), [LIME](https://github.com/marcotcr/lime)
and the [ERASER](https://www.eraserbenchmark.com/) faithfulness metrics are
the foundations this package operationalizes end-to-end.  Sister package:
[judgekit](https://github.com/mohammadi-hadi/judgekit), bias probes for LLM
judges, each with a bootstrap confidence interval.

## Citing

This package operationalizes methods developed across:

```bibtex
@article{mohammadi2024explainable,
  title   = {A Transparent Pipeline for Identifying Sexism in Social Media: Combining Explainability with Model Prediction},
  author  = {Mohammadi, Hadi and Giachanou, Anastasia and Bagheri, Robert A.},
  journal = {Applied Sciences},
  volume  = {14},
  number  = {19},
  pages   = {8620},
  year    = {2024},
  doi     = {10.3390/app14198620}
}

@inproceedings{mohammadi2025reliability,
  title     = {Assessing the Reliability of LLMs Annotations in the Context of Demographic Bias and Model Explanation},
  author    = {Mohammadi, Hadi and Shahedi, Tina and Mosteiro, Pablo and Poesio, Massimo and Bagheri, Robert A. and Giachanou, Anastasia},
  booktitle = {Proceedings of the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP)},
  year      = {2025},
  doi       = {10.18653/v1/2025.gebnlp-1.9}
}

@article{mohammadi2025token,
  title         = {Explainability-Based Token Replacement on LLM-Generated Text},
  author        = {Mohammadi, Hadi and Giachanou, Anastasia and Oberski, Daniel L. and Bagheri, Robert A.},
  year          = {2025},
  journal       = {arXiv preprint},
  eprint        = {2506.04050},
  archivePrefix = {arXiv}
}

@article{mohammadi2025xnlp,
  title         = {Explainability in Practice: A Survey of Explainable NLP Across Various Domains},
  author        = {Mohammadi, Hadi and Bagheri, Robert A. and Giachanou, Anastasia and Oberski, Daniel L.},
  year          = {2025},
  journal       = {arXiv preprint},
  eprint        = {2502.00837},
  archivePrefix = {arXiv}
}
```

## License

MIT.
