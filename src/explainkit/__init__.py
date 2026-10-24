"""Explain the classifier, then test the explanation.

Word-level attributions for any text classifier behind a predict function,
plus the tools that make an attribution earn its keep: corpus-level token
selection, prompt steering, replacement probes, and faithfulness,
plausibility and agreement metrics.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from explainkit.agreement import rank_correlation, topk_jaccard
from explainkit.explain import Explainer, drop_spans
from explainkit.faithfulness import (
    aopc,
    comprehensiveness,
    deletion_curve,
    permute_scores,
    sufficiency,
)
from explainkit.io import dump_explanations, load_explanations
from explainkit.plausibility import rationale_f1, rationale_iou
from explainkit.replace import (
    Replacement,
    RobustnessReport,
    Strategy,
    delete_strategy,
    mask_strategy,
    replace_tokens,
    robustness_probe,
    synonym_strategy,
)
from explainkit.schema import Explanation, MetaValue, PredictFn, Span, Tokenizer
from explainkit.select import TokenImportance, select_tokens, token_importance
from explainkit.steer import highlight, label_distribution, majority_vote
from explainkit.tokenizers import WORD_PATTERN, regex_tokenizer, word_tokenizer

try:
    # One source for the number: the version pyproject.toml gave the install.
    __version__ = _version("explainkit")
except PackageNotFoundError:  # imported from a source tree that isn't installed
    __version__ = "0+unknown"

__all__ = [
    "WORD_PATTERN",
    "Explainer",
    "Explanation",
    "MetaValue",
    "PredictFn",
    "Replacement",
    "RobustnessReport",
    "Span",
    "Strategy",
    "TokenImportance",
    "Tokenizer",
    "__version__",
    "aopc",
    "comprehensiveness",
    "delete_strategy",
    "deletion_curve",
    "drop_spans",
    "dump_explanations",
    "highlight",
    "label_distribution",
    "load_explanations",
    "majority_vote",
    "mask_strategy",
    "permute_scores",
    "rank_correlation",
    "rationale_f1",
    "rationale_iou",
    "regex_tokenizer",
    "replace_tokens",
    "robustness_probe",
    "select_tokens",
    "sufficiency",
    "synonym_strategy",
    "token_importance",
    "topk_jaccard",
    "word_tokenizer",
]
