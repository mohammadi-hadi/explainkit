"""Explain a Hugging Face sentiment model and compare two attribution methods.

Needs the extras:  pip install "explainkit[transformers,shap]"

Wraps an off-the-shelf classifier, explains the same sentence with occlusion
and with SHAP, and prints the rank agreement between the two — the quick
sanity check worth running before trusting either.
"""

from __future__ import annotations

from explainkit import Explainer, rank_correlation, topk_jaccard
from explainkit.hf import pipeline_predict_fn

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
TEXT = "the plot is thin and the pacing sluggish, but the closing scene is quietly devastating"


def main() -> None:
    explainer = Explainer(pipeline_predict_fn(MODEL))

    occlusion = explainer.explain(TEXT)
    shap = explainer.explain(TEXT, method="shap", n_samples=512, seed=0)

    print(f"prediction: class {occlusion.predicted_class} at p={occlusion.prediction:.3f}\n")
    print(f"{'token':>14}  {'occlusion':>10}  {'shap':>10}")
    for token, occ, sh in zip(occlusion.tokens, occlusion.scores, shap.scores, strict=True):
        print(f"{token:>14}  {occ:>+10.4f}  {sh:>+10.4f}")

    rho = rank_correlation(occlusion, shap)
    jaccard = topk_jaccard(occlusion, shap, k=5)
    print(f"\nspearman rho = {rho:.3f}, top-5 jaccard = {jaccard:.3f}")
    print("disagreement here is a finding, not a nuisance: two views of one model.")


if __name__ == "__main__":
    main()
