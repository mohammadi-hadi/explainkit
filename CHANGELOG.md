# Changelog

## 0.2.0 (2026-10-24)

The first release on PyPI. Most of these fixes came out of running the shap
backend on a model explainkit did not build: DistilBERT fine-tuned on SST-2.

- **shap scores change.** The shap backend now builds its masked texts by
  splicing the word spans, as occlusion and the surrogate do, and like them it
  deletes a dropped word by default. Before, shap's Text masker wrote the word
  `MASK` with no space after it, so the stand-in stuck to the next word: "a
  good day" with "good" dropped reached the model as "a MASKday", which also
  destroyed "day". It also lost whatever came before the first word and after
  the last, so even the unmasked text was missing its closing punctuation.
  The `mask` given to `Explainer` now applies to shap too.
- The Hugging Face wrapper in `explainkit.hf` carries the tokenizer's
  `mask_token`, so `Explainer(predict, mask=predict.mask_token)` replaces
  dropped words with `[MASK]` instead of deleting them.
- A one-word text gets its exact Shapley value with `method="shap"`. It used to
  fail with "zero-size array to reduction operation maximum which has no
  identity".
- On Python 3.12 and later the `shap` extra needs shap 0.53, the first version
  whose text explanations follow `seed`. Older versions shuffled the words with
  a random generator that `np.random.seed` does not reach, so repeated runs of
  one sentence with seed 0 gave different top words. On Python 3.11, where shap
  stops at 0.51, explainkit warns about this instead.
- `explainkit.__version__` comes from the installed package, so it matches the
  release. The v0.1.1 tag still reported 0.1.0.
- A release workflow publishes to PyPI from a version tag, after checking that
  the tag matches the package version and running the tests.

## 0.1.1 (2026-08-05)

- The version in `CITATION.cff`. Nothing else changed, and the package itself
  still reported 0.1.0.

## 0.1.0 (2026-08-05)

First release, installed from GitHub.

- `Explainer` with occlusion and a LIME-style surrogate in the base install,
  and shap and lime backends behind extras, all scoring the same word spans.
- A wrapper for Hugging Face text-classification pipelines.
- Faithfulness, plausibility and cross-method agreement metrics.
- Token selection, prompt steering and replacement probes.
- A demo on a synthetic corpus with implanted rationales, and pinned results
  that CI rebuilds.
