"""Steer a local LLM annotator with attribution-selected tokens (GenXAI).

Needs a running Ollama server:  ollama pull llama3.2  &&  pip install requests

The four prompting scenarios of the GeBNLP @ ACL 2025 study, reproduced on
the synthetic corpus: plain annotation (GenAI), a persona prompt (GenP),
attribution-guided bolding of the important tokens (GenXAI), and both
combined (GenPXAI).  Six generations per text act as virtual annotators;
majority vote gives the hard label.
"""

from __future__ import annotations

import json

import requests

from explainkit import (
    Explainer,
    highlight,
    majority_vote,
    select_tokens,
    token_importance,
)
from explainkit.dataset import generate, train_test_split
from explainkit.demo import _NaiveBayes

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "llama3.2"
PERSONA = "You are a 34-year-old project manager who values direct but respectful feedback."


def ask(prompt: str, seed: int) -> str:
    response = requests.post(
        OLLAMA,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "seed": seed},
        },
        timeout=120,
    )
    return str(json.loads(response.text)["response"])


def annotate(text: str, *, persona: bool, tokens: list[str] | None, votes: int = 6) -> int:
    body = highlight(text, tokens) if tokens else text
    prompt = (
        (PERSONA + "\n\n" if persona else "")
        + "Label the following workplace comment as hostile or civil. "
        + ("The words in **bold** carried the most weight for a reference "
           "classifier; weigh them accordingly. " if tokens else "")
        + f'Answer with the single word "hostile" or "civil".\n\nComment: {body}'
    )
    ballots = ["hostile" in ask(prompt, seed).lower() for seed in range(votes)]
    (winner,) = majority_vote([[int(b) for b in ballots]])
    return winner


def main() -> None:
    train, test = train_test_split(generate(240, seed=0), n_test=40)
    model = _NaiveBayes().fit(train.texts, train.labels)
    explainer = Explainer(model.predict_proba)

    explanations = explainer.explain_batch(test.texts)
    importances = token_importance(explanations, test.labels)
    vocabulary = [item.token for item in select_tokens(importances, threshold=0.95)]

    scenarios = {
        "GenAI": dict(persona=False, tokens=None),
        "GenP": dict(persona=True, tokens=None),
        "GenXAI": dict(persona=False, tokens=vocabulary),
        "GenPXAI": dict(persona=True, tokens=vocabulary),
    }
    sample = list(range(10))
    for name, options in scenarios.items():
        labels = [
            annotate(test.texts[i], persona=bool(options["persona"]),
                     tokens=options["tokens"])  # type: ignore[arg-type]
            for i in sample
        ]
        accuracy = sum(labels[j] == test.labels[i] for j, i in enumerate(sample)) / len(sample)
        print(f"{name:>8}: accuracy {accuracy:.2f} on {len(sample)} texts")


if __name__ == "__main__":
    main()
