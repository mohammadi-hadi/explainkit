"""The replacement strategies of arXiv:2506.04050 as explainkit Strategy callables.

HSR (Human Similar Replacement) swaps a classifier's most influential tokens
for their nearest neighbours in a Word2Vec space trained on human-written
text; PSR adds a part-of-speech gate; the Datamuse strategy asks a public
synonym API instead.  Each is a dozen lines against the Strategy protocol —
the point of this file is the pattern, not the specific resources.

Needs:  pip install gensim nltk requests
        python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
"""

from __future__ import annotations

from collections.abc import Sequence

import nltk
import requests
from gensim.models import Word2Vec

from explainkit import Span, Strategy


def hsr_strategy(human_texts: Sequence[str]) -> Strategy:
    """Replace tokens with their nearest neighbour in human-written text."""
    sentences = [nltk.word_tokenize(text.lower()) for text in human_texts]
    space = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=1, seed=0)

    def strategy(span: Span, text: str) -> str | None:
        token = span.text.lower()
        if token not in space.wv.key_to_index:
            return None
        (neighbour, _), = space.wv.most_similar(token, topn=1)
        return neighbour

    return strategy


def psr_strategy(human_texts: Sequence[str]) -> Strategy:
    """HSR, but only accept a neighbour with the same part-of-speech tag."""
    sentences = [nltk.word_tokenize(text.lower()) for text in human_texts]
    space = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=1, seed=0)

    def strategy(span: Span, text: str) -> str | None:
        token = span.text.lower()
        if token not in space.wv.key_to_index:
            return None
        (_, original_tag), = nltk.pos_tag([token])
        for neighbour, _ in space.wv.most_similar(token, topn=10):
            (_, tag), = nltk.pos_tag([neighbour])
            if tag == original_tag:
                return neighbour
        return None

    return strategy


def datamuse_strategy() -> Strategy:
    """Replace tokens with the top Datamuse "means like" suggestion."""

    def strategy(span: Span, text: str) -> str | None:
        response = requests.get(
            "https://api.datamuse.com/words",
            params={"ml": span.text.lower(), "max": 1},
            timeout=10,
        )
        words = response.json()
        return words[0]["word"] if words else None

    return strategy


if __name__ == "__main__":
    from explainkit import Explainer, robustness_probe
    from explainkit.demo import _NaiveBayes
    from explainkit.dataset import generate, train_test_split

    train, test = train_test_split(generate(240, seed=0), n_test=40)
    model = _NaiveBayes().fit(train.texts, train.labels)
    strategy = hsr_strategy(train.texts)
    report = robustness_probe(Explainer(model.predict_proba), test.texts[:10], strategy, top_k=3)
    print(f"HSR on 10 texts: mean drop {report.mean_drop:+.3f}, flip rate {report.flip_rate:.2f}")
