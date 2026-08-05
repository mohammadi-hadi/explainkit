"""Word tokenizers that keep character offsets.

The package works at the word level on purpose: words are what humans read in
an explanation, and what selection, steering and replacement operate on.  A
model's subword vocabulary stays its own business — perturbed texts are handed
back to the model as raw strings and it re-tokenizes them however it likes.
"""

from __future__ import annotations

import re

from explainkit.schema import Span, Tokenizer

# Unicode word characters, with an optional apostrophe part ("don't", "l'idea").
WORD_PATTERN = r"\w+(?:'\w+)?"

_WORD = re.compile(WORD_PATTERN)


def word_tokenizer(text: str) -> list[Span]:
    """The default tokenizer: unicode words with their character spans."""
    return [Span(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text)]


def regex_tokenizer(pattern: str) -> Tokenizer:
    """A tokenizer whose tokens are the matches of ``pattern``."""
    compiled = re.compile(pattern)

    def tokenize(text: str) -> list[Span]:
        return [Span(m.group(0), m.start(), m.end()) for m in compiled.finditer(text)]

    return tokenize
