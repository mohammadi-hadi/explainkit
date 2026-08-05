"""Every span must read back as its own token — the package's one invariant."""

from __future__ import annotations

from explainkit import regex_tokenizer, word_tokenizer


def test_spans_read_back_exactly() -> None:
    text = "Well, José didn't say — l'idea was fine (mostly)."
    for span in word_tokenizer(text):
        assert text[span.start : span.end] == span.text


def test_apostrophes_stay_inside_words() -> None:
    tokens = [span.text for span in word_tokenizer("didn't we")]
    assert tokens == ["didn't", "we"]


def test_empty_and_symbol_only_texts() -> None:
    assert word_tokenizer("") == []
    assert word_tokenizer("   ") == []
    assert word_tokenizer("!!! ... ???") == []


def test_unicode_words_are_tokens() -> None:
    tokens = [span.text for span in word_tokenizer("señora Müller étude")]
    assert tokens == ["señora", "Müller", "étude"]


def test_regex_tokenizer_honors_pattern() -> None:
    tokenize = regex_tokenizer(r"[A-Z]+")
    text = "ABC def GHI"
    spans = tokenize(text)
    assert [span.text for span in spans] == ["ABC", "GHI"]
    for span in spans:
        assert text[span.start : span.end] == span.text
