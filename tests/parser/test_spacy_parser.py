#! /usr/bin/env python
import logging

import pytest

from fonduer.parser.spacy_parser import Spacy


def test_spacy_support(caplog):
    caplog.set_level(logging.INFO)

    # Supported language
    lingual_parser = Spacy("en")
    assert lingual_parser.has_tokenizer_support()
    assert lingual_parser.has_NLP_support()

    # Alpha-supported language
    lingual_parser = Spacy("ja")
    assert lingual_parser.has_tokenizer_support()
    assert not lingual_parser.has_NLP_support()

    # Non supported language
    lingual_parser = Spacy("non-supported-lang")
    assert not lingual_parser.has_tokenizer_support()
    assert not lingual_parser.has_NLP_support()

    # Language not specified
    with pytest.raises(TypeError):
        lingual_parser = Spacy()


def test_spacy_split_sentences(caplog):
    caplog.set_level(logging.INFO)

    lingual_parser = Spacy("en")
    tokenize_and_split_sentences = lingual_parser.split_sentences
    text = "This is a text. This is another text."

    iterator = tokenize_and_split_sentences(text)
    with pytest.raises(AttributeError):
        next(iterator)

    lingual_parser.load_lang_model()
    iterator = tokenize_and_split_sentences(text)
    assert len(list(iterator)) == 2
