import os

import pytest

from fonduer.parser.lingual_parser.spacy_parser import (
    SpacyParser,
    TokenPreservingTokenizer,
    set_custom_boundary,
)
from fonduer.parser.models import Sentence


@pytest.mark.skipif(
    "CI" not in os.environ, reason="Only run non-English tests on Travis"
)
def test_spacy_support():
    # Supported language
    lingual_parser = SpacyParser("en")
    assert lingual_parser.has_tokenizer_support()
    assert lingual_parser.has_NLP_support()

    # Alpha-supported language
    lingual_parser = SpacyParser("ja")
    assert lingual_parser.has_tokenizer_support()
    assert not lingual_parser.has_NLP_support()

    # Non supported language
    lingual_parser = SpacyParser("non-supported-lang")
    assert not lingual_parser.has_tokenizer_support()
    assert not lingual_parser.has_NLP_support()

    # Language not specified
    with pytest.raises(TypeError):
        lingual_parser = SpacyParser()


def test_spacy_split_sentences():
    lingual_parser = SpacyParser("en")
    tokenize_and_split_sentences = lingual_parser.split_sentences
    text = "This is a text. This is another text."

    iterator = tokenize_and_split_sentences(text)
    assert len(list(iterator)) == 2


def test_split_sentences_by_char_limit():
    lingual_parser = SpacyParser("en")

    text = "This is a text. This is another text."
    all_sentences = [
        Sentence(**parts) for parts in lingual_parser.split_sentences(text)
    ]
    assert len(all_sentences) == 2
    assert [len(sentence.text) for sentence in all_sentences] == [15, 21]

    lingual_parser.model.remove_pipe("sentencizer")
    lingual_parser.model.add_pipe(
        set_custom_boundary, before="parser", name="sentence_boundary_detector"
    )

    sentence_batches = lingual_parser._split_sentences_by_char_limit(all_sentences, 20)
    assert len(sentence_batches) == 2
    sentence_batches = lingual_parser._split_sentences_by_char_limit(all_sentences, 100)
    assert len(sentence_batches) == 1

    sentence_batch = sentence_batches[0]
    custom_tokenizer = TokenPreservingTokenizer(lingual_parser.model.vocab)
    doc = custom_tokenizer(sentence_batch)
    doc.user_data = sentence_batch
    for name, proc in lingual_parser.model.pipeline:  # iterate over components in order
        doc = proc(doc)
    assert doc.is_parsed

    # See if the number of parsed spaCy sentences matches that of input sentences
    assert len(list(doc.sents)) == len(sentence_batch)
