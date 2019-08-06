import logging

from fonduer.parser.lingual_parser import SimpleParser


def test_simple_split_sentences(caplog):
    caplog.set_level(logging.INFO)

    tokenize_and_split_sentences = SimpleParser().split_sentences
    text = "This is a text.<NB>This is another text."

    iterator = tokenize_and_split_sentences(text)
    assert len(list(iterator)) == 2
