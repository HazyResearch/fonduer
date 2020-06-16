"""Fonduer visual_linker unit tests."""
import random
from operator import attrgetter

from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser.visual_linker import VisualLinker
from tests.parser.test_parser import get_parser_udf


def test_visual_linker_not_affected_by_order_of_sentences():
    """Test if visual_linker result is not affected by the order of sentences."""
    docs_path = "tests/data/html/2N6427.html"
    pdf_path = "tests/data/pdf/2N6427.pdf"

    # Initialize preprocessor, parser, visual_linker.
    # Note that parser is initialized with `visual=False` and that visual_linker
    # will be used to attach "visual" information to sentences after parsing.
    preprocessor = HTMLDocPreprocessor(docs_path)
    parser_udf = get_parser_udf(
        structural=True, lingual=False, tabular=True, visual=False
    )
    visual_linker = VisualLinker(pdf_path=pdf_path)

    doc = parser_udf.apply(next(preprocessor.__iter__()))
    # Sort sentences by sentence.position
    doc.sentences = sorted(doc.sentences, key=attrgetter("position"))
    sentences0 = [
        sent for sent in visual_linker.link(doc.name, doc.sentences, pdf_path)
    ]
    # Sort again in case visual_linker.link changes the order
    sentences0 = sorted(sentences0, key=attrgetter("position"))

    doc = parser_udf.apply(next(preprocessor.__iter__()))
    # Shuffle
    random.shuffle(doc.sentences)
    sentences1 = [
        sent for sent in visual_linker.link(doc.name, doc.sentences, pdf_path)
    ]
    # Sort sentences by sentence.position
    sentences1 = sorted(sentences1, key=attrgetter("position"))

    # This should hold as both sentences are sorted by their position
    assert all(
        [
            sent0.position == sent1.position
            for (sent0, sent1) in zip(sentences0, sentences1)
        ]
    )

    # The following assertion should hold if the visual_linker result is not affected
    # by the order of sentences.
    assert all(
        [sent0.left == sent1.left for (sent0, sent1) in zip(sentences0, sentences1)]
    )
