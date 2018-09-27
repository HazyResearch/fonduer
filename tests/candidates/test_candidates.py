#! /usr/bin/env python
import logging
import os

import pytest

from fonduer import Meta
from fonduer.candidates import (
    CandidateExtractor,
    MentionExtractor,
    MentionFigures,
    MentionNgrams,
)
from fonduer.candidates.matchers import (
    LambdaFunctionFigureMatcher,
    LambdaFunctionMatcher,
    PersonMatcher,
)
from fonduer.candidates.mentions import Ngrams
from fonduer.candidates.models import Candidate, candidate_subclass, mention_subclass
from fonduer.parser import Parser
from fonduer.parser.models import Document, Sentence
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.utils.data_model_utils import get_row_ngrams
from tests.shared.hardware_matchers import part_matcher, temp_matcher, volt_matcher
from tests.shared.hardware_spaces import (
    MentionNgramsPart,
    MentionNgramsTemp,
    MentionNgramsVolt,
)
from tests.shared.hardware_throttlers import temp_throttler, volt_throttler

logger = logging.getLogger(__name__)
ATTRIBUTE = "stg_temp_max"
DB = "cand_test"


def test_ngram_split(caplog):
    """Test ngram split."""
    caplog.set_level(logging.INFO)
    ngrams = Ngrams(split_tokens=["-", "/"])
    sent = Sentence()

    # When a split_token appears in the middle of the text.
    sent.text = "New-Text"
    sent.words = ["New-Text"]
    sent.char_offsets = [0]
    sent.abs_char_offsets = [0]
    result = list(ngrams.apply(sent))

    assert len(result) == 3
    assert result[0].get_span() == "New-Text"
    assert result[1].get_span() == "New"
    assert result[2].get_span() == "Text"

    # When a text ends with a split_token.
    sent.text = "New-"
    sent.words = ["New-"]
    result = list(ngrams.apply(sent))

    assert len(result) == 2
    assert result[0].get_span() == "New-"
    assert result[1].get_span() == "New"

    # When a text starts with a split_token.
    sent.text = "-Text"
    sent.words = ["-Text"]
    result = list(ngrams.apply(sent))

    assert len(result) == 2
    assert result[0].get_span() == "-Text"
    assert result[1].get_span() == "Text"

    # When more than one split_token appears.
    sent.text = "New/Text-Word"
    sent.words = ["New/Text-Word"]
    result = list(ngrams.apply(sent))

    assert len(result) == 6
    spans = [r.get_span() for r in result]
    assert "New/Text-Word" in spans
    assert "New" in spans
    assert "New/Text" in spans
    assert "Text" in spans
    assert "Text-Word" in spans
    assert "Word" in spans

    sent.text = "A-B/C-D"
    sent.words = ["A-B/C-D"]
    result = list(ngrams.apply(sent))

    assert len(result) == 10
    spans = [r.get_span() for r in result]
    assert "A-B/C-D" in spans
    assert "A-B/C" in spans
    assert "B/C-D" in spans
    assert "A-B" in spans
    assert "C-D" in spans
    assert "B/C" in spans
    assert "A" in spans
    assert "B" in spans
    assert "C" in spans
    assert "D" in spans

    ngrams = Ngrams(split_tokens=["~", "~~"])
    sent = Sentence()

    sent.text = "a~b~~c~d"
    sent.words = ["a~b~~c~d"]
    sent.char_offsets = [0]
    sent.abs_char_offsets = [0]
    result = list(ngrams.apply(sent))

    assert len(result) == 10
    spans = [r.get_span() for r in result]
    assert "a~b~~c~d" in spans
    assert "a" in spans
    assert "a~b" in spans
    assert "a~b~~c" in spans
    assert "b" in spans
    assert "b~~c" in spans
    assert "b~~c~d" in spans
    assert "c" in spans
    assert "c~d" in spans
    assert "d" in spans

    ngrams = Ngrams(split_tokens=["~a", "a~"])
    sent = Sentence()

    sent.text = "~a~b~~c~d"
    sent.words = ["~a~b~~c~d"]
    sent.char_offsets = [0]
    sent.abs_char_offsets = [0]
    result = list(ngrams.apply(sent))

    assert len(result) == 2
    spans = [r.get_span() for r in result]
    assert "~a~b~~c~d" in spans
    assert "~b~~c~d" in spans

    ngrams = Ngrams(split_tokens=["-", "/", "*"])
    sent = Sentence()

    sent.text = "A-B/C*D"
    sent.words = ["A-B/C*D"]
    sent.char_offsets = [0]
    sent.abs_char_offsets = [0]
    result = list(ngrams.apply(sent))

    assert len(result) == 10
    spans = [r.get_span() for r in result]
    assert "A-B/C*D" in spans
    assert "A" in spans
    assert "A-B" in spans
    assert "A-B/C" in spans
    assert "B" in spans
    assert "B/C" in spans
    assert "B/C*D" in spans
    assert "C" in spans
    assert "C*D" in spans
    assert "D" in spans


def test_span_char_start_and_char_end(caplog):
    """Test chart_start and char_end of TemporarySpan that comes from Ngrams.apply."""
    caplog.set_level(logging.INFO)
    ngrams = Ngrams()
    sent = Sentence()
    sent.text = "BC548BG"
    sent.words = ["BC548BG"]
    sent.char_offsets = [0]
    sent.abs_char_offsets = [0]
    result = list(ngrams.apply(sent))

    assert len(result) == 1
    assert result[0].get_span() == "BC548BG"
    assert result[0].char_start == 0
    assert result[0].char_end == 6


def test_cand_gen(caplog):
    """Test extracting candidates from mentions from documents."""
    caplog.set_level(logging.INFO)
    # SpaCy on mac has issue on parallel parsing
    if os.name == "posix":
        logger.info("Using single core.")
        PARALLEL = 1
    else:
        PARALLEL = 2  # Travis only gives 2 cores

    def do_nothing_matcher(fig):
        return True

    max_docs = 10
    session = Meta.init("postgres://localhost:5432/" + DB).Session()

    docs_path = "tests/data/html/"
    pdf_path = "tests/data/pdf/"

    # Parsing
    logger.info("Parsing...")
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = Parser(
        session, structural=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
    assert session.query(Document).count() == max_docs
    assert session.query(Sentence).count() == 5548
    docs = session.query(Document).order_by(Document.name).all()

    # Mention Extraction
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
    temp_ngrams = MentionNgramsTemp(n_max=2)
    volt_ngrams = MentionNgramsVolt(n_max=1)
    figs = MentionFigures(types="png")

    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")
    Volt = mention_subclass("Volt")
    Fig = mention_subclass("Fig")

    fig_matcher = LambdaFunctionFigureMatcher(func=do_nothing_matcher)

    with pytest.raises(ValueError):
        mention_extractor = MentionExtractor(
            session,
            [Part, Temp, Volt],
            [part_ngrams, volt_ngrams],  # Fail, mismatched arity
            [part_matcher, temp_matcher, volt_matcher],
        )
    with pytest.raises(ValueError):
        mention_extractor = MentionExtractor(
            session,
            [Part, Temp, Volt],
            [part_ngrams, temp_matcher, volt_ngrams],
            [part_matcher, temp_matcher],  # Fail, mismatched arity
        )

    mention_extractor = MentionExtractor(
        session,
        [Part, Temp, Volt, Fig],
        [part_ngrams, temp_ngrams, volt_ngrams, figs],
        [part_matcher, temp_matcher, volt_matcher, fig_matcher],
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Part).count() == 234
    assert session.query(Volt).count() == 107
    assert session.query(Temp).count() == 136
    assert session.query(Fig).count() == 223
    part = session.query(Part).order_by(Part.id).all()[0]
    volt = session.query(Volt).order_by(Volt.id).all()[0]
    temp = session.query(Temp).order_by(Temp.id).all()[0]
    logger.info("Part: {}".format(part.span))
    logger.info("Volt: {}".format(volt.span))
    logger.info("Temp: {}".format(temp.span))

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])
    PartVolt = candidate_subclass("PartVolt", [Part, Volt])

    with pytest.raises(ValueError):
        candidate_extractor = CandidateExtractor(
            session,
            [PartTemp, PartVolt],
            throttlers=[
                temp_throttler,
                volt_throttler,
                volt_throttler,
            ],  # Fail, mismatched arity
        )

    with pytest.raises(ValueError):
        candidate_extractor = CandidateExtractor(
            session,
            [PartTemp],  # Fail, mismatched arity
            throttlers=[temp_throttler, volt_throttler],
        )

    # Test that no throttler in candidate extractor
    candidate_extractor = CandidateExtractor(
        session, [PartTemp, PartVolt]
    )  # Pass, no throttler

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    assert session.query(PartTemp).count() == 4141
    assert session.query(PartVolt).count() == 3610
    assert session.query(Candidate).count() == 7751
    candidate_extractor.clear_all(split=0)
    assert session.query(Candidate).count() == 0

    # Test with None in throttlers in candidate extractor
    candidate_extractor = CandidateExtractor(
        session, [PartTemp, PartVolt], throttlers=[temp_throttler, None]
    )

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    assert session.query(PartTemp).count() == 3879
    assert session.query(PartVolt).count() == 3610
    assert session.query(Candidate).count() == 7489
    candidate_extractor.clear_all(split=0)
    assert session.query(Candidate).count() == 0

    candidate_extractor = CandidateExtractor(
        session, [PartTemp, PartVolt], throttlers=[temp_throttler, volt_throttler]
    )

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    assert session.query(PartTemp).count() == 3879
    assert session.query(PartVolt).count() == 3266
    assert session.query(Candidate).count() == 7145
    assert docs[0].name == "112823"
    assert len(docs[0].parts) == 70
    assert len(docs[0].volts) == 33
    assert len(docs[0].temps) == 24

    # Test that deletion of a Candidate does not delete the Mention
    session.query(PartTemp).delete()
    assert session.query(PartTemp).count() == 0
    assert session.query(Temp).count() == 136
    assert session.query(Part).count() == 234

    # Test deletion of Candidate if Mention is deleted
    assert session.query(PartVolt).count() == 3266
    assert session.query(Volt).count() == 107
    session.query(Volt).delete()
    assert session.query(Volt).count() == 0
    assert session.query(PartVolt).count() == 0


def test_ngrams(caplog):
    """Test ngram limits in mention extraction"""
    caplog.set_level(logging.INFO)
    PARALLEL = 1

    max_docs = 1
    session = Meta.init("postgres://localhost:5432/" + DB).Session()

    docs_path = "tests/data/pure_html/lincoln_short.html"

    logger.info("Parsing...")
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = Parser(session, structural=True, lingual=True)
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
    assert session.query(Document).count() == max_docs
    assert session.query(Sentence).count() == 503
    docs = session.query(Document).order_by(Document.name).all()

    # Mention Extraction
    Person = mention_subclass("Person")
    person_ngrams = MentionNgrams(n_max=3)
    person_matcher = PersonMatcher()

    mention_extractor = MentionExtractor(
        session, [Person], [person_ngrams], [person_matcher]
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Person).count() == 126
    mentions = session.query(Person).all()
    assert len([x for x in mentions if x.span.get_num_words() == 1]) == 50
    assert len([x for x in mentions if x.span.get_num_words() > 3]) == 0

    # Test for unigram exclusion
    person_ngrams = MentionNgrams(n_min=2, n_max=3)
    mention_extractor = MentionExtractor(
        session, [Person], [person_ngrams], [person_matcher]
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)
    assert session.query(Person).count() == 76
    mentions = session.query(Person).all()
    assert len([x for x in mentions if x.span.get_num_words() == 1]) == 0
    assert len([x for x in mentions if x.span.get_num_words() > 3]) == 0


def test_mention_longest_match(caplog):
    """Test longest match filtering in mention extraction."""
    caplog.set_level(logging.INFO)
    # SpaCy on mac has issue on parallel parsing
    PARALLEL = 1

    max_docs = 1
    session = Meta.init("postgres://localhost:5432/" + DB).Session()

    docs_path = "tests/data/pure_html/lincoln_short.html"

    # Parsing
    logger.info("Parsing...")
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = Parser(session, structural=True, lingual=True)
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
    docs = session.query(Document).order_by(Document.name).all()
    # Mention Extraction
    name_ngrams = MentionNgramsPart(n_max=3)
    place_ngrams = MentionNgramsTemp(n_max=4)

    Name = mention_subclass("Name")
    Place = mention_subclass("Place")

    def is_birthplace_table_row(mention):
        if not mention.sentence.is_tabular():
            return False
        ngrams = get_row_ngrams(mention, lower=True)
        if "birth_place" in ngrams:
            return True
        else:
            return False

    birthplace_matcher = LambdaFunctionMatcher(
        func=is_birthplace_table_row, longest_match_only=False
    )
    mention_extractor = MentionExtractor(
        session,
        [Name, Place],
        [name_ngrams, place_ngrams],
        [PersonMatcher(), birthplace_matcher],
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)
    mentions = session.query(Place).all()
    mention_spans = [x.span.get_span() for x in mentions]
    assert "Sinking Spring Farm" in mention_spans
    assert "Farm" in mention_spans
    assert len(mention_spans) == 23

    birthplace_matcher = LambdaFunctionMatcher(
        func=is_birthplace_table_row, longest_match_only=True
    )
    mention_extractor = MentionExtractor(
        session,
        [Name, Place],
        [name_ngrams, place_ngrams],
        [PersonMatcher(), birthplace_matcher],
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)
    mentions = session.query(Place).all()
    mention_spans = [x.span.get_span() for x in mentions]
    assert "Sinking Spring Farm" in mention_spans
    assert "Farm" not in mention_spans
    assert len(mention_spans) == 4
