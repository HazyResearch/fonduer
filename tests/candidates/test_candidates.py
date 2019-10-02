import logging
from sys import platform

import pytest

from fonduer import Meta
from fonduer.candidates import (
    CandidateExtractor,
    MentionCaptions,
    MentionCells,
    MentionDocuments,
    MentionExtractor,
    MentionFigures,
    MentionNgrams,
    MentionParagraphs,
    MentionSections,
    MentionSentences,
    MentionTables,
)
from fonduer.candidates.matchers import (
    DoNothingMatcher,
    LambdaFunctionFigureMatcher,
    LambdaFunctionMatcher,
    PersonMatcher,
)
from fonduer.candidates.mentions import Ngrams
from fonduer.candidates.models import (
    Candidate,
    Mention,
    candidate_subclass,
    mention_subclass,
)
from fonduer.parser import Parser
from fonduer.parser.models import Document, Sentence
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.utils.data_model_utils import get_col_ngrams, get_row_ngrams
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


def test_cand_gen_cascading_delete(caplog):
    """Test cascading the deletion of candidates."""
    caplog.set_level(logging.INFO)

    if platform == "darwin":
        logger.info("Using single core.")
        PARALLEL = 1
    else:
        logger.info("Using two cores.")
        PARALLEL = 2  # Travis only gives 2 cores

    max_docs = 1
    session = Meta.init("postgresql://localhost:5432/" + DB).Session()

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
    assert session.query(Sentence).count() == 799
    docs = session.query(Document).order_by(Document.name).all()

    # Mention Extraction
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
    temp_ngrams = MentionNgramsTemp(n_max=2)

    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")

    mention_extractor = MentionExtractor(
        session, [Part, Temp], [part_ngrams, temp_ngrams], [part_matcher, temp_matcher]
    )
    mention_extractor.clear_all()
    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Mention).count() == 93
    assert session.query(Part).count() == 70
    assert session.query(Temp).count() == 23
    part = session.query(Part).order_by(Part.id).all()[0]
    temp = session.query(Temp).order_by(Temp.id).all()[0]
    logger.info(f"Part: {part.context}")
    logger.info(f"Temp: {temp.context}")

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])

    candidate_extractor = CandidateExtractor(
        session, [PartTemp], throttlers=[temp_throttler]
    )

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    assert session.query(PartTemp).count() == 1432
    assert session.query(Candidate).count() == 1432
    assert docs[0].name == "112823"
    assert len(docs[0].parts) == 70
    assert len(docs[0].temps) == 23

    # Delete from parent class should cascade to child
    x = session.query(Candidate).first()
    session.query(Candidate).filter_by(id=x.id).delete(synchronize_session="fetch")
    assert session.query(Candidate).count() == 1431
    assert session.query(PartTemp).count() == 1431

    # Clearing Mentions should also delete Candidates
    mention_extractor.clear()
    assert session.query(Mention).count() == 0
    assert session.query(Part).count() == 0
    assert session.query(Temp).count() == 0
    assert session.query(PartTemp).count() == 0
    assert session.query(Candidate).count() == 0


def test_cand_gen(caplog):
    """Test extracting candidates from mentions from documents."""
    caplog.set_level(logging.INFO)

    if platform == "darwin":
        logger.info("Using single core.")
        PARALLEL = 1
    else:
        logger.info("Using as many cores as available (PARALLEL=32)")
        PARALLEL = 32

    def do_nothing_matcher(fig):
        return True

    max_docs = 1
    session = Meta.init("postgresql://localhost:5432/" + DB).Session()

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
    assert session.query(Sentence).count() == 799
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

    assert session.query(Part).count() == 70
    assert session.query(Volt).count() == 33
    assert session.query(Temp).count() == 23
    assert session.query(Fig).count() == 31
    part = session.query(Part).order_by(Part.id).all()[0]
    volt = session.query(Volt).order_by(Volt.id).all()[0]
    temp = session.query(Temp).order_by(Temp.id).all()[0]
    logger.info(f"Part: {part.context}")
    logger.info(f"Volt: {volt.context}")
    logger.info(f"Temp: {temp.context}")

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

    assert session.query(PartTemp).count() == 1610
    assert session.query(PartVolt).count() == 2310
    assert session.query(Candidate).count() == 3920
    candidate_extractor.clear_all(split=0)
    assert session.query(Candidate).count() == 0
    assert session.query(PartTemp).count() == 0
    assert session.query(PartVolt).count() == 0

    # Test with None in throttlers in candidate extractor
    candidate_extractor = CandidateExtractor(
        session, [PartTemp, PartVolt], throttlers=[temp_throttler, None]
    )

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)
    assert session.query(PartTemp).count() == 1432
    assert session.query(PartVolt).count() == 2310
    assert session.query(Candidate).count() == 3742
    candidate_extractor.clear_all(split=0)
    assert session.query(Candidate).count() == 0

    candidate_extractor = CandidateExtractor(
        session, [PartTemp, PartVolt], throttlers=[temp_throttler, volt_throttler]
    )

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    assert session.query(PartTemp).count() == 1432
    assert session.query(PartVolt).count() == 1993
    assert session.query(Candidate).count() == 3425
    assert docs[0].name == "112823"
    assert len(docs[0].parts) == 70
    assert len(docs[0].volts) == 33
    assert len(docs[0].temps) == 23

    # Test that deletion of a Candidate does not delete the Mention
    session.query(PartTemp).delete(synchronize_session="fetch")
    assert session.query(PartTemp).count() == 0
    assert session.query(Temp).count() == 23
    assert session.query(Part).count() == 70

    # Test deletion of Candidate if Mention is deleted
    assert session.query(PartVolt).count() == 1993
    assert session.query(Volt).count() == 33
    session.query(Volt).delete(synchronize_session="fetch")
    assert session.query(Volt).count() == 0
    assert session.query(PartVolt).count() == 0


def test_ngrams(caplog):
    """Test ngram limits in mention extraction"""
    caplog.set_level(logging.INFO)

    PARALLEL = 4

    max_docs = 1
    session = Meta.init("postgresql://localhost:5432/" + DB).Session()

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

    assert session.query(Person).count() == 118
    mentions = session.query(Person).all()
    assert len([x for x in mentions if x.context.get_num_words() == 1]) == 49
    assert len([x for x in mentions if x.context.get_num_words() > 3]) == 0

    # Test for unigram exclusion
    person_ngrams = MentionNgrams(n_min=2, n_max=3)
    mention_extractor = MentionExtractor(
        session, [Person], [person_ngrams], [person_matcher]
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)
    assert session.query(Person).count() == 69
    mentions = session.query(Person).all()
    assert len([x for x in mentions if x.context.get_num_words() == 1]) == 0
    assert len([x for x in mentions if x.context.get_num_words() > 3]) == 0


def test_row_col_ngram_extraction(caplog):
    """Test whether row/column ngrams list is empty, if mention is not in a table."""
    caplog.set_level(logging.INFO)
    PARALLEL = 1
    max_docs = 1
    session = Meta.init("postgresql://localhost:5432/" + DB).Session()
    docs_path = "tests/data/pure_html/lincoln_short.html"

    # Parsing
    logger.info("Parsing...")
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = Parser(session, structural=True, lingual=True)
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
    docs = session.query(Document).order_by(Document.name).all()

    # Mention Extraction
    place_ngrams = MentionNgramsTemp(n_max=4)
    Place = mention_subclass("Place")

    def get_row_and_column_ngrams(mention):
        row_ngrams = list(get_row_ngrams(mention))
        col_ngrams = list(get_col_ngrams(mention))
        if not mention.sentence.is_tabular():
            assert len(row_ngrams) == 1 and row_ngrams[0] is None
            assert len(col_ngrams) == 1 and col_ngrams[0] is None
        else:
            assert not any(x is None for x in row_ngrams)
            assert not any(x is None for x in col_ngrams)
        if "birth_place" in row_ngrams:
            return True
        else:
            return False

    birthplace_matcher = LambdaFunctionMatcher(func=get_row_and_column_ngrams)
    mention_extractor = MentionExtractor(
        session, [Place], [place_ngrams], [birthplace_matcher]
    )

    mention_extractor.apply(docs, parallelism=PARALLEL)


def test_mention_longest_match(caplog):
    """Test longest match filtering in mention extraction."""
    caplog.set_level(logging.INFO)
    # SpaCy on mac has issue on parallel parsing
    PARALLEL = 1

    max_docs = 1
    session = Meta.init("postgresql://localhost:5432/" + DB).Session()

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
    mention_spans = [x.context.get_span() for x in mentions]
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
    mention_spans = [x.context.get_span() for x in mentions]
    assert "Sinking Spring Farm" in mention_spans
    assert "Farm" not in mention_spans
    assert len(mention_spans) == 4


def test_multimodal_cand(caplog):
    """Test multimodal candidate generation"""
    caplog.set_level(logging.INFO)

    PARALLEL = 4

    max_docs = 1
    session = Meta.init("postgresql://localhost:5432/" + DB).Session()

    docs_path = "tests/data/pure_html/radiology.html"

    logger.info("Parsing...")
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = Parser(session, structural=True, lingual=True)
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
    assert session.query(Document).count() == max_docs

    assert session.query(Sentence).count() == 35
    docs = session.query(Document).order_by(Document.name).all()

    # Mention Extraction

    ms_doc = mention_subclass("m_doc")
    ms_sec = mention_subclass("m_sec")
    ms_tab = mention_subclass("m_tab")
    ms_fig = mention_subclass("m_fig")
    ms_cell = mention_subclass("m_cell")
    ms_para = mention_subclass("m_para")
    ms_cap = mention_subclass("m_cap")
    ms_sent = mention_subclass("m_sent")

    m_doc = MentionDocuments()
    m_sec = MentionSections()
    m_tab = MentionTables()
    m_fig = MentionFigures()
    m_cell = MentionCells()
    m_para = MentionParagraphs()
    m_cap = MentionCaptions()
    m_sent = MentionSentences()

    ms = [ms_doc, ms_cap, ms_sec, ms_tab, ms_fig, ms_para, ms_sent, ms_cell]
    m = [m_doc, m_cap, m_sec, m_tab, m_fig, m_para, m_sent, m_cell]
    matchers = [DoNothingMatcher()] * 8

    mention_extractor = MentionExtractor(session, ms, m, matchers, parallelism=PARALLEL)

    mention_extractor.apply(docs)

    assert session.query(ms_doc).count() == 1
    assert session.query(ms_cap).count() == 2
    assert session.query(ms_sec).count() == 5
    assert session.query(ms_tab).count() == 2
    assert session.query(ms_fig).count() == 2
    assert session.query(ms_para).count() == 30
    assert session.query(ms_sent).count() == 35
    assert session.query(ms_cell).count() == 21


def test_subclass_before_meta_init(caplog):
    """Test if it is possible to create a mention (candidate) subclass even before Meta
    is initialized.
    """
    caplog.set_level(logging.INFO)

    conn_string = "postgresql://localhost:5432/" + DB
    Part = mention_subclass("Part")
    logger.info(f"Create a mention subclass '{Part.__tablename__}'")
    Meta.init(conn_string).Session()
    Temp = mention_subclass("Temp")
    logger.info(f"Create a mention subclass '{Temp.__tablename__}'")
