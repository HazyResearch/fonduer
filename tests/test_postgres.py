"""Unit tests that involve postgres access."""
import logging

from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionFigures
from fonduer.candidates.matchers import LambdaFunctionFigureMatcher
from fonduer.candidates.models import (
    Candidate,
    Mention,
    candidate_subclass,
    mention_subclass,
)
from fonduer.parser import Parser
from fonduer.parser.models import Document, Sentence
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser.visual_parser import PdfVisualParser
from tests.shared.hardware_matchers import part_matcher, temp_matcher, volt_matcher
from tests.shared.hardware_spaces import (
    MentionNgramsPart,
    MentionNgramsTemp,
    MentionNgramsVolt,
)
from tests.shared.hardware_throttlers import temp_throttler

logger = logging.getLogger(__name__)


def test_preprocessor_parse_file_called_once(database_session, mocker):
    """Test if DocPreprocessor._parse_file is called only once during parser.apply."""
    max_docs = 1
    session = database_session
    docs_path = "tests/data/html/"
    # Set up preprocessor, parser, and spy on preprocessor
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    spy = mocker.spy(doc_preprocessor, "_parse_file")
    corpus_parser = Parser(session)

    # Check if udf.last_docs is empty.
    assert len(corpus_parser.get_last_documents()) == 0

    # Parsing
    corpus_parser.apply(doc_preprocessor)

    # Check if udf.last_docs is correctly updated.
    assert len(corpus_parser.get_last_documents()) == max_docs
    # doc_preprocessor._parse_file should be called only once (#434).
    spy.assert_called_once()


def test_cand_gen_cascading_delete(database_session):
    """Test cascading the deletion of candidates."""
    # GitHub Actions gives 2 cores
    # help.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners
    PARALLEL = 2

    max_docs = 1
    session = database_session

    docs_path = "tests/data/html/"
    pdf_path = "tests/data/pdf/"

    # Parsing
    logger.info("Parsing...")
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = Parser(
        session,
        structural=True,
        lingual=True,
        visual_parser=PdfVisualParser(pdf_path),
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

    assert session.query(PartTemp).count() == 1431
    assert session.query(Candidate).count() == 1431
    assert docs[0].name == "112823"
    assert len(docs[0].parts) == 70
    assert len(docs[0].temps) == 23

    # Delete from parent class should cascade to child
    x = session.query(Candidate).first()
    session.query(Candidate).filter_by(id=x.id).delete(synchronize_session="fetch")
    assert session.query(Candidate).count() == 1430
    assert session.query(PartTemp).count() == 1430

    # Test that deletion of a Candidate does not delete the Mention
    x = session.query(PartTemp).first()
    candidate = session.query(PartTemp).filter_by(id=x.id).first()
    session.delete(candidate)
    assert session.query(PartTemp).count() == 1429
    assert session.query(Temp).count() == 23
    assert session.query(Part).count() == 70

    # Clearing Mentions should also delete Candidates
    mention_extractor.clear()
    assert session.query(Mention).count() == 0
    assert session.query(Part).count() == 0
    assert session.query(Temp).count() == 0
    assert session.query(PartTemp).count() == 0
    assert session.query(Candidate).count() == 0


def test_too_many_clients_error_should_not_happen(database_session):
    """Too many clients error should not happens."""
    PARALLEL = 32
    logger.info("Parallel: {PARALLEL}")

    def do_nothing_matcher(fig):
        return True

    max_docs = 1
    session = database_session

    docs_path = "tests/data/html/"
    pdf_path = "tests/data/pdf/"

    # Parsing
    logger.info("Parsing...")
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = Parser(
        session,
        structural=True,
        lingual=True,
        visual_parser=PdfVisualParser(pdf_path),
    )
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
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

    mention_extractor = MentionExtractor(
        session,
        [Part, Temp, Volt, Fig],
        [part_ngrams, temp_ngrams, volt_ngrams, figs],
        [part_matcher, temp_matcher, volt_matcher, fig_matcher],
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])
    PartVolt = candidate_subclass("PartVolt", [Part, Volt])

    # Test that no throttler in candidate extractor
    candidate_extractor = CandidateExtractor(
        session, [PartTemp, PartVolt]
    )  # Pass, no throttler

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)
    candidate_extractor.clear_all(split=0)

    # Test with None in throttlers in candidate extractor
    candidate_extractor = CandidateExtractor(
        session, [PartTemp, PartVolt], throttlers=[temp_throttler, None]
    )

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)


def test_parse_error_doc_skipping(database_session):
    """Test skipping of faulty htmls."""
    faulty_doc_path = "tests/data/html_faulty/ext_diseases_missing_table_tag.html"
    preprocessor = HTMLDocPreprocessor(faulty_doc_path)
    session = database_session
    corpus_parser = Parser(session)
    corpus_parser.apply(preprocessor)
    # This returns documents that apply() was called on
    assert corpus_parser.last_docs == {"ext_diseases_missing_table_tag"}
    # This returns only documents that are successfully parsed.
    assert corpus_parser.get_last_documents() == []
