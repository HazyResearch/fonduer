"""
These tests expect that postgres is installed and that a database named
visualizer_test has been created for the purpose of testing.

If you are testing locally, you will need to create this db.
"""
from fonduer import Meta
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.matchers import OrganizationMatcher
from fonduer.candidates.models import candidate_subclass, mention_subclass
from fonduer.parser import Parser
from fonduer.parser.models import Document
from fonduer.parser.preprocessors import HTMLDocPreprocessor

DB = "visualizer_test"
# Use 127.0.0.1 instead of localhost (#351)
CONN_STRING = f"postgresql://127.0.0.1:5432/{DB}"


def test_visualizer():
    from fonduer.utils.visualizer import Visualizer  # noqa

    """Unit test of visualizer using the md document.
    """
    session = Meta.init(CONN_STRING).Session()

    PARALLEL = 1
    max_docs = 1
    docs_path = "tests/data/html_simple/md.html"
    pdf_path = "tests/data/pdf_simple/md.pdf"

    # Preprocessor for the Docs
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    # Create an Parser and parse the md document
    corpus_parser = Parser(
        session, structural=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

    # Grab the md document
    doc = session.query(Document).order_by(Document.name).all()[0]
    assert doc.name == "md"

    organization_ngrams = MentionNgrams(n_max=1)

    Org = mention_subclass("Org")

    organization_matcher = OrganizationMatcher()

    mention_extractor = MentionExtractor(
        session, [Org], [organization_ngrams], [organization_matcher]
    )

    mention_extractor.apply([doc], parallelism=PARALLEL)

    Organization = candidate_subclass("Organization", [Org])

    candidate_extractor = CandidateExtractor(session, [Organization])

    candidate_extractor.apply([doc], split=0, parallelism=PARALLEL)

    cands = session.query(Organization).filter(Organization.split == 0).all()

    # Test visualizer
    pdf_path = "tests/data/pdf_simple"
    vis = Visualizer(pdf_path)
    vis.display_candidates([cands[0]])


def test_get_pdf_dim():
    from fonduer.utils.visualizer import get_pdf_dim  # noqa

    """Test get_pdf_dim on different pages"""
    assert get_pdf_dim("tests/data/pdf/BC546A_Series_B14-521026.pdf") == (729, 1032)
    assert get_pdf_dim("tests/data/pdf/BC546A_Series_B14-521026.pdf", page=1) == (
        729,
        1032,
    )
    assert get_pdf_dim("tests/data/pdf/BC546A_Series_B14-521026.pdf", page=6) == (
        612,
        792,
    )
