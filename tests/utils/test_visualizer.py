"""
These tests expect that postgres is installed and that a database named
visualizer_test has been created for the purpose of testing.

If you are testing locally, you will need to create this db.
"""
from fonduer.candidates import MentionNgrams
from fonduer.candidates.candidates import CandidateExtractorUDF
from fonduer.candidates.matchers import OrganizationMatcher
from fonduer.candidates.mentions import MentionExtractorUDF
from fonduer.candidates.models import candidate_subclass, mention_subclass
from tests.candidates.test_candidates import parse_doc


def test_visualizer():
    from fonduer.utils.visualizer import Visualizer  # noqa

    """Unit test of visualizer using the md document.
    """
    docs_path = "tests/data/html_simple/md.html"
    pdf_path = "tests/data/pdf_simple/md.pdf"

    # Grab the md document
    doc = parse_doc(docs_path, "md", pdf_path)
    assert doc.name == "md"

    organization_ngrams = MentionNgrams(n_max=1)

    Org = mention_subclass("Org")

    organization_matcher = OrganizationMatcher()

    mention_extractor_udf = MentionExtractorUDF(
        [Org], [organization_ngrams], [organization_matcher]
    )

    doc = mention_extractor_udf.apply(doc)

    Organization = candidate_subclass("Organization", [Org])

    candidate_extractor_udf = CandidateExtractorUDF(
        [Organization], None, False, False, True
    )

    doc = candidate_extractor_udf.apply(doc, split=0)

    cands = doc.organizations

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
