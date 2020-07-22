"""Fonduer data model's structural utils unit tests."""
import pytest

from fonduer.candidates.mentions import MentionNgrams
from fonduer.parser.models import Document
from fonduer.parser.parser import ParserUDF
from fonduer.utils.data_model_utils import common_ancestor, lowest_common_ancestor_depth


def get_parser_udf(
    structural=True,  # structural information
    blacklist=["style", "script"],  # ignore tag types, default: style, script
    flatten=["span", "br"],  # flatten tag types, default: span, br
    language="en",
    lingual=True,  # lingual information
    lingual_parser=None,
    strip=True,
    replacements=[("[\u2010\u2011\u2012\u2013\u2014\u2212]", "-")],
    tabular=True,  # tabular information
    visual=False,  # visual information
    vizlink=None,
    pdf_path=None,
):
    """Return an instance of ParserUDF."""
    parser_udf = ParserUDF(
        structural=structural,
        blacklist=blacklist,
        flatten=flatten,
        lingual=lingual,
        lingual_parser=lingual_parser,
        strip=strip,
        replacements=replacements,
        tabular=tabular,
        visual=visual,
        vizlink=vizlink,
        pdf_path=pdf_path,
        language=language,
    )
    return parser_udf


@pytest.fixture()
def doc_setup():
    """Set up a document."""
    parser_udf = get_parser_udf()

    doc = Document(id=1, name="test", stable_id="1::document:0:0")
    doc.text = """<html>
                    <body>
                        <h1>test1</h1>
                        <h2>test2</h2>
                        <div>
                            <h3>test3</h3>
                            <table>
                                <tr>
                                    <td>test4</td>
                                    <td>test5</td>
                                </tr>
                            </table>
                            <table>
                                <tr>
                                    <td>test6</td>
                                    <td>test7</td>
                                </tr>
                            </table>
                        </div>
                        <p>test8 test9</p>
                    </body>
                </html>"""
    doc = parser_udf.apply(doc)

    return doc


@pytest.mark.parametrize(
    "mention_ids, output_common_ancestor, output_lcad",
    [
        ([], ["", "html", "body"], 1),
        ([0, 1], ["", "html", "body"], 1),
        ([2, 3], ["", "html", "body", "div"], 1),
        ([3, 4], ["", "html", "body", "div", "table[1]", "tr"], 1),
        ([4, 5], ["", "html", "body", "div"], 3),
        ([5, 6], ["", "html", "body", "div", "table[2]", "tr"], 1),
        ([3, 5], ["", "html", "body", "div"], 3),
        ([7, 8], ["", "html", "body", "p"], 0),
    ],
)
def test_ancestors(doc_setup, mention_ids, output_common_ancestor, output_lcad):
    """Test if get_vert_ngrams works."""
    doc = doc_setup

    # Create 1-gram span mentions
    space = MentionNgrams(n_min=1, n_max=1)
    mentions = [tc for tc in space.apply(doc)]
    assert len(mentions) == len([word for sent in doc.sentences for word in sent.words])

    # Test mentions extraction
    assert mentions[0].sentence.text == "test1"
    assert mentions[1].sentence.text == "test2"
    assert mentions[2].sentence.text == "test3"
    assert mentions[3].sentence.text == "test4"
    assert mentions[4].sentence.text == "test5"
    assert mentions[5].sentence.text == "test6"
    assert mentions[6].sentence.text == "test7"
    assert mentions[7].sentence.text == "test8 test9"
    assert mentions[7].get_span() == "test8"
    assert mentions[8].get_span() == "test9"

    test_mentions = (
        [mentions[i] for i in mention_ids] if len(mention_ids) > 0 else mentions
    )

    # Test commont ancestor calculation
    overall_common_ancestor = common_ancestor(test_mentions)
    assert overall_common_ancestor == output_common_ancestor

    # Test lowest commont ancestor depth calculation
    overall_lowest_common_ancestor_depth = lowest_common_ancestor_depth(test_mentions)
    assert overall_lowest_common_ancestor_depth == output_lcad
