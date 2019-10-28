import pytest

from fonduer.candidates.matchers import Intersect, Inverse, RegexMatchSpan, Union
from fonduer.candidates.mentions import MentionNgrams
from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.parser.lingual_parser.spacy_parser import SpacyParser
from fonduer.parser.models import Document, Sentence


@pytest.fixture()
def doc_setup():
    doc = Document(id=1, name="test", stable_id="1::document:0:0")
    doc.text = "This is apple"
    lingual_parser = SpacyParser("en")
    for parts in lingual_parser.split_sentences(doc.text):
        parts["document"] = doc
        Sentence(**parts)
    return doc


def test_union(doc_setup):
    doc = doc_setup
    space = MentionNgrams(n_min=1, n_max=2)
    tc: TemporarySpanMention
    assert set(tc.get_span() for tc in space.apply(doc)) == {
        "This is",
        "is apple",
        "This",
        "is",
        "apple",
    }

    # Match any span that contains "apple"
    matcher0 = RegexMatchSpan(
        rgx=r"apple", search=True, full_match=True, longest_match_only=False
    )
    assert set(tc.get_span() for tc in matcher0.apply(space.apply(doc))) == {
        "is apple",
        "apple",
    }

    # Match any span that contains "this" (case insensitive)
    matcher1 = RegexMatchSpan(
        rgx=r"this", search=False, full_match=False, longest_match_only=False
    )
    assert set(tc.get_span() for tc in matcher1.apply(space.apply(doc))) == {
        "This is",
        "This",
    }

    matcher = Union(matcher0, matcher1, longest_match_only=False)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {
        "is apple",
        "apple",
        "This is",
        "This",
    }

    # longest_match_only of each matcher is ignored.
    matcher = Union(matcher0, matcher1, longest_match_only=True)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {
        "This is",
        "is apple",
    }


def test_intersect(doc_setup):
    doc = doc_setup
    space = MentionNgrams(n_min=1, n_max=3)
    tc: TemporarySpanMention

    # Match any span that contains "apple"
    matcher0 = RegexMatchSpan(
        rgx=r"apple", search=True, full_match=True, longest_match_only=False
    )
    assert set(tc.get_span() for tc in matcher0.apply(space.apply(doc))) == {
        "This is apple",
        "is apple",
        "apple",
    }

    # Match any span that contains "this" (case insensitive)
    matcher1 = RegexMatchSpan(
        rgx=r"this", search=False, full_match=False, longest_match_only=False
    )
    assert set(tc.get_span() for tc in matcher1.apply(space.apply(doc))) == {
        "This is apple",
        "This is",
        "This",
    }

    # Intersection of matcher0 and matcher1
    matcher = Intersect(matcher0, matcher1, longest_match_only=False)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {
        "This is apple"
    }

    # Intersection of matcher0 and matcher0
    matcher = Intersect(matcher0, matcher0, longest_match_only=False)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {
        "This is apple",
        "is apple",
        "apple",
    }

    # longest_match_only=True overrides that of child matchers.
    matcher = Intersect(matcher0, matcher0, longest_match_only=True)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {
        "This is apple"
    }


def test_inverse(doc_setup):
    doc = doc_setup
    space = MentionNgrams(n_min=1, n_max=2)
    tc: TemporarySpanMention
    assert set(tc.get_span() for tc in space.apply(doc)) == {
        "This is",
        "is apple",
        "This",
        "is",
        "apple",
    }

    # Match any span that contains "apple" with longest_match_only=False
    matcher0 = RegexMatchSpan(
        rgx=r"apple", search=True, full_match=True, longest_match_only=False
    )
    assert set(tc.get_span() for tc in matcher0.apply(space.apply(doc))) == {
        "is apple",
        "apple",
    }

    # Take an inverse
    matcher = Inverse(matcher0, longest_match_only=False)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {
        "This is",
        "This",
        "is",
    }

    # longest_match_only=True
    matcher = Inverse(matcher0, longest_match_only=True)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"This is"}

    # Match any span that contains "apple" with longest_match_only=True
    matcher0 = RegexMatchSpan(
        rgx=r"apple", search=True, full_match=True, longest_match_only=True
    )
    assert set(tc.get_span() for tc in matcher0.apply(space.apply(doc))) == {"is apple"}

    # longest_match_only=False on Inverse is in effect.
    matcher = Inverse(matcher0, longest_match_only=False)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {
        "This is",
        "This",
        "is",
    }

    # longest_match_only=True on Inverse is in effect.
    matcher = Inverse(matcher0, longest_match_only=True)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"This is"}

    # Check if Inverse raises an error when no child matcher is provided.
    with pytest.raises(ValueError):
        Inverse()

    # Check if Inverse raises an error when two child matchers are provided.
    with pytest.raises(ValueError):
        Inverse(matcher0, matcher0)
