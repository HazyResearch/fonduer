"""Fonduer unit tests for matchers."""
from unittest.mock import Mock

import pytest
from nltk.stem.porter import PorterStemmer

from fonduer.candidates.matchers import (
    Concat,
    DateMatcher,
    DictionaryMatch,
    Intersect,
    Inverse,
    LambdaFunctionFigureMatcher,
    LambdaFunctionMatcher,
    LocationMatcher,
    MiscMatcher,
    NumberMatcher,
    OrganizationMatcher,
    PersonMatcher,
    RegexMatchEach,
    RegexMatchSpan,
    Union,
)
from fonduer.candidates.mentions import MentionFigures, MentionNgrams
from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.parser.lingual_parser.spacy_parser import SpacyParser
from fonduer.parser.models import Document, Figure, Sentence


@pytest.fixture()
def doc_setup():
    """Set up document."""
    doc = Document(id=1, name="test", stable_id="1::document:0:0")
    doc.text = "This is apple"
    lingual_parser = SpacyParser("en")
    for parts in lingual_parser.split_sentences(doc.text):
        parts["document"] = doc
        Sentence(**parts)
    return doc


def test_union(doc_setup):
    """Test union matcher."""
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

    # Unsupported option should raise an exception
    with pytest.raises(Exception):
        Union(matcher0, matcher1, long_match_only=False)


def test_intersect(doc_setup):
    """Test intersect matcher."""
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
    """Test inverse matcher."""
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


def test_cancat(doc_setup):
    """Test Concat matcher."""
    doc = doc_setup
    space = MentionNgrams(n_min=1, n_max=2)

    # Match any span that contains "this"
    matcher0 = RegexMatchSpan(
        rgx=r"this", search=False, full_match=False, longest_match_only=False
    )
    # Match any span that contains "is"
    matcher1 = RegexMatchSpan(
        rgx=r"is", search=False, full_match=False, longest_match_only=False
    )
    matcher = Concat(matcher0, matcher1)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"This is"}

    # Test if matcher raises an error when _f is given non-TemporarySpanMention
    with pytest.raises(ValueError):
        list(matcher.apply(doc.sentences[0].words))

    # Test if an error is raised when the number of child matchers is not 2.
    matcher = Concat(matcher0)
    with pytest.raises(ValueError):
        list(matcher.apply(space.apply(doc)))

    # Test with left_required=False
    matcher = Concat(matcher0, matcher1, left_required=False)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {
        "This is",
        "is apple",
    }

    # Test with right_required=False
    matcher = Concat(matcher0, matcher1, right_required=False)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"This is"}

    # Test with permutations=False
    matcher = Concat(matcher1, matcher0, permutations=False)
    assert set(matcher.apply(space.apply(doc))) == set()
    # Test with permutations=True
    matcher = Concat(matcher1, matcher0, permutations=True)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"This is"}

    # TODO: Add a test for ignore_sep=False


def test_dictionary_match(doc_setup):
    """Test DictionaryMatch matcher."""
    doc = doc_setup
    space = MentionNgrams(n_min=1, n_max=1)

    # Test with a list of str
    matcher = DictionaryMatch(d=["this"])
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"This"}

    # Test without a dictionary
    with pytest.raises(Exception):
        DictionaryMatch()

    # TODO: test with plural words
    matcher = DictionaryMatch(d=["is"], stemmer=PorterStemmer())
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"is"}

    # Test if matcher raises an error when _f is given non-TemporarySpanMention
    matcher = DictionaryMatch(d=["this"])
    with pytest.raises(ValueError):
        list(matcher.apply(doc.sentences[0].words))


def test_do_not_use_stemmer_when_UnicodeDecodeError():
    """Test DictionaryMatch when stemmer causes UnicodeDecodeError."""
    stemmer = PorterStemmer()
    matcher = DictionaryMatch(d=["is"], stemmer=stemmer)
    # _stem(w) should return a word stem.
    assert matcher._stem("caresses") == "caress"

    stemmer.stem = Mock(
        side_effect=UnicodeDecodeError("dummycodec", b"\x00\x00", 1, 2, "Dummy  !")
    )
    matcher = DictionaryMatch(d=["is"], stemmer=stemmer)
    # _stem(w) should return w as stemmer.stem raises UnicodeDecodeError.
    assert matcher._stem("caresses") == "caresses"


def test_lambda_function_matcher(doc_setup):
    """Test DictionaryMatch matcher."""
    doc = doc_setup
    space = MentionNgrams(n_min=1, n_max=1)

    # Test with a lambda function
    matcher = LambdaFunctionMatcher(func=lambda x: True)
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {
        "This",
        "is",
        "apple",
    }

    # Test if matcher raises an error when _f is given non-TemporarySpanMention
    with pytest.raises(ValueError):
        list(matcher.apply(doc.sentences[0].words))

    # Test if an error raised when a func is not provided.
    with pytest.raises(Exception):
        LambdaFunctionMatcher()


def test_regex_match(doc_setup):
    """Test RegexMatch matcher."""
    doc = doc_setup
    space = MentionNgrams(n_min=1, n_max=2)

    # a wrong option name should raise an excetiopn
    with pytest.raises(Exception):
        RegexMatchSpan(regex=r"apple")

    # Test if matcher raises an error when _f is given non-TemporarySpanMention
    matcher = RegexMatchSpan(rgx=r"apple")
    with pytest.raises(ValueError):
        list(matcher.apply(doc.sentences[0].words))

    matcher = RegexMatchEach(rgx=r"apple")
    with pytest.raises(ValueError):
        list(matcher.apply(doc.sentences[0].words))

    # Test if RegexMatchEach works as expected.
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"apple"}

    # Test ignore_case option
    matcher = RegexMatchEach(rgx=r"Apple", ignore_case=False)
    assert list(matcher.apply(space.apply(doc))) == []

    # Test sep option
    matcher = RegexMatchSpan(rgx=r"isapple", sep=" ")
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"is apple"}


def test_ner_matchers():
    """Test different ner type matchers."""
    # Set up a document
    doc = Document(id=1, name="test", stable_id="1::document:0:0")
    doc.text = " ".join(
        [
            "Tim Cook was born in USA in 1960.",
            "He is the CEO of Apple.",
            "He sold 100 million of iPhone.",
        ]
    )
    lingual_parser = SpacyParser("en")
    for parts in lingual_parser.split_sentences(doc.text):
        parts["document"] = doc
        Sentence(**parts)
    # Manually attach ner_tags as the result from spacy may fluctuate.
    doc.sentences[0].ner_tags = [
        "PERSON",
        "PERSON",
        "O",
        "O",
        "O",
        "GPE",
        "O",
        "DATE",
        "O",
    ]
    doc.sentences[1].ner_tags = ["O", "O", "O", "O", "O", "ORG", "O"]
    doc.sentences[2].ner_tags = ["O", "O", "CARDINAL", "CARDINAL", "O", "MISC", "O"]

    # the length of words and that of ner_tags should match.
    assert len(doc.sentences[0].words) == len(doc.sentences[0].ner_tags)
    assert len(doc.sentences[1].words) == len(doc.sentences[1].ner_tags)

    space = MentionNgrams(n_min=1, n_max=2)

    # Test if PersonMatcher works as expected
    matcher = PersonMatcher()
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"Tim Cook"}

    # Test if LocationMatcher works as expected
    matcher = LocationMatcher()
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"USA"}

    # Test if DateMatcher works as expected
    matcher = DateMatcher()
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"1960"}

    # Test if OrganizationMatcher works as expected
    matcher = OrganizationMatcher()
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"Apple"}

    # Test if NumberMatcher works as expected
    matcher = NumberMatcher()
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {
        "100 million"
    }

    # Test if MiscMatcher works as expected
    matcher = MiscMatcher()
    assert set(tc.get_span() for tc in matcher.apply(space.apply(doc))) == {"iPhone"}


def test_figure_matcher(doc_setup):
    """Test matchers for figures."""
    doc = doc_setup
    # Create two dummy figures
    Figure(id=2, document=doc)
    Figure(id=3, document=doc)
    assert len(doc.figures) == 2

    space = MentionFigures()
    assert len(list(space.apply(doc))) == 2

    # Set up a matcher that matches figures with id==2.
    matcher = LambdaFunctionFigureMatcher(
        func=lambda tf: True if tf.figure.id == 2 else False
    )

    # Test if matcher only matches the first figure.
    assert len(list(matcher.apply(space.apply(doc)))) == 1
    assert set(tf.figure.id for tf in matcher.apply(space.apply(doc))) == {2}

    # The keyword arg should be "func"
    with pytest.raises(Exception):
        LambdaFunctionFigureMatcher(
            function=lambda tf: True if tf.figure.id == 2 else False
        )

    # LambdaFunctionFigureMatcher only supports TemporaryFigureMention.
    space = MentionNgrams(n_min=1, n_max=2)
    with pytest.raises(ValueError):
        list(matcher.apply(space.apply(doc)))
