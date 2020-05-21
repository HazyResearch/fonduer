import pytest

from fonduer.candidates.mentions import MentionNgrams
from fonduer.parser.lingual_parser.spacy_parser import SpacyParser
from fonduer.parser.models import Document, Sentence
from fonduer.utils.data_model_utils.visual import get_horz_ngrams, get_vert_ngrams


@pytest.fixture()
def doc_setup():
    doc = Document(id=1, name="test", stable_id="1::document:0:0")
    doc.text = "This is apple. That is orange. Where is banaba? I like Apple."
    lingual_parser = SpacyParser("en")
    # Split sentences
    for parts in lingual_parser.split_sentences(doc.text):
        parts["document"] = doc
        Sentence(**parts)
    # Enrich sentences
    for _ in lingual_parser.enrich_sentences_with_NLP(doc.sentences):
        pass

    # Pick one sentence and add visual information
    # so that all the words get aligned horizontally.
    sentence: Sentence = doc.sentences[0]
    sentence.page = [1, 1, 1, 1]
    sentence.top = [0, 0, 0, 0]
    sentence.bottom = [10, 10, 10, 10]
    sentence.left = [0, 10, 20, 30]
    sentence.right = [10, 20, 30, 40]

    # Assume the 2nd sentence is horizontally aligned with 1st.
    sentence: Sentence = doc.sentences[1]
    sentence.page = [1, 1, 1, 1]
    sentence.top = [0, 0, 0, 0]
    sentence.bottom = [10, 10, 10, 10]
    sentence.left = [40, 50, 60, 70]
    sentence.right = [50, 60, 70, 80]

    # Assume the 3rd sentence is vertically aligned with 1st.
    sentence: Sentence = doc.sentences[2]
    sentence.page = [1, 1, 1, 1]
    sentence.top = [10, 10, 10, 10]
    sentence.bottom = [20, 20, 20, 20]
    sentence.left = [0, 10, 20, 30]
    sentence.right = [10, 20, 30, 40]

    # Assume the 4th sentence is in 2nd page.
    sentence: Sentence = doc.sentences[3]
    sentence.page = [2, 2, 2, 2]
    sentence.top = [0, 0, 0, 0]
    sentence.bottom = [10, 10, 10, 10]
    sentence.left = [0, 10, 20, 30]
    sentence.right = [10, 20, 30, 40]

    return doc


def test_get_vert_ngrams(doc_setup):
    """Test if get_vert_ngrams works."""
    doc = doc_setup
    sentence: Sentence = doc.sentences[0]

    # Assert this sentence is visual.
    assert sentence.is_visual()

    # Assert this sentence is not tabular.
    assert not sentence.is_tabular()

    # Create 1-gram span mentions
    space = MentionNgrams(n_min=1, n_max=1)
    mentions = [tc for tc in space.apply(doc)]
    assert len(mentions) == len([word for sent in doc.sentences for word in sent.words])

    # Pick "apple" span mention.
    mention = mentions[2]
    assert mention.get_span() == "apple"
    # from_sentence=True (ie ngrams from all aligned Sentences but its Sentence)
    ngrams = list(get_vert_ngrams(mention))
    assert ngrams == ["where", "is", "banaba", "?"]


def test_get_horz_ngrams(doc_setup):
    """Test if get_horz_ngrams works."""
    doc = doc_setup
    sentence: Sentence = doc.sentences[0]

    # Assert this sentence is visual.
    assert sentence.is_visual()

    # Assert this sentence is not tabular.
    assert not sentence.is_tabular()

    # Create 1-gram span mentions
    space = MentionNgrams(n_min=1, n_max=1)
    mentions = [tc for tc in space.apply(doc)]
    assert len(mentions) == len([word for sent in doc.sentences for word in sent.words])

    # Pick "apple" span mention.
    mention = mentions[2]
    assert mention.get_span() == "apple"
    # from_sentence=True (ie ngrams from all aligned Sentences but its Sentence)
    ngrams = list(get_horz_ngrams(mention))
    assert ngrams == ["that", "is", "orange", "."]

    # Check the from_sentence=False (ie all aligned ngrams but itself)
    assert mention.get_span() == "apple"
    ngrams = list(get_horz_ngrams(mention, from_sentence=False))
    assert ngrams == ["this", "is", ".", "that", "is", "orange", "."]

    # Check attrib="lemmas"
    ngrams = list(get_horz_ngrams(mention, attrib="lemmas"))
    assert ngrams == ["that", "be", "orange", "."]

    # Check attrib="pos_tags"
    ngrams = list(get_horz_ngrams(mention, attrib="pos_tags"))
    assert ngrams == ["dt", "vbz", "jj", "."]

    # Check lower option
    ngrams = list(get_horz_ngrams(mention, lower=False, from_sentence=False))
    assert ngrams == ["This", "is", ".", "That", "is", "orange", "."]

    # Pick "This" span mention.
    mention = mentions[0]
    assert mention.get_span() == "This"
    ngrams = list(get_horz_ngrams(mention, from_sentence=False))
    assert ngrams == ["is", "apple", ".", "that", "is", "orange", "."]

    # Check n_max=2
    ngrams = list(get_horz_ngrams(mention, n_max=2, from_sentence=False))
    assert ngrams == [
        "is apple",
        "apple.",
        "is",
        "apple",
        ".",
        "that is",
        "is orange",
        "orange.",
        "that",
        "is",
        "orange",
        ".",
    ]


def test_get_ngrams_that_match_in_string(doc_setup):
    """Test if ngrams can be obtained even if they match mention's span in string."""
    doc = doc_setup
    sentence: Sentence = doc.sentences[0]
    # Assert this sentence is visual.
    assert sentence.is_visual()
    # Assert this sentence is not tabular.
    assert not sentence.is_tabular()

    # Create 1-gram span mentions
    space = MentionNgrams(n_min=1, n_max=1)
    mentions = [tc for tc in space.apply(doc)]
    assert len(mentions) == len([word for sent in doc.sentences for word in sent.words])

    # Pick "is" from the apple sentence that matches "is" in the orange sentence.
    mention = mentions[1]
    assert mention.get_span() == "is"
    # Check if the "is" in the orange sentence can be obtained.
    ngrams = list(get_horz_ngrams(mention, from_sentence=False))
    assert "is" in ngrams
