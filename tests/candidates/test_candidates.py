import logging
import pickle
from typing import Optional

import pytest

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
from fonduer.candidates.candidates import CandidateExtractorUDF
from fonduer.candidates.matchers import (
    DoNothingMatcher,
    LambdaFunctionFigureMatcher,
    LambdaFunctionMatcher,
    PersonMatcher,
)
from fonduer.candidates.mentions import MentionExtractorUDF, Ngrams
from fonduer.candidates.models import candidate_subclass, mention_subclass
from fonduer.parser.models import Sentence
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.utils.data_model_utils import get_col_ngrams, get_row_ngrams
from tests.parser.test_parser import get_parser_udf
from tests.shared.hardware_matchers import part_matcher, temp_matcher, volt_matcher
from tests.shared.hardware_spaces import (
    MentionNgramsPart,
    MentionNgramsTemp,
    MentionNgramsVolt,
)
from tests.shared.hardware_throttlers import temp_throttler, volt_throttler

logger = logging.getLogger(__name__)


def parse_doc(docs_path: str, file_name: str, pdf_path: Optional[str] = None):
    max_docs = 1

    logger.info("Parsing...")
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    doc = next(doc_preprocessor._parse_file(docs_path, file_name))

    # Create an Parser and parse the md document
    parser_udf = get_parser_udf(
        structural=True,
        tabular=True,
        lingual=True,
        visual=True if pdf_path else False,
        pdf_path=pdf_path,
        language="en",
    )
    doc = parser_udf.apply(doc)
    return doc


def test_ngram_split():
    """Test ngram split."""
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


def test_span_char_start_and_char_end():
    """Test chart_start and char_end of TemporarySpan that comes from Ngrams.apply."""
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


def test_cand_gen():
    """Test extracting candidates from mentions from documents."""

    def do_nothing_matcher(fig):
        return True

    docs_path = "tests/data/html/112823.html"
    pdf_path = "tests/data/pdf/112823.pdf"
    doc = parse_doc(docs_path, "112823", pdf_path)

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
        MentionExtractor(
            "dummy",
            [Part, Temp, Volt],
            [part_ngrams, volt_ngrams],  # Fail, mismatched arity
            [part_matcher, temp_matcher, volt_matcher],
        )
    with pytest.raises(ValueError):
        MentionExtractor(
            "dummy",
            [Part, Temp, Volt],
            [part_ngrams, temp_matcher, volt_ngrams],
            [part_matcher, temp_matcher],  # Fail, mismatched arity
        )

    mention_extractor_udf = MentionExtractorUDF(
        [Part, Temp, Volt, Fig],
        [part_ngrams, temp_ngrams, volt_ngrams, figs],
        [part_matcher, temp_matcher, volt_matcher, fig_matcher],
    )
    doc = mention_extractor_udf.apply(doc)

    assert len(doc.parts) == 70
    assert len(doc.volts) == 33
    assert len(doc.temps) == 23
    assert len(doc.figs) == 31
    part = doc.parts[0]
    volt = doc.volts[0]
    temp = doc.temps[0]
    logger.info(f"Part: {part.context}")
    logger.info(f"Volt: {volt.context}")
    logger.info(f"Temp: {temp.context}")

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])
    PartVolt = candidate_subclass("PartVolt", [Part, Volt])

    with pytest.raises(ValueError):
        CandidateExtractor(
            "dummy",
            [PartTemp, PartVolt],
            throttlers=[
                temp_throttler,
                volt_throttler,
                volt_throttler,
            ],  # Fail, mismatched arity
        )

    with pytest.raises(ValueError):
        CandidateExtractor(
            "dummy",
            [PartTemp],  # Fail, mismatched arity
            throttlers=[temp_throttler, volt_throttler],
        )

    # Test that no throttler in candidate extractor
    candidate_extractor_udf = CandidateExtractorUDF(
        [PartTemp, PartVolt], [None, None], False, False, True  # Pass, no throttler
    )

    doc = candidate_extractor_udf.apply(doc, split=0)

    assert len(doc.part_temps) == 1610
    assert len(doc.part_volts) == 2310

    # Clear
    doc.part_temps = []
    doc.part_volts = []

    # Test with None in throttlers in candidate extractor
    candidate_extractor_udf = CandidateExtractorUDF(
        [PartTemp, PartVolt], [temp_throttler, None], False, False, True
    )

    doc = candidate_extractor_udf.apply(doc, split=0)
    assert len(doc.part_temps) == 1432
    assert len(doc.part_volts) == 2310

    # Clear
    doc.part_temps = []
    doc.part_volts = []

    candidate_extractor_udf = CandidateExtractorUDF(
        [PartTemp, PartVolt], [temp_throttler, volt_throttler], False, False, True
    )

    doc = candidate_extractor_udf.apply(doc, split=0)

    assert len(doc.part_temps) == 1432
    assert len(doc.part_volts) == 1993
    assert len(doc.parts) == 70
    assert len(doc.volts) == 33
    assert len(doc.temps) == 23


def test_ngrams():
    """Test ngram limits in mention extraction"""
    file_name = "lincoln_short"
    docs_path = f"tests/data/pure_html/{file_name}.html"
    doc = parse_doc(docs_path, file_name)

    # Mention Extraction
    Person = mention_subclass("Person")
    person_ngrams = MentionNgrams(n_max=3)
    person_matcher = PersonMatcher()

    mention_extractor_udf = MentionExtractorUDF(
        [Person], [person_ngrams], [person_matcher]
    )
    doc = mention_extractor_udf.apply(doc)

    assert len(doc.persons) == 118
    mentions = doc.persons
    assert len([x for x in mentions if x.context.get_num_words() == 1]) == 49
    assert len([x for x in mentions if x.context.get_num_words() > 3]) == 0

    # Test for unigram exclusion
    for mention in doc.persons[:]:
        doc.persons.remove(mention)
    assert len(doc.persons) == 0

    person_ngrams = MentionNgrams(n_min=2, n_max=3)
    mention_extractor_udf = MentionExtractorUDF(
        [Person], [person_ngrams], [person_matcher]
    )
    doc = mention_extractor_udf.apply(doc)
    assert len(doc.persons) == 69
    mentions = doc.persons
    assert len([x for x in mentions if x.context.get_num_words() == 1]) == 0
    assert len([x for x in mentions if x.context.get_num_words() > 3]) == 0


def test_row_col_ngram_extraction():
    """Test whether row/column ngrams list is empty, if mention is not in a table."""
    file_name = "lincoln_short"
    docs_path = f"tests/data/pure_html/{file_name}.html"
    doc = parse_doc(docs_path, file_name)

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
    mention_extractor_udf = MentionExtractorUDF(
        [Place], [place_ngrams], [birthplace_matcher]
    )

    doc = mention_extractor_udf.apply(doc)


def test_mention_longest_match():
    """Test longest match filtering in mention extraction."""
    file_name = "lincoln_short"
    docs_path = f"tests/data/pure_html/{file_name}.html"
    doc = parse_doc(docs_path, file_name)

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
    mention_extractor_udf = MentionExtractorUDF(
        [Name, Place],
        [name_ngrams, place_ngrams],
        [PersonMatcher(), birthplace_matcher],
    )
    doc = mention_extractor_udf.apply(doc)
    mentions = doc.places
    mention_spans = [x.context.get_span() for x in mentions]
    assert "Sinking Spring Farm" in mention_spans
    assert "Farm" in mention_spans
    assert len(mention_spans) == 23

    # Clear manually
    for mention in doc.places[:]:
        doc.places.remove(mention)

    birthplace_matcher = LambdaFunctionMatcher(
        func=is_birthplace_table_row, longest_match_only=True
    )
    mention_extractor_udf = MentionExtractorUDF(
        [Name, Place],
        [name_ngrams, place_ngrams],
        [PersonMatcher(), birthplace_matcher],
    )
    doc = mention_extractor_udf.apply(doc)
    mentions = doc.places
    mention_spans = [x.context.get_span() for x in mentions]
    assert "Sinking Spring Farm" in mention_spans
    assert "Farm" not in mention_spans
    assert len(mention_spans) == 4


def test_multimodal_cand():
    """Test multimodal candidate generation"""
    file_name = "radiology"
    docs_path = f"tests/data/pure_html/{file_name}.html"
    doc = parse_doc(docs_path, file_name)

    assert len(doc.sentences) == 35

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

    mention_extractor_udf = MentionExtractorUDF(ms, m, matchers)

    doc = mention_extractor_udf.apply(doc)

    assert len(doc.m_docs) == 1
    assert len(doc.m_caps) == 2
    assert len(doc.m_secs) == 5
    assert len(doc.m_tabs) == 2
    assert len(doc.m_figs) == 2
    assert len(doc.m_paras) == 30
    assert len(doc.m_sents) == 35
    assert len(doc.m_cells) == 21

    # Candidate Extraction
    cs_doc = candidate_subclass("cs_doc", [ms_doc])
    cs_sec = candidate_subclass("cs_sec", [ms_sec])
    cs_tab = candidate_subclass("cs_tab", [ms_tab])
    cs_fig = candidate_subclass("cs_fig", [ms_fig])
    cs_cell = candidate_subclass("cs_cell", [ms_cell])
    cs_para = candidate_subclass("cs_para", [ms_para])
    cs_cap = candidate_subclass("cs_cap", [ms_cap])
    cs_sent = candidate_subclass("cs_sent", [ms_sent])

    candidate_extractor_udf = CandidateExtractorUDF(
        [cs_doc, cs_sec, cs_tab, cs_fig, cs_cell, cs_para, cs_cap, cs_sent],
        [None, None, None, None, None, None, None, None],
        False,
        False,
        True,
    )

    doc = candidate_extractor_udf.apply(doc, split=0)

    assert len(doc.cs_docs) == 1
    assert len(doc.cs_caps) == 2
    assert len(doc.cs_secs) == 5
    assert len(doc.cs_tabs) == 2
    assert len(doc.cs_figs) == 2
    assert len(doc.cs_paras) == 30
    assert len(doc.cs_sents) == 35
    assert len(doc.cs_cells) == 21


def test_pickle_subclasses():
    """Test if it is possible to pickle mention/candidate subclasses and their objects.
    """
    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])

    logger.info(f"Test if mention/candidate subclasses are picklable")
    pickle.loads(pickle.dumps(Part))
    pickle.loads(pickle.dumps(Temp))
    pickle.loads(pickle.dumps(PartTemp))

    logger.info(f"Test if their objects are pickable")
    part = Part()
    temp = Temp()
    parttemp = PartTemp()
    pickle.loads(pickle.dumps(part))
    pickle.loads(pickle.dumps(temp))
    pickle.loads(pickle.dumps(parttemp))
