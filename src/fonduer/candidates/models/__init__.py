"""Fonduer's candidate model module."""
from fonduer.candidates.models.candidate import Candidate, candidate_subclass
from fonduer.candidates.models.caption_mention import CaptionMention
from fonduer.candidates.models.cell_mention import CellMention
from fonduer.candidates.models.document_mention import DocumentMention
from fonduer.candidates.models.figure_mention import FigureMention
from fonduer.candidates.models.implicit_span_mention import ImplicitSpanMention
from fonduer.candidates.models.mention import Mention, mention_subclass
from fonduer.candidates.models.paragraph_mention import ParagraphMention
from fonduer.candidates.models.section_mention import SectionMention
from fonduer.candidates.models.span_mention import SpanMention
from fonduer.candidates.models.table_mention import TableMention

__all__ = [
    "Candidate",
    "CaptionMention",
    "CellMention",
    "DocumentMention",
    "FigureMention",
    "ImplicitSpanMention",
    "Mention",
    "ParagraphMention",
    "SectionMention",
    "SpanMention",
    "TableMention",
    "candidate_subclass",
    "mention_subclass",
]
