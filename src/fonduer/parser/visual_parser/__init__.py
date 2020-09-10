"""Fonduer's visual parser module."""
from fonduer.parser.visual_parser.hocr_visual_parser import HocrVisualParser
from fonduer.parser.visual_parser.pdf_visual_parser import PdfVisualParser
from fonduer.parser.visual_parser.visual_parser import VisualParser

__all__ = ["VisualParser", "PdfVisualParser", "HocrVisualParser"]
