"""Fonduer's lingual parser module."""
from fonduer.parser.lingual_parser.lingual_parser import LingualParser
from fonduer.parser.lingual_parser.simple_parser import SimpleParser
from fonduer.parser.lingual_parser.spacy_parser import SpacyParser

__all__ = ["LingualParser", "SpacyParser", "SimpleParser"]
