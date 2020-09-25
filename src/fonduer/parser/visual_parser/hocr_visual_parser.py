"""Fonduer visual parser that parses visual information from hOCR."""
import itertools
from typing import Dict, Iterable, Iterator, List

import spacy
import spacy.gold
from packaging import version
from spacy.gold import align

from fonduer.parser.models import Sentence
from fonduer.parser.visual_parser.visual_parser import VisualParser


class HocrVisualParser(VisualParser):
    """Visual Parser for hOCR."""

    def __init__(self, sep: str = " "):
        """Initialize a visual parser.

        :param sep: word separator, defaults to " " (single space).
          This separator is used when joining words into a sentence.
          Use a single space (" ") for English and no space ("") for CJK languages.
        :raises ImportError: an error is raised when spaCy is not 2.2.2 or later.
        """
        if version.parse(spacy.__version__) < version.parse("2.2.2"):
            raise ImportError(
                f"You are using spaCy {spacy.__version__},"
                f"but it should be 2.2.2 or later."
            )
        spacy.gold.USE_NEW_ALIGN = True
        self.sep = sep

    def parse(
        self, document_name: str, sentences: Iterable[Sentence]
    ) -> Iterator[Sentence]:
        """Parse visual information embedded in sentence's html_attrs.

        :param document_name: the document name.
        :param sentences: sentences to be linked with visual information.
        :return: A generator of ``Sentence``.
        """

        def attrib_parse(html_attrs: List[str]) -> Dict[str, List[int]]:
            ret = {}
            for attr in html_attrs:
                key, values = attr.split("=")
                if key in ["left", "top", "right", "bottom", "ppageno", "cuts"]:
                    ret[key] = [int(x) for x in values.split()]
            return ret

        for _, group in itertools.groupby(sentences, key=lambda x: x.xpath):
            sents = list(group)

            # Get bbox from document
            attribs = attrib_parse(sents[0].html_attrs)
            lefts = attribs["left"]
            tops = attribs["top"]
            rights = attribs["right"]
            bottoms = attribs["bottom"]
            ppagenos = attribs["ppageno"]

            # Get a list of all tokens tokenized by spaCy.
            spacy_tokens = [word for sent in sents for word in sent.words]

            text = self.sep.join([sent.text for sent in sents])

            # Get a list of all tokens represented by ocrx_word in hOCR
            start = 0
            hocr_tokens = []
            for end in attribs["cuts"]:
                hocr_tokens.append(text[start:end])
                start = end

            # gold.align assumes that both tokenizations add up to the same string.
            cost, h2s, s2h, h2s_multi, s2h_multi = align(hocr_tokens, spacy_tokens)

            ptr = 0  # word pointer
            for sent in sents:
                sent.left = []
                sent.top = []
                sent.right = []
                sent.bottom = []
                sent.page = []
                for i, word in enumerate(sent.words):
                    # One-to-one mapping is NOT available
                    if s2h[ptr + i] == -1:
                        if ptr + i in s2h_multi:  # One spacy token-to-multi hOCR words
                            left = lefts[s2h_multi[ptr + i]]
                            top = tops[s2h_multi[ptr + i]]
                            right = rights[s2h_multi[ptr + i]]
                            bottom = bottoms[s2h_multi[ptr + i]]
                            ppageno = ppagenos[s2h_multi[ptr + i]]
                        else:
                            h2s_multi_idx = [
                                k for k, v in h2s_multi.items() if ptr + i == v
                            ]
                            if h2s_multi_idx:  # One hOCR word-to-multi spacy tokens
                                start = h2s_multi_idx[0]
                                end = h2s_multi_idx[-1] + 1
                                # calculate a bbox that can include all
                                left = min(lefts[start:end])
                                top = min(tops[start:end])
                                right = max(rights[start:end])
                                bottom = max(bottoms[start:end])
                                ppageno = ppagenos[start]
                            else:
                                raise RuntimeError(
                                    "Tokens are not aligned!",
                                    f"hocr tokens: {hocr_tokens}",
                                    f"spacy tokens: {spacy_tokens}",
                                )
                    # One-to-one mapping is available
                    else:
                        left = lefts[s2h[ptr + i]]
                        top = tops[s2h[ptr + i]]
                        right = rights[s2h[ptr + i]]
                        bottom = bottoms[s2h[ptr + i]]
                        ppageno = ppagenos[s2h[ptr + i]]
                    sent.left.append(left)
                    sent.top.append(top)
                    sent.right.append(right)
                    sent.bottom.append(bottom)
                    sent.page.append(ppageno + 1)  # 1-based in Fonduer
                ptr += len(sent.words)
                yield sent

    def is_parsable(self, filename: str) -> bool:
        """Whether bbox is embdded in sentence's html_attrs.

        :param document_name: The path to the PDF document.
        """
        return True
