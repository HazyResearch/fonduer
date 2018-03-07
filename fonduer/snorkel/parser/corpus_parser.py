from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from fonduer.snorkel.models import Candidate, Context, Sentence
from fonduer.snorkel.udf import UDF, UDFRunner


class CorpusParser(UDFRunner):
    def __init__(self, parser=None, fn=None):
        self.parser = parser
        super(CorpusParser, self).__init__(
            CorpusParserUDF, parser=self.parser, fn=fn)

    def clear(self, session, **kwargs):
        session.query(Context).delete()
        # We cannot cascade up from child contexts to parent Candidates,
        # so we delete all Candidates too
        session.query(Candidate).delete()


class CorpusParserUDF(UDF):
    def __init__(self, parser, fn, **kwargs):
        super(CorpusParserUDF, self).__init__(**kwargs)
        self.parser = parser
        self.req_handler = parser.connect()
        self.fn = fn

    def apply(self, x, **kwargs):
        """Given a Document object and its raw text, parse into Sentences"""
        doc, text = x
        for parts in self.req_handler.parse(doc, text):
            parts = self.fn(parts) if self.fn is not None else parts
            yield Sentence(**parts)
