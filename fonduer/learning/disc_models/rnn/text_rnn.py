import numpy as np

from fonduer.learning.disc_models.rnn.rnn_base import RNNBase
from fonduer.learning.disc_models.rnn.utils import SymbolTable


class TextRNN(RNNBase):
    """TextRNN for strings of text."""

    def _preprocess_data(self, candidates, extend=False):
        """Convert candidate sentences to lookup sequences

        :param candidates: candidates to process
        :param extend: extend symbol table for tokens (train), or lookup (test)?
        """
        if not hasattr(self, "word_dict"):
            self.word_dict = SymbolTable()
        data, ends = [], []
        for candidate in candidates:
            toks = candidate.get_contexts()[0].text.split()
            # Either extend word table or retrieve from it
            f = self.word_dict.get if extend else self.word_dict.lookup
            data.append(np.array(list(map(f, toks))))
            ends.append(len(toks))
        return data, ends
