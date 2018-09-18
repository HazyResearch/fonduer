import numpy as np
import torch


class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols

    :param starting_symbol: Starting index of symbol.
    :type starting_symbol: int
    :param unknown_symbol: Index of unknown symbol.
    :type unknown_symbol: int
    """

    def __init__(self, starting_symbol=2, unknown_symbol=1):
        self.s = starting_symbol
        self.unknown = unknown_symbol
        self.d = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, self.unknown)

    def lookup_strict(self, w):
        return self.d.get(w)

    def len(self):
        return self.s

    def reverse(self):
        return {v: k for k, v in self.d.iteritems()}


def mention_to_tokens(mention, token_type="words", lowercase=False):
    """
    Extract tokens from the mention

    :param mention: mention object.
    :param token_type: token type that wants to extract.
    :type token_type: str
    :param lowercase: use lowercase or not.
    :type lowercase: bool
    :return: The token list.
    :rtype: list
    """

    tokens = mention.span.sentence.__dict__[token_type]
    return [w.lower() if lowercase else w for w in tokens]


def mark(l, h, idx):
    """
    Produce markers based on argument positions

    :param l: sentence position of first word in argument.
    :type l: int
    :param h: sentence position of last word in argument.
    :type h: int
    :param idx: argument index (1 or 2).
    :type idx: int
    :return: markers.
    :rtype: list of markers
    """
    return [(l, "{}{}".format("~~[[", idx)), (h + 1, "{}{}".format(idx, "]]~~"))]


def mark_sentence(s, args):
    """Insert markers around relation arguments in word sequence

    :param s: list of tokens in sentence.
    :type s: list
    :param args: list of triples (l, h, idx) as per @_mark(...) corresponding
               to relation arguments
    :type args: list
    :return: The marked sentence.
    :rtype: list

    Example: Then Barack married Michelle.
         ->  Then ~~[[1 Barack 1]]~~ married ~~[[2 Michelle 2]]~~.
    """
    marks = sorted([y for m in args for y in mark(*m)], reverse=True)
    x = list(s)
    for k, v in marks:
        x.insert(k, v)
    return x


def pad_batch(batch, max_len=0, type="int"):
    """Pad the batch into matrix

    :param batch: The data for padding.
    :type batch: list of word index sequences
    :param max_len: Max length of sequence of padding.
    :type max_len: int
    :param type: mask value type.
    :type type: str
    :return: The padded matrix and correspoing mask matrix.
    :rtype: pair of torch.Tensors with shape (batch_size, max_sent_len)
    """
    batch_size = len(batch)
    max_sent_len = int(np.max([len(x) for x in batch]))
    if max_len > 0 and max_len < max_sent_len:
        max_sent_len = max_len
    if type == "float":
        idx_matrix = np.zeros((batch_size, max_sent_len), dtype=np.float32)
    else:
        idx_matrix = np.zeros((batch_size, max_sent_len), dtype=np.int)

    for idx1, i in enumerate(batch):
        for idx2, j in enumerate(i):
            if idx2 >= max_sent_len:
                break
            idx_matrix[idx1, idx2] = j
    idx_matrix = torch.tensor(idx_matrix)
    mask_matrix = torch.tensor(torch.eq(idx_matrix.data, 0))
    return idx_matrix, mask_matrix
