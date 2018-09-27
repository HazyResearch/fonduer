import logging
import os
import re

# Travis will not import the PorterStemmer
if "CI" not in os.environ:
    try:
        from nltk.stem.porter import PorterStemmer
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("nltk not installed- some default functionality may be absent.")

WORDS = "words"


class _Matcher(object):
    """
    Applies a function ``f : m -> {True,False}`` to a generator of mentions,
    returning only mentions *m* s.t. *f(m) == True*,
    where f can be compositionally defined.
    """

    def __init__(self, *children, **opts):
        self.children = children
        self.opts = opts
        self.longest_match_only = self.opts.get("longest_match_only", True)
        self.init()
        self._check_opts()

    def init(self):
        pass

    def _check_opts(self):
        """
        Checks for unsupported opts, throws error if found
        NOTE: Must be called _after_ init()
        """
        for opt in self.opts.keys():
            if opt not in self.__dict__:
                raise Exception("Unsupported option: %s" % opt)

    def _f(self, m):
        """The internal (non-composed) version of filter function f"""
        return True

    def f(self, m):
        """
        The recursively composed version of filter function f.
        By default, returns logical **conjunction** of operator and single
        child operator
        """
        if len(self.children) == 0:
            return self._f(m)
        elif len(self.children) == 1:
            return self._f(m) and self.children[0].f(m)
        else:
            raise Exception(
                "%s does not support more than one child Matcher" % self.__name__
            )

    def _is_subspan(self, m, span):
        """
        Tests if mention m is subspan of span, where span is defined specific
        to mention type.
        """
        return False

    def _get_span(self, m):
        """
        Gets a tuple that identifies a span for the specific mention class
        that m belongs to.
        """
        return m

    def apply(self, mentions):
        """
        Apply the Matcher to a **generator** of mentions.
        Optionally only takes the longest match (NOTE: assumes this is the
        *first* match)
        """
        seen_spans = set()
        for m in mentions:
            if self.f(m) and (
                not self.longest_match_only
                or not any([self._is_subspan(m, s) for s in seen_spans])
            ):
                if self.longest_match_only:
                    seen_spans.add(self._get_span(m))
                yield m


class _NgramMatcher(_Matcher):
    """Matcher base class for Ngram objects"""

    def _is_subspan(self, m, span):
        """
        Tests if mention m is subspan of span, where span is defined
        specific to mention type.
        """
        return (
            m.sentence.id == span[0]
            and m.char_start >= span[1]
            and m.char_end <= span[2]
        )

    def _get_span(self, m):
        """
        Gets a tuple that identifies a span for the specific mention class
        that m belongs to.
        """
        return (m.sentence.id, m.char_start, m.char_end)


class DictionaryMatch(_NgramMatcher):
    """Selects mention Ngrams that match against a given list *d*.

    :param d: A list of strings representing a dictionary.
    :type d: list of str
    :param ignore_case: Whether to ignore the case when matching. Default True.
    :type ignore_case: bool
    :param inverse: Whether to invert the results (e.g., return those which are
        not in the list). Default False.
    :type inverse: bool
    :param stemmer: Optionally provide a stemmer to preprocess the dictionary.
        Can be any object which has a ``stem()`` method. Use stemmer="porter"
        to use a PorterStemmer(). Default None.
    """

    def init(self):
        self.ignore_case = self.opts.get("ignore_case", True)
        self.attrib = self.opts.get("attrib", WORDS)
        self.inverse = self.opts.get("inverse", False)
        try:
            self.d = frozenset(
                w.lower() if self.ignore_case else w for w in self.opts["d"]
            )
        except KeyError:
            raise Exception("Please supply a dictionary (list of strings) d as d=d.")

        # Optionally use a stemmer, preprocess the dictionary
        # Note that user can provide *an object having a stem() method*
        self.stemmer = self.opts.get("stemmer", None)
        if self.stemmer is not None:
            if self.stemmer == "porter":
                self.stemmer = PorterStemmer()
            self.d = frozenset(self._stem(w) for w in list(self.d))

    def _stem(self, w):
        """Apply stemmer, handling encoding errors"""
        try:
            return self.stemmer.stem(w)
        except UnicodeDecodeError:
            return w

    def _f(self, m):
        p = m.get_attrib_span(self.attrib)
        p = p.lower() if self.ignore_case else p
        p = self._stem(p) if self.stemmer is not None else p
        return (not self.inverse) if p in self.d else self.inverse


class LambdaFunctionMatcher(_NgramMatcher):
    """Selects Ngrams that return True when fed to a function f.

    :param func: The function to evaluate.
    :type func: function
    :param longest_match_only: Whether to only return the longest span matched,
        rather than all spans. Default False.
    :type longest_match_only: bool
    """

    def init(self):
        self.attrib = self.opts.get("attrib", WORDS)

        # Set longest match only to False by default.
        self.longest_match_only = self.opts.get("longest_match_only", False)
        try:
            self.func = self.opts["func"]
        except KeyError:
            raise Exception("Please supply a function f as func=f.")

    def _f(self, m):
        """The internal (non-composed) version of filter function f"""
        return self.func(m)


class Union(_NgramMatcher):
    """Takes the union of mention sets returned by the provided ``Matchers``."""

    def f(self, m):
        for child in self.children:
            if child.f(m) > 0:
                return True
        return False


class Intersect(_Matcher):
    """Takes the intersection of mention sets returned by the provided ``Matchers``."""

    def f(self, m):
        for child in self.children:
            if not child.f(m):
                return False
        return True


class Inverse(_Matcher):
    """Returns the opposite result of ifs child ``Matcher``.

    :raises ValueError: If more than one Matcher is provided.
    """

    # TODO: confirm that this only has one child
    def f(self, m):
        if len(self.children) > 1:
            raise ValueError("Provide a single Matcher.")
        for child in self.children:
            return not child.f(m)


class Concat(_NgramMatcher):
    """Selects mentions which are the concatenation of adjacent matches from
    child operators.

    :Example:
        A concatenation of a NumberMatcher and PersonMatcher could match on
        a span of text like "10 Obama".

    :param permutations: Default False.
    :type permutations: bool
    :param left_required: Whether or not to require the left child to match.
        Default True.
    :type left_required: bool
    :param right_required: Whether or not to require the right child to match.
        Default True.
    :type right_required: bool
    :param ignore_sep: Whether or not to ignore the separator. Default True.
    :type ignore_sep: bool
    :param sep: If not ignoring the separator, specify which separator to look
        for. Default sep=" ".
    :type set: str
    :raises ValueError: If Concat is not provided with two child matcher
        objects.

    .. note:: Currently slices on **word index** and considers concatenation
        along these divisions only.
    """

    def init(self):
        self.permutations = self.opts.get("permutations", False)
        self.left_required = self.opts.get("left_required", True)
        self.right_required = self.opts.get("right_required", True)
        self.ignore_sep = self.opts.get("ignore_sep", True)
        self.sep = self.opts.get("sep", " ")

    def f(self, m):
        if len(self.children) != 2:
            raise ValueError("Concat takes two child Matcher objects as arguments.")
        if not self.left_required and self.children[1].f(m):
            return True
        if not self.right_required and self.children[0].f(m):
            return True

        # Iterate over mention splits **at the word boundaries**
        for wsplit in range(m.get_word_start_index() + 1, m.get_word_end_index() + 1):
            csplit = (
                m._word_to_char_index(wsplit) - m.char_start
            )  # NOTE the switch to **mention-relative** char index

            # Optionally check for specific separator
            if self.ignore_sep or m.get_span()[csplit - 1] == self.sep:
                m1 = m[: csplit - len(self.sep)]
                m2 = m[csplit:]
                if self.children[0].f(m1) and self.children[1].f(m2):
                    return True
                if (
                    self.permutations
                    and self.children[1].f(m1)
                    and self.children[0].f(m2)
                ):
                    return True
        return False


class _RegexMatch(_NgramMatcher):
    """
    Base regex class. Does not specify specific semantics of *what* is being
    matched yet.
    """

    def init(self):
        try:
            self.rgx = self.opts["rgx"]
        except KeyError:
            raise Exception("Please supply a regular expression string r as rgx=r.")
        self.ignore_case = self.opts.get("ignore_case", True)
        self.attrib = self.opts.get("attrib", WORDS)
        self.sep = self.opts.get("sep", " ")

        # Extending the _RegexMatch to handle search(instead of only match)
        # and adding a toggle for full span match.
        # Default values are set to False and True for search flag and full
        # span matching flag respectively.
        self.search = self.opts.get("search", False)
        self.full_match = self.opts.get("full_match", True)

        # Compile regex matcher
        # NOTE: Enforce full span matching by ensuring that regex ends with $.
        # Group self.rgx first so that $ applies to all components of an 'OR'
        # expression. (e.g., we want r'(a|b)$' instead of r'a|b$')
        self.rgx = (
            self.rgx
            if self.rgx.endswith("$") or not self.full_match
            else ("(" + self.rgx + ")$")
        )
        self.r = re.compile(
            self.rgx, flags=(re.I if self.ignore_case else 0) | re.UNICODE
        )

    def _f(self, m):
        raise NotImplementedError()


class RegexMatchSpan(_RegexMatch):
    """Matches regex pattern on **full concatenated span**.

    :param rgx: The RegEx pattern to use.
    :type rgx: str
    :param ignore_case: Whether or not to ignore case in the RegEx. Default
        True.
    :type ignore_case: bool
    :param search: If True, _search_ regex pattern on full concatenated span.
        Default False.
    :type search: bool
    :param full_match: If True, wrap the provided rgx with ``(<rgx>)$``.
        Default True.
    :type full_match: bool
    :param longest_match_only: If True, only return the longest match. Default
        True.
    :type longest_match_only: bool
    """

    def _f(self, m):
        if self.search:
            return (
                True
                if self.r.search(m.get_attrib_span(self.attrib, sep=self.sep))
                is not None
                else False
            )
        else:
            return (
                True
                if self.r.match(m.get_attrib_span(self.attrib, sep=self.sep))
                is not None
                else False
            )


class RegexMatchEach(_RegexMatch):
    """Matches regex pattern on **each token**.

    :param rgx: The RegEx pattern to use.
    :type rgx: str
    :param ignore_case: Whether or not to ignore case in the RegEx. Default
        True.
    :type ignore_case: bool
    :param full_match: If True, wrap the provided rgx with ``(<rgx>)$``.
        Default True.
    :type full_match: bool
    :param longest_match_only: If True, only return the longest match. Default
        True.
    :type longest_match_only: bool
    """

    def _f(self, m):
        tokens = m.get_attrib_tokens(self.attrib)
        return (
            True
            if tokens and all([self.r.match(t) is not None for t in tokens])
            else False
        )


class PersonMatcher(RegexMatchEach):
    """
    Matches Spans that are the names of people, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as a person (PERSON).
    """

    def __init__(self, *children, **kwargs):
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "PERSON"
        super(PersonMatcher, self).__init__(*children, **kwargs)


class LocationMatcher(RegexMatchEach):
    """
    Matches Spans that are the names of locations, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as a location (GPE or LOC).
    """

    def __init__(self, *children, **kwargs):
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "GPE|LOC"
        super(LocationMatcher, self).__init__(*children, **kwargs)


class OrganizationMatcher(RegexMatchEach):
    """
    Matches Spans that are the names of organizations, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as an organization (NORG or ORG).
    """

    def __init__(self, *children, **kwargs):
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "NORG|ORG"
        super(OrganizationMatcher, self).__init__(*children, **kwargs)


class DateMatcher(RegexMatchEach):
    """
    Matches Spans that are dates, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as a date (DATE).
    """

    def __init__(self, *children, **kwargs):
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "DATE"
        super(DateMatcher, self).__init__(*children, **kwargs)


class NumberMatcher(RegexMatchEach):
    """
    Matches Spans that are numbers, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as a number (NUMBER or QUANTITY).
    """

    def __init__(self, *children, **kwargs):
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "NUMBER|QUANTITY"
        super(NumberMatcher, self).__init__(*children, **kwargs)


class MiscMatcher(RegexMatchEach):
    """
    Matches Spans that are miscellaneous named entities, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as miscellaneous (MISC).
    """

    def __init__(self, *children, **kwargs):
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "MISC"
        super(MiscMatcher, self).__init__(*children, **kwargs)


class _FigureMatcher(_Matcher):
    """Matcher base class for Figure objects"""

    def _is_subspan(self, m, span):
        """Tests if mention m does exist"""
        return m.figure.document.id == span[0] and m.figure.position == span[1]

    def _get_span(self, m):
        """
        Gets a tuple that identifies a figure for the specific mention class
        that m belongs to.
        """
        return (m.figure.document.id, m.figure.position)


class LambdaFunctionFigureMatcher(_FigureMatcher):
    """Selects Figures that return True when fed to a function f.

    :param func: The function to evaluate.
    :type func: function
    """

    def init(self):
        # Set longest match only to False
        self.longest_match_only = False
        try:
            self.func = self.opts["func"]
        except KeyError:
            raise Exception("Please supply a function f as func=f.")

    def _f(self, m):
        """The internal (non-composed) version of filter function f"""
        return self.func(m)
