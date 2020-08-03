"""Fonduer matcher."""
import re
from typing import Iterator, Set

from fonduer.candidates.models.figure_mention import TemporaryFigureMention
from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.candidates.models.temporary_context import TemporaryContext

WORDS = "words"


class _Matcher(object):
    """Matcher class.

    Apply a function ``f : m -> {True, False}`` to a generator of mentions,
    returning only mentions *m* s.t. *f(m) == True*,
    where f can be compositionally defined.
    """

    def __init__(self, *children, **opts):  # type: ignore
        self.children = children
        self.opts = opts
        self.longest_match_only = self.opts.get("longest_match_only", True)
        self.init()
        self._check_opts()

    def init(self) -> None:
        pass

    def _check_opts(self) -> None:
        """Check for unsupported opts, throws error if found.

        NOTE: Must be called _after_ init()
        """
        for opt in self.opts.keys():
            if opt not in self.__dict__:
                raise Exception(f"Unsupported option: {opt}")

    def _f(self, m: TemporaryContext) -> bool:
        """Non-composed version of filter function f."""
        return True

    def f(self, m: TemporaryContext) -> bool:
        """(Recursively) Composed version of filter function f.

        By default, returns logical **conjunction** of operator and single
        child operator
        """
        if len(self.children) == 0:
            return self._f(m)
        elif len(self.children) == 1:
            return self._f(m) and self.children[0].f(m)
        else:
            raise Exception(
                f"{self.__class__.__name__} does not support two or more child Matcher"
            )

    def apply(self, mentions: Iterator[TemporaryContext]) -> Iterator[TemporaryContext]:
        """Apply the Matcher to a **generator** of mentions.

        Optionally only takes the longest match (NOTE: assumes this is the
        *first* match)
        """
        seen_mentions: Set[TemporaryContext] = set()
        for m in mentions:
            if self.f(m) and (
                not self.longest_match_only or not any([m in s for s in seen_mentions])
            ):
                if self.longest_match_only:
                    seen_mentions.add(m)
                yield m


class DictionaryMatch(_Matcher):
    """Select mention Ngrams that match against a given list *d*.

    :param d: A list of strings representing a dictionary.
    :type d: list of str
    :param ignore_case: Whether to ignore the case when matching. Default True.
    :type ignore_case: bool
    :param inverse: Whether to invert the results (e.g., return those which are
        not in the list). Default False.
    :type inverse: bool
    :param stemmer: Optionally provide a stemmer to preprocess the dictionary.
        Can be any object which has a ``stem(str) -> str`` method
        like ``PorterStemmer()``. Default None.
    """

    def init(self) -> None:
        """Initialize the dictionary matcher."""
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
        if self.stemmer:
            self.d = frozenset(self._stem(w) for w in list(self.d))

    def _stem(self, w: str) -> str:
        """Apply stemmer, handling encoding errors."""
        try:
            return self.stemmer.stem(w)
        except UnicodeDecodeError:
            return w

    def _f(self, m: TemporaryContext) -> bool:
        if not isinstance(m, TemporarySpanMention):
            raise ValueError(
                f"{self.__class__.__name__} only supports TemporarySpanMention"
            )
        p = m.get_attrib_span(self.attrib)
        p = p.lower() if self.ignore_case else p
        p = self._stem(p) if self.stemmer is not None else p
        return (not self.inverse) if p in self.d else self.inverse


class LambdaFunctionMatcher(_Matcher):
    """Select ``Ngrams`` that return True when fed to a function f.

    :param func: The function to evaluate with a signature of ``f: m -> {True, False}``,
        where ``m`` denotes a mention. More precisely, ``m`` is an instance of child
        class of :class:`TemporaryContext`, depending on which :class:`MentionSpace` is
        used. E.g., :class:`TemporarySpanMention` when :class:`MentionNgrams` is used.
    :type func: function
    :param longest_match_only: Whether to only return the longest span matched,
        rather than all spans. Default False.
    :type longest_match_only: bool
    """

    def init(self) -> None:
        """Initialize the lambda function matcher."""
        self.attrib = self.opts.get("attrib", WORDS)

        # Set longest match only to False by default.
        self.longest_match_only = self.opts.get("longest_match_only", False)
        try:
            self.func = self.opts["func"]
        except KeyError:
            raise Exception("Please supply a function f as func=f.")

    def _f(self, m: TemporaryContext) -> bool:
        """Non-composed version of filter function f."""
        if not isinstance(m, TemporarySpanMention):
            raise ValueError(
                f"{self.__class__.__name__} only supports TemporarySpanMention"
            )
        return self.func(m)


class Union(_Matcher):
    """Take the union of mention sets returned by the provided ``Matchers``.

    :param longest_match_only: If True, only return the longest match. Default True.
        Overrides longest_match_only of its child ``Matchers``.
    :type longest_match_only: bool
    """

    def f(self, m: TemporaryContext) -> bool:
        """Non-composed version of filter function f."""
        for child in self.children:
            if child.f(m):
                return True
        return False


class Intersect(_Matcher):
    """Take the intersection of mention sets returned by the provided ``Matchers``.

    :param longest_match_only: If True, only return the longest match. Default True.
        Overrides longest_match_only of its child ``Matchers``.
    :type longest_match_only: bool
    """

    def f(self, m: TemporaryContext) -> bool:
        """Non-composed version of filter function f."""
        for child in self.children:
            if not child.f(m):
                return False
        return True


class Inverse(_Matcher):
    """Return the opposite result of ifs child ``Matcher``.

    :raises ValueError: If more than one Matcher is provided.
    :param longest_match_only: If True, only return the longest match. Default True.
        Overrides longest_match_only of its child ``Matchers``.
    :type longest_match_only: bool
    """

    def __init__(self, *children, **opts):  # type: ignore
        """Initialize inverse matcher."""
        if not len(children) == 1:
            raise ValueError("Provide a single Matcher.")
        super().__init__(*children, **opts)

    def f(self, m: TemporaryContext) -> bool:
        """Non-composed version of filter function f."""
        child = self.children[0]
        return not child.f(m)


class Concat(_Matcher):
    """Concatenate mentions generated by Matchers.

    Select mentions which are the concatenation of adjacent matches from child
    operators.

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

    def init(self) -> None:
        """Initialize concatenate matcher."""
        self.permutations = self.opts.get("permutations", False)
        self.left_required = self.opts.get("left_required", True)
        self.right_required = self.opts.get("right_required", True)
        self.ignore_sep = self.opts.get("ignore_sep", True)
        self.sep = self.opts.get("sep", " ")

    def f(self, m: TemporaryContext) -> bool:
        """Non-composed version of filter function f."""
        if not isinstance(m, TemporarySpanMention):
            raise ValueError(
                f"{self.__class__.__name__} only supports TemporarySpanMention"
            )
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


class _RegexMatch(_Matcher):
    """Regex matcher class.

    Select mentions using a regular expression. Does not specify the
    semantics of *what* is being matched yet.
    """

    def init(self) -> None:
        try:
            self.rgx = self.opts["rgx"]
        except KeyError:
            raise Exception("Please supply a regular expression string r as rgx=r.")
        self.ignore_case = self.opts.get("ignore_case", True)
        self.attrib = self.opts.get("attrib", WORDS)
        self.sep = self.opts.get("sep", "")

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

    def _f(self, m: TemporaryContext) -> bool:
        raise NotImplementedError()


class RegexMatchSpan(_RegexMatch):
    """Match regex pattern on **full concatenated span**.

    :param rgx: The RegEx pattern to use.
    :type rgx: str
    :param ignore_case: Whether or not to ignore case in the RegEx. Default
        True.
    :type ignore_case: bool
    :param search: If True, *search* the regex pattern through the concatenated span.
        If False, try to *match* the regex patten only at its beginning. Default False.
    :type search: bool
    :param full_match: If True, wrap the provided rgx with ``(<rgx>)$``.
        Default True.
    :type full_match: bool
    :param longest_match_only: If True, only return the longest match. Default True.
        Will be overridden by the parent matcher like :class:`Union` when it is wrapped
        by :class:`Union`, :class:`Intersect`, or :class:`Inverse`.
    :type longest_match_only: bool
    """

    def _f(self, m: TemporaryContext) -> bool:
        if not isinstance(m, TemporarySpanMention):
            raise ValueError(
                f"{self.__class__.__name__} only supports TemporarySpanMention"
            )
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
    """Match regex pattern on **each token**.

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

    def _f(self, m: TemporaryContext) -> bool:
        if not isinstance(m, TemporarySpanMention):
            raise ValueError(
                f"{self.__class__.__name__} only supports TemporarySpanMention"
            )
        tokens = m.get_attrib_tokens(self.attrib)
        return (
            True
            if tokens and all([self.r.match(t) is not None for t in tokens])
            else False
        )


class PersonMatcher(RegexMatchEach):
    """
    Match Spans that are the names of people, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as a person (PERSON).
    """

    def __init__(self, *children, **kwargs):  # type: ignore
        """Initialize person matcher."""
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "PERSON"
        super().__init__(*children, **kwargs)


class LocationMatcher(RegexMatchEach):
    """
    Match Spans that are the names of locations, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as a location (GPE or LOC).
    """

    def __init__(self, *children, **kwargs):  # type: ignore
        """Initialize location matcher."""
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "GPE|LOC"
        super().__init__(*children, **kwargs)


class OrganizationMatcher(RegexMatchEach):
    """
    Match Spans that are the names of organizations, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as an organization (NORG or ORG).
    """

    def __init__(self, *children, **kwargs):  # type: ignore
        """Initialize organization matcher."""
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "NORG|ORG"
        super().__init__(*children, **kwargs)


class DateMatcher(RegexMatchEach):
    """
    Match Spans that are dates, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as a date (DATE).
    """

    def __init__(self, *children, **kwargs):  # type: ignore
        """Initialize date matcher."""
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "DATE"
        super().__init__(*children, **kwargs)


class NumberMatcher(RegexMatchEach):
    """
    Match Spans that are numbers, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as a number (CARDINAL or QUANTITY).
    """

    def __init__(self, *children, **kwargs):  # type: ignore
        """Initialize number matcher."""
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "CARDINAL|QUANTITY"
        super().__init__(*children, **kwargs)


class MiscMatcher(RegexMatchEach):
    """
    Match Spans that are miscellaneous named entities, as identified by spaCy.

    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as miscellaneous (MISC).
    """

    def __init__(self, *children, **kwargs):  # type: ignore
        """Initialize miscellaneous matcher."""
        kwargs["attrib"] = "ner_tags"
        kwargs["rgx"] = "MISC"
        super().__init__(*children, **kwargs)


class LambdaFunctionFigureMatcher(_Matcher):
    """Select Figures that return True when fed to a function f.

    :param func: The function to evaluate. See :class:`LambdaFunctionMatcher` for
        details.
    :type func: function
    """

    def init(self) -> None:
        """Initialize lambda function figure matcher."""
        # Set longest match only to False
        self.longest_match_only = False
        try:
            self.func = self.opts["func"]
        except KeyError:
            raise Exception("Please supply a function f as func=f.")

    def _f(self, m: TemporaryContext) -> bool:
        """Non-composed version of filter function f."""
        if not isinstance(m, TemporaryFigureMention):
            raise ValueError(
                f"{self.__class__.__name__} only supports TemporaryFigureMention"
            )
        return self.func(m)


class DoNothingMatcher(_Matcher):
    """Matcher class for doing nothing."""

    pass
