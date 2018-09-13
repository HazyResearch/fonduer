Candidate Extraction
====================
The second stage of Fonduer_'s pipeline is to extract Mentions and Candidates
from the data model.

Candidate Model Classes
-----------------------

The following describes elements of used for Mention and Candidate
extraction.

.. automodule:: fonduer.candidates.models
    :members:
    :inherited-members:
    :show-inheritance:

Core Objects
------------

These are Fonduer_'s core objects used for Mention and Candidate extraction.

.. automodule:: fonduer.candidates
    :members:
    :inherited-members:
    :show-inheritance:

Matchers
--------

This shows the *matchers* included with Fonduer_. These matchers can be used
alone, or combined together, to define what spans of text should be made into
Mentions.

.. autoclass:: fonduer.candidates.matchers.DateMatcher
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.DictionaryMatch
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.FigureMatcher
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.LambdaFunctionFigureMatcher
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.LambdaFunctionMatcher
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.LocationMatcher
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.MiscMatcher
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.NumberMatcher
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.OrganizationMatcher
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.PersonMatcher
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.RegexMatchEach
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.RegexMatchSpan
    :show-inheritance:

Matcher Operators
-----------------

These are the operators which can be use to compose *matchers*.

.. autoclass:: fonduer.candidates.matchers.Concat
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.Intersect
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.Inverse
    :show-inheritance:

.. autoclass:: fonduer.candidates.matchers.Union
    :show-inheritance:

.. _Fonduer: https://github.com/HazyResearch/fonduer
