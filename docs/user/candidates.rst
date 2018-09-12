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
    :undoc-members:

Core Objects
------------

These are Fonduer_'s core objects used for Mention and Candidate extraction.

.. automodule:: fonduer.candidates
    :members:
    :undoc-members:

Matchers
--------

This shows the *matchers* included with Fonduer_. These matchers can be used
alone, or combined together, to define what spans of text should be made into
Mentions.

.. automodule:: fonduer.candidates.matchers
    :members:
    :undoc-members:


.. _Fonduer: https://github.com/HazyResearch/fonduer
