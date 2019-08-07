Parsing
=======
The first stage of Fonduer_'s pipeline is to parse an input corpus of documents
into the Fonduer_ data model.

Multimodal Data Model
---------------------

The following docs describe elements of Fonduer_'s data model. These attributes
can be used when creating *matchers*, *throttlers*, and *labeling functions*.

.. automodule:: fonduer.parser.models
    :members:
    :inherited-members:
    :show-inheritance:

Core Objects
------------

This is Fonduer_'s core Parser object.

.. automodule:: fonduer.parser
    :members:
    :inherited-members:
    :show-inheritance:

.. automodule:: fonduer.parser.visual_linker
    :members:
    :inherited-members:
    :show-inheritance:

Lingual Parsers
---------------

The following docs describe various lingual parsers. They split text into sentences
and enrich them with NLP.

.. automodule:: fonduer.parser.lingual_parser
    :members:
    :inherited-members:
    :show-inheritance:

Preprocessors
-------------

The following shows descriptions of the various document preprocessors included
with Fonduer_ which are used in parsing documents of different formats.

.. automodule:: fonduer.parser.preprocessors
    :members:
    :inherited-members:
    :show-inheritance:

.. _Fonduer: https://github.com/HazyResearch/fonduer
