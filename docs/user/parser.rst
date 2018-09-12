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
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Core Objects
------------

This is Fonduer_'s core Parser object.

.. automodule:: fonduer.parser
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Preprocessors
-------------

The following shows descriptions of the various document preprocessors included
with Fonduer_ which are used in parsing documents of different formats.

.. automodule:: fonduer.parser.preprocessors
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. _Fonduer: https://github.com/HazyResearch/fonduer
