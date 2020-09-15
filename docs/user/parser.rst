Parsing
=======
The first stage of Fonduer_'s pipeline is to parse an input corpus of documents
into the Fonduer_ data model.

Fonduer supports different file formats: CSV/TSV, TXT, HTML, and hOCR.
:class:`DocPreprocessor` transforms files in different formats into an uniform format that :class:`Parser` can parse.
The diagram below illustrates example pipelines (up to :class:`Parser`) for each format:

.. mermaid::

    graph LR;
        Image[(Image)]--OCR-->hOCR[(hOCR)];
        CSV[(CSV)]-->CSVDoc;
        TXT[(TXT)]-->TXTDoc;
        HTML[(HTML)]-->HTMLDoc;
        HTML2[(HTML)]-->HTMLDoc;
        PDF[(PDF)]--Convert-->HTML2[(HTML)];
        hOCR-->HOCRDoc;
        PDF-->vizlink;
    subgraph Fonduer
        CSVDoc(CSVDocPreprocessor)-->parser;
        TXTDoc(TXTDocPreprocessor)-->parser;
        HTMLDoc(HTMLDocPreprocessor)-->parser;
        HOCRDoc(HOCRDocPreprocessor)-->parser;
        parser(Parser)-->others(..);
        vizlink(VisualLinker) --- parser;
    end
    classDef source fill:#aaf;
    classDef preproc fill:#afa;
    class Image,CSV,TXT,HTML,PDF source;
    class HOCRDoc,CSVDoc,TXTDoc,HTMLDoc preproc;

Nodes in dark blue represent original source files.
Some of them are converted into different files: an image (incl. non-searchable PDF) is OCRed and exported in hOCR, a (born-digital) PDF is converted into HTML using tools like `pdftotree`.

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
