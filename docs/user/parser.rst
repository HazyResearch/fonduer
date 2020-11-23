Parsing
=======
The first stage of Fonduer_'s pipeline is to parse an input corpus of documents
into the Fonduer_ data model.

Fonduer supports different file formats: CSV/TSV, TXT, HTML, and hOCR.
The diagram below illustrates how files in each format are preprocessed and consumed by
:class:`Parser`. Nodes in dark blue represent original source files.
You have to convert some of them into the formats that Fonduer can consume:
a scanned document (incl. non-searchable PDF) is OCRed and exported in hOCR,
a (born-digital) PDF is converted into hOCR using tools like pdftotree_.
It is also possible to convert PDF into HTML using third-party tools,
but not recommended (see `Visual Parsers`_).

.. mermaid::

    graph LR;
        CSV[(CSV)]-->CSVDoc;
        TXT[(TXT)]-->TXTDoc;
        HTML[(HTML)]-->HTMLDoc;
        PDF[(PDF)]--Convert-->HTML2[(HTML)];
        HTML2-->HTMLDoc;
        PDF--Convert-->hOCR2[(hOCR)];
        Scan[(Scan)]--OCR-->hOCR[(hOCR)];
        hOCR-->HOCRDoc;
        hOCR2-->HOCRDoc;
    subgraph Fonduer
        CSVDoc(CSVDocPreprocessor)-->parser;
        TXTDoc(TXTDocPreprocessor)-->parser;
        HTMLDoc(HTMLDocPreprocessor)-->parser;
        HOCRDoc(HOCRDocPreprocessor)-->parser;
        parser(Parser)-->others(..);
    end
    classDef source fill:#aaf;
    classDef preproc fill:#afa;
    class Scan,CSV,TXT,HTML,PDF source;
    class HOCRDoc,CSVDoc,TXTDoc,HTMLDoc preproc;

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

Lingual Parsers
---------------

The following docs describe various lingual parsers. They split text into sentences
and enrich them with NLP.

.. automodule:: fonduer.parser.lingual_parser
    :members:
    :inherited-members:
    :show-inheritance:

Visual Parsers
--------------

The following docs describe various visual parsers. They parse visual information,
e.g., bounding boxes of each word.
Fonduer can parse visual information only for hOCR and HTML files
with help of :class:`HocrVisualParser` and :class:`PdfVisualParser`, respectively.
It is recommended to provide documents in hOCR instead of HTML,
because :class:`PdfVisualParser` is not always accurate by its nature and could
assign a wrong bounding box to a word.
(see `#12 <https://github.com/HazyResearch/fonduer/issues/12>`_).

.. mermaid::

    graph LR;
        PDF[(PDF)]--Convert-->HTML;
        PDF--Convert-->hOCR;
        hOCR[(hOCR)]-->HOCRDoc;
        HTML[(HTML)]-->HTMLDoc;
        PDF-.->pdf_visual_parser;
    subgraph Fonduer
        HOCRDoc(HOCRDocPreprocessor)-->parser;
        parser(Parser)-->hocr_visual_parser(HocrVisualParser);
        HTMLDoc(HTMLDocPreprocessor)-->parser2;
        parser2(Parser)-->pdf_visual_parser(PdfVisualParser);
        hocr_visual_parser-->others;
        pdf_visual_parser-->others(..);
    end
    classDef source fill:#aaf;
    classDef preproc fill:#afa;
    class PDF source;
    class HTMLDoc,HOCRDoc preproc;

.. automodule:: fonduer.parser.visual_parser
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
.. _pdftotree: https://github.com/HazyResearch/pdftotree
