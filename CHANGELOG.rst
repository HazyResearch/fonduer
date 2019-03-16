[Unreleased]
------------

Added
^^^^^
* `@lukehsiao`_: provide ``fonduer.init_logging()`` as a way to configure
  logging to a temp directory by default.

.. note::

    Although you can still configure ``logging`` manually, with this change
    we also provide a function for initializing logging. For example, you
    can call:

    .. code:: python

        import logging
        import fonduer

        # Optionally configure logging
        fonduer.init_logging(
          log_dir="log_folder",
          format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
          level=logging.INFO
        )

        session = fonduer.Meta.init(conn_string).Session()

    which will create logs within the ``log_filder`` directory. If logging is
    not explicitly initialized, we will provide a default configuration which
    will store logs in a temporary directory.

Fixed
^^^^^
* `@senwu`_: Update the metal version.
* `@senwu`_: Expose the ``b`` and ``pos_label`` in training.
* `@senwu`_: Fix the issue that pdfinfo causes parsing error when it contains
  more than one ``Page``.


[0.6.0] - 2019-02-17
--------------------

Removed
^^^^^^^
* `@lukehsiao`_: Removed the redundant ``get_gold_labels`` function.

.. note::

    Rather than calling get_gold_labels directly, call it from the Labeler:

    .. code:: python

        from fonduer.supervision import Labeler
        labeler = Labeler(session, [relations])
        L_gold_train = labeler.get_gold_labels(train_cands, annotator='gold')

    Rather than:

    .. code:: python

        from fonduer.supervision import Labeler, get_gold_labels
        labeler = Labeler(session, [relations])
        L_gold_train = get_gold_labels(session, train_cands, annotator_name='gold')

Changed
^^^^^^^
* `@lukehsiao`_: improved performance of ``data_model_utils`` through caching
  and simplifying the underlying queries.
  (`#212 <https://github.com/HazyResearch/fonduer/pull/212>`_,
  `#215 <https://github.com/HazyResearch/fonduer/pull/215>`_)
* `@senwu`_: upgrade to PyTorch v1.0.0.
  (`#209 <https://github.com/HazyResearch/fonduer/pull/209>`_)


Fixed
^^^^^
* `@senwu`_: Improve type checking in featurization.
* `@lukehsiao`_: Fixed sentence.sentence_num bug in get_neighbor_sentence_ngrams.
* `@lukehsiao`_: Add session synchronization to sqlalchemy delete queries.
  (`#214 <https://github.com/HazyResearch/fonduer/pull/214>`_)
* `@lukehsiao`_: Update PyYAML dependency to patch CVE-2017-18342.
  (`#205 <https://github.com/HazyResearch/fonduer/pull/205>`_)
* `@KenSugimoto`_: Fix max/min in ``visualizer.get_box``

[0.5.0] - 2019-01-01
--------------------

Added
^^^^^
* `@senwu`_: Support CSV, TSV, Text input data format.
  For CSV format, ``CSVDocPreprocessor`` treats each line in the input file as
  a document. It assumes that each column is one section and content in each
  column as one paragraph as default. However, if the column is complex, an
  advanced parser may be used by specifying ``parser_rule`` parameter in a dict
  format where key is the column index and value is the specific parser.

.. note::

    In Fonduer v0.5.0, you can use ``CSVDocPreprocessor``:

    .. code:: python

        from fonduer.parser import Parser
        from fonduer.parser.preprocessors import CSVDocPreprocessor
        from fonduer.utils.utils_parser import column_constructor

        max_docs = 10

        # Define specific parser for the third column (index 2), which takes ``text``,
        # ``name=None``, ``type="text"``, and ``delim=None`` as input and generate
        # ``(content type, content name, content)`` for ``build_node``
        # in ``fonduer.utils.utils_parser``.
        parser_rule = {
            2: partial(column_constructor, type="figure"),
        }

        doc_preprocessor = CSVDocPreprocessor(
            PATH_TO_DOCS, max_docs=max_docs, header=True, parser_rule=parser_rule
        )

        corpus_parser = Parser(session, structural=True, lingual=True, visual=False)
        corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

        all_docs = corpus_parser.get_documents()

  For TSV format, ``TSVDocPreprocessor`` assumes each line in input file as a
  document which should follow (doc_name <tab> doc_text) format.

  For Text format, ``TextDocPreprocessor`` assumes one document per file.

Fixed
^^^^^
* `@HiromuHota`_: Modify docstring of functions that return get_sparse_matrix
* `@lukehsiao`_: Fix the behavior of ``get_last_documents`` to return Documents
  that are correctly linked to the database and can be navigated by the user.
  (`#201 <https://github.com/HazyResearch/fonduer/pull/201>`_)
* `@lukehsiao`_: Fix the behavior of MentionExtractor ``clear`` and
  ``clear_all`` to also delete the Candidates that correspond to the Mentions.

Changed
^^^^^^^
* `@senwu`_: Reorganize ``learning`` module to use pytorch dataloader, include
  ``MultiModalDataset`` to better handle multimodal information, and simplify
  the code
* `@senwu`_: Remove ``batch_size`` input argument from ``_calc_logits``,
  ``marginals``, ``predict``, and ``score`` in ``Classifier``
* `@senwu`_: Rename ``predictions`` to ``predict`` in ``Classifier`` and update
  the input arguments to have ``pos_label`` (assign positive label for binary class
  prediction) and ``return_probs`` (If True, return predict probablities as well)
* `@senwu`_: Update ``score`` function in ``Classifier`` to include:
  (1) For binary: precision, recall, F-beta score, accuracy, ROC-AUC score;
  (2) For categorical: accuracy;
* `@senwu`_: Remove ``LabelBalancer``
* `@senwu`_: Remove original ``Classifier`` class, rename ``NoiseAwareModel`` to
  ``Classifier`` and use the same setting for both binary and multi-class classifier
* `@senwu`_: Unify the loss (``SoftCrossEntropyLoss``) for all settings
* `@senwu`_: Rename ``layers`` in learning module to ``modules``
* `@senwu`_: Update code to use Python 3.6+'s f-strings
* `@HiromuHota`_: Reattach doc with the current session at
  MentionExtractorUDF#apply to avoid doing so at each MentionSpace.

[0.4.1] - 2018-12-12
--------------------

Added
^^^^^
* `@senwu`_: Added alpha spacy support for Chinese tokenizer.

Fixed
^^^^^
* `@senwu`_: fix non-deterministic issue from get_candidates and get_mentions
  by parallel candidate/mention generation.

Changed
^^^^^^^
* `@lukehsiao`_: Add soft version pinning to avoid failures due to dependency
  API changes.
* `@j-rausch`_: Change ``get_row_ngrams`` and ``get_col_ngrams`` to return
  ``None`` if the passed ``Mention`` argument is not inside a table.
  (`#194 <https://github.com/HazyResearch/fonduer/pull/194>`_)


[0.4.0] - 2018-11-27
--------------------

Added
^^^^^
* `@senwu`_: Rename ``span`` attribute to ``context`` in mention_subclass to
  better support mulitmodal mentions.
  (`#184 <https://github.com/HazyResearch/fonduer/pull/184>`_)

.. note::
    The way to retrieve corresponding data model object from mention changed.
    In Fonduer v0.3.6, we use ``.span``:

    .. code:: python

        # sent_mention is a SentenceMention
        sentence = sent_mention.span.sentence

    With this release, we use ``.context``:

    .. code:: python

        # sent_mention is a SentenceMention
        sentence = sent_mention.context.sentence

* `@senwu`_: Add support to extract multimodal candidates and add
  ``DoNothingMatcher`` matcher.
  (`#184 <https://github.com/HazyResearch/fonduer/pull/184>`_)

.. note::
    The Mention extraction support all data types in data model. In Fonduer
    v0.3.6, Mention extraction only supports ``MentionNgrams`` and
    ``MentionFigures``:

    .. code:: python

        from fonduer.candidates import (
            MentionFigures,
            MentionNgrams,
        )

    With this release, it supports all data types:

    .. code:: python

        from fonduer.candidates import (
            MentionCaptions,
            MentionCells,
            MentionDocuments,
            MentionFigures,
            MentionNgrams,
            MentionParagraphs,
            MentionSections,
            MentionSentences,
            MentionTables,
        )

* `@senwu`_: Add support to parse multiple sections in parser, fix webpage
  context, and add name column for each context in data model.
  (`#182 <https://github.com/HazyResearch/fonduer/pull/182>`_)

Fixed
^^^^^
* `@senwu`_: Remove unnecessary backref in mention generation.
* `@j-rausch`_: Improve error handling for invalid row spans.
  (`#183 <https://github.com/HazyResearch/fonduer/pull/183>`_)


[0.3.6] - 2018-11-15
--------------------

Fixed
^^^^^
* `@lukehsiao`_: Updated snorkel-metal version requirement to ensure new syntax
  works when a user upgrades Fonduer.
* `@lukehsiao`_: Improve error messages on PostgreSQL connection and update FAQ.


[0.3.5] - 2018-11-04
--------------------

Added
^^^^^^^
* `@senwu`_: Add ``SparseLSTM`` support reducing the memory used by the LSTM
  for large applications.
  (`#175 <https://github.com/HazyResearch/fonduer/pull/175>`_)

.. note::
    With the SparseLSTM discriminative model, we save memory for the origin
    LSTM model while sacrificing runtime. In Fonduer v0.3.5, SparseLSTM is as
    follows:

    .. code:: python

        from fonduer.learning import SparseLSTM

        disc_model = SparseLSTM()
        disc_model.train(
            (train_cands, train_feature), train_marginals, n_epochs=5, lr=0.001
        )


Fixed
^^^^^
* `@senwu`_: Fix issue with ``get_last_documents`` returning the incorrect
  number of docs and update the tests.
  (`#176 <https://github.com/HazyResearch/fonduer/pull/176>`_)

* `@senwu`_: Use the latest MeTaL syntax and fix flake8 issues.
  (`#173 <https://github.com/HazyResearch/fonduer/pull/173>`_)


[0.3.4] - 2018-10-17
--------------------

Changed
^^^^^^^
* `@senwu`_: Use ``sqlalchemy`` to check connection string. Use ``postgresql``
  instead of ``postgres`` in connection string.

Fixed
^^^^^
* `@lukehsiao`_: The features/labels/gold_label key tables were not properly
  designed for multiple relations in that they indistinguishably shared the
  global index of keys. This fixes this issue by including the names of the
  relations associated with each key. In addition, this ensures that clearing a
  single relation, or relabeling a single training relation does not
  inadvertently corrupt the global index of keys.
  (`#167 <https://github.com/HazyResearch/fonduer/pull/167>`_)

[0.3.3] - 2018-09-27
--------------------
Changed
^^^^^^^
* `@lukehsiao`_: Added ``longest_match_only`` parameter to
  :class:`LambdaFunctionMatcher`, which defaults to False, rather than True.
  (`#165 <https://github.com/HazyResearch/fonduer/pull/165>`_)

Fixed
^^^^^
* `@lukehsiao`_: Fixes the behavior of the ``get_between_ngrams`` data model
  util. (`#164 <https://github.com/HazyResearch/fonduer/pull/164>`_)
* `@lukehsiao`_: Batch queries so that PostgreSQL buffers aren't exceeded.
  (`#162 <https://github.com/HazyResearch/fonduer/pull/162>`_)

[0.3.2] - 2018-09-20
--------------------
Fixed
^^^^^
* `@lukehsiao`_: Fix attribute error when using MentionFigures.

Changed
^^^^^^^
* `@lukehsiao`_: :class:`MentionNgrams` ``split_tokens`` now defaults to an
  empty list and splits on all occurrences, rather than just the first
  occurrence.
* `@j-rausch`_: Parser will now skip documents with parsing errors rather than
  crashing.

[0.3.1] - 2018-09-18
--------------------
Fixed
^^^^^
* `@lukehsiao`_: Fix the layers module in fonduer.learning.disc_models.layers.

[0.3.0] - 2018-09-18
--------------------
Added
^^^^^
* `@lukehsiao`_: Add supporting functions for incremental knowledge base
  construction. (`#154 <https://github.com/HazyResearch/fonduer/pull/154>`_)
* `@j-rausch`_: Added alpha spacy support for Japanese tokenizer.
* `@senwu`_: Add sparse logistic regression support.
* `@senwu`_: Support Python 3.7.
* `@lukehsiao`_: Allow user to change featurization settings by providing
  ``.fonduer-config.yaml`` in their project.
* `@lukehsiao`_: Add a new Mention object, and have Candidate objects be
  composed of Mention objects, rather than directly of Spans. This allows a
  single Mention to be reused in multiple relations.
* `@lukehsiao`_: Improved connection-string validation for the Meta class.

Changed
^^^^^^^
* `@j-rausch`_: ``Document.text`` now returns the modified document text, based
  on the user-defined html-tag stripping in the parsing stage.
* `@j-rausch`_: ``Ngrams`` now has a ``n_min`` argument to specify a minimum
  number of tokens per extracted n-gram.
* `@lukehsiao`_: Rename ``BatchLabelAnnotator`` to ``Labeler`` and
  ``BatchFeatureAnnotator`` to ``Featurizer``. The classes now support multiple
  relations.
* `@j-rausch`_: Made spacy tokenizer to default tokenizer, as long as there
  is (alpha) support for the chosen language. ```lingual``` argument now
  specifies whether additional spacy NLP processing shall be performed.
* `@senwu`_: Reorganize the disc model structure.
  (`#126 <https://github.com/HazyResearch/fonduer/pull/126>`_)
* `@lukehsiao`_: Add ``session`` and ``parallelism`` as a parameter to all UDF
  classes.
* `@j-rausch`_: Sentence splitting in lingual mode is now performed by
  spacy's sentencizer instead of the dependency parser. This can lead to
  variations in sentence segmentation and tokenization.
* `@j-rausch`_: Added ``language`` argument to ``Parser`` for specification
  of language used by ``spacy_parser``. E.g. ``language='en'```.
* `@senwu`_: Change weak supervision learning framework from numbskull to
  `MeTaL <https://github.com/HazyResearch/metal>_`.
  (`#119 <https://github.com/HazyResearch/fonduer/pull/119>`_)
* `@senwu`_: Change learning framework from Tensorflow to PyTorch.
  (`#115 <https://github.com/HazyResearch/fonduer/pull/115>`_)
* `@lukehsiao`_: Blacklist <script> nodes by default when parsing HTML docs.
* `@lukehsiao`_: Reorganize ReadTheDocs structure to mirror the repository
  structure. Now, each pipeline phase's user-facing API is clearly shown.
* `@lukehsiao`_: Rather than importing ambiguously from ``fonduer`` directly,
  disperse imports into their respective pipeline phases. This eliminates
  circular dependencies, and makes imports more explicit and clearer to the
  user where each import is originating from.
* `@lukehsiao`_: Provide debug logging of external subprocess calls.
* `@lukehsiao`_: Use ``tdqm`` for progress bar (including multiprocessing).
* `@lukehsiao`_: Set the default PostgreSQL client encoding to "utf8".
* `@lukehsiao`_: Organize documentation for ``data_model_utils`` by modality.
  (`#85 <https://github.com/HazyResearch/fonduer/pull/85>`_)
* `@lukehsiao`_: Rename ``lf_helpers`` to ``data_model_utils``, since they can
  be applied more generally to throttlers or used for error analysis, and are
  not limited to just being used in labeling functions.
* `@lukehsiao`_: Update the CHANGELOG to start following `KeepAChangelog
  <https://keepachangelog.com/en/1.0.0/>`_ conventions.

Removed
^^^^^^^
* `@lukehsiao`_: Remove the XMLMultiDocPreprocessor.
* `@lukehsiao`_: Remove the ``reduce`` option for UDFs, which were unused.
* `@lukehsiao`_: Remove get parent/children/sentence generator from Context.
  (`#87 <https://github.com/HazyResearch/fonduer/pull/87>`_)
* `@lukehsiao`_: Remove dependency on ``pdftotree``, which is currently unused.

Fixed
^^^^^
* `@j-rausch`_: Improve ``spacy_parser`` performance. We split the lingual
  parsing pipeline into two stages. First, we parse structure and gather all
  sentences for a document. Then, we merge and feed all sentences per document
  into the spacy NLP pipeline for more efficient processing.
* `@senwu`_: Speed-up of ``_get_node`` using caching.
* `@HiromuHota`_: Fixed bug with Ngram splitting and empty TemporarySpans.
  (`#108 <https://github.com/HazyResearch/fonduer/pull/108>`_,
  `#112 <https://github.com/HazyResearch/fonduer/pull/112>`_)
* `@lukehsiao`_: Fixed PDF path validation when using ``visual=True`` during
  parsing.
* `@lukehsiao`_: Fix Meta bug which would not switch databases when init() was
  called with a new connection string.

.. note::
    With the addition of Mentions, the process of Candidate extraction has
    changed. In Fonduer v0.2.3, Candidate extraction was as follows:

    .. code:: python

        candidate_extractor = CandidateExtractor(PartAttr,
                                [part_ngrams, attr_ngrams],
                                [part_matcher, attr_matcher],
                                candidate_filter=candidate_filter)

        candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    With this release, you will now first extract Mentions and then extract
    Candidates based on those Mentions:

    .. code:: python

        # Mention Extraction
        part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
        temp_ngrams = MentionNgramsTemp(n_max=2)
        volt_ngrams = MentionNgramsVolt(n_max=1)

        Part = mention_subclass("Part")
        Temp = mention_subclass("Temp")
        Volt = mention_subclass("Volt")
        mention_extractor = MentionExtractor(
            session,
            [Part, Temp, Volt],
            [part_ngrams, temp_ngrams, volt_ngrams],
            [part_matcher, temp_matcher, volt_matcher],
        )
        mention_extractor.apply(docs, split=0, parallelism=PARALLEL)

        # Candidate Extraction
        PartTemp = candidate_subclass("PartTemp", [Part, Temp])
        PartVolt = candidate_subclass("PartVolt", [Part, Volt])

        candidate_extractor = CandidateExtractor(
            session,
            [PartTemp, PartVolt],
            throttlers=[temp_throttler, volt_throttler]
        )

        candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    Furthermore, because Candidates are now composed of Mentions rather than
    directly of Spans, to get the Span object from a mention, use the ``.span``
    attribute of a Mention.

.. note::
    Fonduer has been reorganized to require more explicit import syntax. In
    Fonduer v0.2.3, nearly everything was imported directly from fonduer:

    .. code:: python

        from fonduer import (
            CandidateExtractor,
            DictionaryMatch,
            Document,
            FeatureAnnotator,
            GenerativeModel,
            HTMLDocPreprocessor,
            Intersect,
            LabelAnnotator,
            LambdaFunctionMatcher,
            MentionExtractor,
            Meta,
            Parser,
            RegexMatchSpan,
            Sentence,
            SparseLogisticRegression,
            Union,
            candidate_subclass,
            load_gold_labels,
            mention_subclass,
        )

    With this release, you will now import from each pipeline phase. This makes
    imports more explicit and allows you to more clearly see which pipeline
    phase each import is associated with:

    .. code:: python

        from fonduer import Meta
        from fonduer.candidates import CandidateExtractor, MentionExtractor
        from fonduer.candidates.matchers import (
            DictionaryMatch,
            Intersect,
            LambdaFunctionMatcher,
            RegexMatchSpan,
            Union,
        )
        from fonduer.candidates.models import candidate_subclass, mention_subclass
        from fonduer.features import Featurizer
        from metal.label_model import LabelModel # GenerativeModel in v0.2.3
        from fonduer.learning import SparseLogisticRegression
        from fonduer.parser import Parser
        from fonduer.parser.models import Document, Sentence
        from fonduer.parser.preprocessors import HTMLDocPreprocessor
        from fonduer.supervision import Labeler, get_gold_labels


[0.2.3] - 2018-07-23
--------------------

* `@lukehsiao`_: Support Figures nested in Cell contexts and Paragraphs in
  Figure contexts.
  (`#84 <https://github.com/HazyResearch/fonduer/pull/84>`_)

[0.2.2] - 2018-07-22
--------------------

.. note::
    Version 0.2.0 and 0.2.1 had to be skipped due to errors in uploading those
    versions to PyPi. Consequently, v0.2.2 is the version directly after
    v0.1.8.

.. warning::
    This release is NOT backwards compatable with v0.1.8. The code has now been
    refactored into submodules, where each submodule corresponds with a phase
    of the Fonduer pipeline. Consequently, you may need to adjust the paths
    of your imports from Fonduer.

* `@lukehsiao`_: Remove the futures imports, truly making Fonduer Python 3
  only. Also reorganize the codebase into submodules for each pipeline phase.
  (`#59 <https://github.com/HazyResearch/fonduer/pull/59>`_)
* `@lukehsiao`_: Split models and preprocessors into individual files.
  (`#60 <https://github.com/HazyResearch/fonduer/pull/60>`_,
  `#64 <https://github.com/HazyResearch/fonduer/pull/64>`_)
* `@senwu`_: Add branding, OSX tests.
  (`#61 <https://github.com/HazyResearch/fonduer/pull/61>`_,
  `#62 <https://github.com/HazyResearch/fonduer/pull/62>`_)
* `@lukehsiao`_: Rename to Phrase to Sentence.
  (`#72 <https://github.com/HazyResearch/fonduer/pull/72>`_)
* `@lukehsiao`_: Update the Data Model to include Caption, Section, Paragraph.
  (`#76 <https://github.com/HazyResearch/fonduer/pull/76>`_,
  `#77 <https://github.com/HazyResearch/fonduer/pull/77>`_,
  `#78 <https://github.com/HazyResearch/fonduer/pull/78>`_)
* `@senwu`_: Split up lf_helpers into separate files for each modality.
  (`#81 <https://github.com/HazyResearch/fonduer/pull/81>`_)
* A variety of small bugfixes and code cleanup.
  (`view milestone <https://github.com/HazyResearch/fonduer/milestone/8>`_)

[0.1.8] - 2018-06-01
--------------------

* `@senwu`_: Remove the Viewer, which is unused in Fonduer
  (`#55 <https://github.com/HazyResearch/fonduer/pull/55>`_)
* `@senwu`_: Fix SimpleTokenizer for lingual features are disabled
  (`#53 <https://github.com/HazyResearch/fonduer/pull/53>`_)
* `@prabh06`_: Extend styles parsing and add regex search
  (`#52 <https://github.com/HazyResearch/fonduer/pull/52>`_)
* `@lukehsiao`_: Remove unnecessary encoding in __repr__
  (`#50 <https://github.com/HazyResearch/fonduer/pull/50>`_)
* `@lukehsiao`_: Fix LocationMatch NER tags for spaCy
  (`#50 <https://github.com/HazyResearch/fonduer/pull/50>`_)

[0.1.7] - 2018-04-04
--------------------

.. warning::
    This release is NOT backwards compatable with v0.1.6. Specifically, the
    ``snorkel`` submodule in fonduer has been removed. Any previous imports of
    the form:

    .. code:: python

        from fonduer.snorkel._ import _

    Should drop the ``snorkel`` submodule:

    .. code:: python

        from fonduer._ import _

.. tip::
    To leverage the logging output of Fonduer, such as in a Jupyter Notebook,
    you can configure a logger in your application:

    .. code:: python

        import logging

        logging.basicConfig(stream=sys.stdout, format='[%(levelname)s] %(name)s - %(message)s')
        log = logging.getLogger('fonduer')
        log.setLevel(logging.INFO)


* `@lukehsiao`_: Remove SQLite code, switch to logging, and absorb snorkel
  codebase directly into the fonduer package for simplicity
  (`#44 <https://github.com/HazyResearch/fonduer/pull/44>`_)
* `@lukehsiao`_: Add lf_helpers to ReadTheDocs
  (`#42 <https://github.com/HazyResearch/fonduer/pull/42>`_)
* `@lukehsiao`_: Remove unused package dependencies
  (`#41 <https://github.com/HazyResearch/fonduer/pull/41>`_)

[0.1.6] - 2018-03-31
--------------------

* `@senwu`_: Fix support for providing a PostgreSQL username and password as
  part of the connection string provided to Meta.init()
  (`#40 <https://github.com/HazyResearch/fonduer/pull/40>`_)
* `@lukehsiao`_: Switch README from Markdown to reStructuredText

[0.1.5] - 2018-03-31
--------------------
.. warning::
    This release is NOT backwards compatable with v0.1.4. Specifically, in order
    to initialize a session with postgresql, you no longer do

    .. code:: python

        os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + DBNAME
        from fonduer import SnorkelSession
        session = SnorkelSession()

    which had the side-effects of manipulating your database tables on import
    (or creating a ``snorkel.db`` file if you forgot to set the environment
    variable). Now, you use the Meta class to initialize your session:

    .. code:: python

        from fonduer import Meta
        session = Meta.init("postgres://localhost:5432/" + DBNAME).Session()

    No side-effects occur until ``Meta`` is initialized.

* `@lukehsiao`_: Remove reliance on environment vars and remove side-effects of
  importing fonduer (`#36 <https://github.com/HazyResearch/fonduer/pull/36>`_)
* `@lukehsiao`_: Bring codebase in PEP8 compliance and add automatic code-style
  checks (`#37 <https://github.com/HazyResearch/fonduer/pull/37>`_)

[0.1.4] - 2018-03-30
--------------------

* `@lukehsiao`_: Separate tutorials into their own repo (`#31
  <https://github.com/HazyResearch/fonduer/pull/31>`_)

[0.1.3] - 2018-03-29
--------------------

Minor hotfix to the README formatting for PyPi.

[0.1.2] - 2018-03-29
--------------------

* `@lukehsiao`_: Deploy Fonduer to PyPi using Travis-CI


..
  For convenience, all username links for contributors can be listed here

.. _@lukehsiao: https://github.com/lukehsiao
.. _@senwu: https://github.com/senwu
.. _@prabh06: https://github.com/Prabh06
.. _@HiromuHota: https://github.com/HiromuHota
.. _@j-rausch: https://github.com/j-rausch
.. _@KenSugimoto: https://github.com/KenSugimoto
