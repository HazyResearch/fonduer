[Unreleased]
------------

Added
^^^^^
* `@j-rausch`_: Added unit tests for changed lingual parsing pipeline.
* `@senwu`_: Support Python 3.7.
* `@lukehsiao`_: Allow user to change featurization settings by providing
  ``.fonduer-config.yaml`` in their project.
* `@lukehsiao`_: Add a new Mention object, and have Candidate objects be
  composed of Mention objects, rather than directly of Spans. This allows a
  single Mention to be reused in multiple relations.
* `@lukehsiao`_: Improved connection-string validation for the Meta class.
* `@lukehsiao`_: Add unit test for Meta class.
* `@lukehsiao`_: Enforce flake8 checks on tests directory.

Changed
^^^^^^^
* `@j-rausch`_: Speed-up of ``spacy_parser``. We split the lingual parsing 
  pipeline into two stages. First, we parse structure and gather all 
  sentences for a document. Then, we merge and feed all sentences per
  document into the spacy NLP pipeline for more efficient processing.  
* `@j-rausch`_: Sentence splitting in lingual mode is now performed by 
  spacy's sentencizer instead of the depdency parser. This can lead to 
  variations in sentence segmentation and tokenization.
* `@j-rausch`_: Added ``language`` argument to ``Parser`` for specification
  of language used by ``spacy_parser``. E.g. ``language='en'```.
* `@senwu`_: Change weak supervision learning framework from numbskull to MeTaL.
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
* `@lukehsiao`_: Rename ``BatchLabelAnnotator`` to ``LabelAnnotator`` and
  ``BatchFeatureAnnotator`` to ``FeatureAnnotator``.
* `@lukehsiao`_: Organize documentation for ``data_model_utils`` by modality.
  (`#85 <https://github.com/HazyResearch/fonduer/pull/85>`_)
* `@lukehsiao`_: Change ``lf_helpers`` to ``data_model_utils``, since they can
  be applied more generally to throttlers or used for error analysis, and are
  not limited to just being used in labeling functions.
* `@lukehsiao`_: Update the CHANGELOG to start following `KeepAChangelog
  <https://keepachangelog.com/en/1.0.0/>`_ conventions.

Removed
^^^^^^^
* `@lukehsiao`_: Remove reduce option for UDFs, which were never used.
* `@lukehsiao`_: Remove get parent/children/sentence generator from Context.
  (`#87 <https://github.com/HazyResearch/fonduer/pull/87>`_)
* `@lukehsiao`_: Remove dependency on ``pdftotree``, which is currently unused.

Fixed
^^^^^
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
            [Part, Temp, Volt],
            [part_ngrams, temp_ngrams, volt_ngrams],
            [part_matcher, temp_matcher, volt_matcher],
        )
        mention_extractor.apply(docs, split=0, parallelism=PARALLEL)

        # Candidate Extraction
        PartTemp = candidate_subclass("PartTemp", [Part, Temp])
        PartVolt = candidate_subclass("PartVolt", [Part, Volt])

        candidate_extractor = CandidateExtractor(
            [PartTemp, PartVolt], throttlers=[temp_throttler, volt_throttler]
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
        from fonduer.features import FeatureAnnotator
        from fonduer.learning import GenerativeModel, SparseLogisticRegression
        from fonduer.parser import Parser
        from fonduer.parser.models import Document, Sentence
        from fonduer.parser.preprocessors import HTMLDocPreprocessor
        from fonduer.supervision import LabelAnnotator, load_gold_labels


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
* `@senwu`_: Split up lf_helpers into separate files for each modaility.
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
