Unreleased_
-----------

0.9.0_ - 2021-06-22
-------------------

Added
^^^^^
* `@HiromuHota`_: Support spaCy v2.3.
  (`#506 <https://github.com/HazyResearch/fonduer/pull/506>`_)
* `@HiromuHota`_: Add :class:`HOCRDocPreprocessor` and :class:`HocrVisualLinker`
  to support hOCR as input file.
  (`#476 <https://github.com/HazyResearch/fonduer/issues/476>`_)
  (`#519 <https://github.com/HazyResearch/fonduer/pull/519>`_)
* `@YasushiMiyata`_: Add multiline Japanese strings support to :class:`fonduer.parser.visual_parser.hocr_visual_parser`.
  (`#534 <https://github.com/HazyResearch/fonduer/issues/534>`_)
  (`#542 <https://github.com/HazyResearch/fonduer/pull/542>`_)
* `@YasushiMiyata`_: Add commit process immediately after add to :class:`fonduer.parser.Parser`.
  (`#494 <https://github.com/HazyResearch/fonduer/issues/494>`_)
  (`#544 <https://github.com/HazyResearch/fonduer/pull/544>`_)

Changed
^^^^^^^
* `@HiromuHota`_: Renamed :class:`VisualLinker` to :class:`PdfVisualParser`,
  which assumes the followings:
  (`#518 <https://github.com/HazyResearch/fonduer/pull/518>`_)

    * ``pdf_path`` should be a directory path, where PDF files exist, and cannot be a file path.
    * The PDF file should have the same basename (:class:`os.path.basename`) as the document.
      E.g., the PDF file should be either "123.pdf" or "123.PDF" for "123.html".

* `@HiromuHota`_: Changed :class:`Parser`'s signature as follows:
  (`#518 <https://github.com/HazyResearch/fonduer/pull/518>`_)

    * Renamed ``vizlink`` to ``visual_parser``.
    * Removed ``pdf_path``. Now this is required only by :class:`PdfVisualParser`.
    * Removed ``visual``. Provide ``visual_parser`` if visual information is to be parsed.

* `@YasushiMiyata`_: Changed :class:`UDFRunner`'s and :class:`UDF`'s data commit process as follows:
  (`#545 <https://github.com/HazyResearch/fonduer/pull/545>`_)

    * Removed ``add`` process on single-thread in :func:`_apply` in :class:`UDFRunner`.
    * Added ``UDFRunner._add`` of ``y`` on multi-threads to :class:`Parser`, :class:`Labeler` and :class:`Featurizer`.
    * Removed ``y`` of document parsed result from ``out_queue`` in :class:`UDF`.

Fixed
^^^^^
* `@YasushiMiyata`_: Fix test code `test_postgres.py::test_cand_gen_cascading_delete`.
  (`#538 <https://github.com/HazyResearch/fonduer/issues/538>`_)
  (`#539 <https://github.com/HazyResearch/fonduer/pull/539>`_)
* `@HiromuHota`_: Process the tail text only after child elements.
  (`#333 <https://github.com/HazyResearch/fonduer/issues/333>`_)
  (`#520 <https://github.com/HazyResearch/fonduer/pull/520>`_)

0.8.3_ - 2020-09-11
-------------------

Added
^^^^^
* `@YasushiMiyata`_: Add :func:`get_max_row_num` to ``fonduer.utils.data_model_utils.tabular``.
  (`#469 <https://github.com/HazyResearch/fonduer/issues/469>`_)
  (`#480 <https://github.com/HazyResearch/fonduer/pull/480>`_)
* `@HiromuHota`_: Add get_bbox() to :class:`Sentence` and :class:`SpanMention`.
  (`#429 <https://github.com/HazyResearch/fonduer/pull/429>`_)
* `@HiromuHota`_: Add a custom MLflow model that allows you to package a Fonduer model.
  See `here <../user/packaging.html>`_ for how to use it.
  (`#259 <https://github.com/HazyResearch/fonduer/issues/259>`_)
  (`#407 <https://github.com/HazyResearch/fonduer/pull/407>`_)
* `@HiromuHota`_: Support spaCy v2.2.
  (`#384 <https://github.com/HazyResearch/fonduer/issues/384>`_)
  (`#432 <https://github.com/HazyResearch/fonduer/pull/432>`_)
* `@wajdikhattel`_: Add multinary candidates.
  (`#455 <https://github.com/HazyResearch/fonduer/issues/455>`_)
  (`#456 <https://github.com/HazyResearch/fonduer/pull/456>`_)
* `@HiromuHota`_: Add ``nullables`` to :func:`candidate_subclass()` to allow NULL mention in a candidate.
  (`#496 <https://github.com/HazyResearch/fonduer/issues/496>`_)
  (`#497 <https://github.com/HazyResearch/fonduer/pull/497>`_)
* `@HiromuHota`_: Copy textual functions in :mod:`data_model_utils.tabular` to :mod:`data_model_utils.textual`.
  (`#503 <https://github.com/HazyResearch/fonduer/issues/503>`_)
  (`#505 <https://github.com/HazyResearch/fonduer/pull/505>`_)

Changed
^^^^^^^
* `@YasushiMiyata`_: Enable `RegexMatchSpan` with concatenates words by sep="(separator)" option.
  (`#270 <https://github.com/HazyResearch/fonduer/issues/270>`_)
  (`#492 <https://github.com/HazyResearch/fonduer/pull/492>`_)
* `@HiromuHota`_: Enabled "Type hints (PEP 484) support for the Sphinx autodoc extension."
  (`#421 <https://github.com/HazyResearch/fonduer/pull/421>`_)
* `@HiromuHota`_: Switched the Cython wrapper for Mecab from mecab-python3 to fugashi.
  Since the Japanese tokenizer remains the same, there should be no impact on users.
  (`#384 <https://github.com/HazyResearch/fonduer/issues/384>`_)
  (`#432 <https://github.com/HazyResearch/fonduer/pull/432>`_)
* `@HiromuHota`_: Log a stack trace on parsing error for better debug experience.
  (`#478 <https://github.com/HazyResearch/fonduer/issues/478>`_)
  (`#479 <https://github.com/HazyResearch/fonduer/pull/479>`_)
* `@HiromuHota`_: :func:`get_cell_ngrams` and :func:`get_neighbor_cell_ngrams` yield
  nothing when the mention is not tabular.
  (`#471 <https://github.com/HazyResearch/fonduer/issues/471>`_)
  (`#504 <https://github.com/HazyResearch/fonduer/pull/504>`_)

Deprecated
^^^^^^^^^^
* `@HiromuHota`_: Deprecated :func:`bbox_from_span` and :func:`bbox_from_sentence`.
  (`#429 <https://github.com/HazyResearch/fonduer/pull/429>`_)
* `@HiromuHota`_: Deprecated :func:`visualizer.get_box` in favor of :func:`span.get_bbox()`.
  (`#445 <https://github.com/HazyResearch/fonduer/issues/445>`_)
  (`#446 <https://github.com/HazyResearch/fonduer/pull/446>`_)
* `@HiromuHota`_: Deprecate textual functions in :mod:`data_model_utils.tabular`.
  (`#503 <https://github.com/HazyResearch/fonduer/issues/503>`_)
  (`#505 <https://github.com/HazyResearch/fonduer/pull/505>`_)

Fixed
^^^^^
* `@senwu`_: Fix pdf_path cannot be without a trailing slash.
  (`#442 <https://github.com/HazyResearch/fonduer/issues/442>`_)
  (`#459 <https://github.com/HazyResearch/fonduer/pull/459>`_)
* `@kaikun213`_: Fix bug in table range difference calculations.
  (`#420 <https://github.com/HazyResearch/fonduer/pull/420>`_)
* `@HiromuHota`_: mention_extractor.apply with clear=True now works even if it's not the first run.
  (`#424 <https://github.com/HazyResearch/fonduer/pull/424>`_)
* `@HiromuHota`_: Fix :func:`get_horz_ngrams` and :func:`get_vert_ngrams` so that they
  work even when the input mention is not tabular.
  (`#425 <https://github.com/HazyResearch/fonduer/issues/425>`_)
  (`#426 <https://github.com/HazyResearch/fonduer/pull/426>`_)
* `@HiromuHota`_: Fix the order of args to Bbox.
  (`#443 <https://github.com/HazyResearch/fonduer/issues/443>`_)
  (`#444 <https://github.com/HazyResearch/fonduer/pull/444>`_)
* `@HiromuHota`_: Fix the non-deterministic behavior in VisualLinker.
  (`#412 <https://github.com/HazyResearch/fonduer/issues/412>`_)
  (`#458 <https://github.com/HazyResearch/fonduer/pull/458>`_)
* `@HiromuHota`_: Fix an issue that the progress bar shows no progress on preprocessing
  by executing preprocessing and parsing in parallel.
  (`#439 <https://github.com/HazyResearch/fonduer/pull/439>`_)
* `@HiromuHota`_: Adopt to mlflow>=1.9.0.
  (`#461 <https://github.com/HazyResearch/fonduer/issues/461>`_)
  (`#463 <https://github.com/HazyResearch/fonduer/pull/463>`_)
* `@HiromuHota`_: Correct the entity type for NumberMatcher from "NUMBER" to "CARDINAL".
  (`#473 <https://github.com/HazyResearch/fonduer/issues/473>`_)
  (`#477 <https://github.com/HazyResearch/fonduer/pull/477>`_)
* `@HiromuHota`_: Fix :func:`_get_axis_ngrams` not to return ``None`` when the input is not tabular.
  (`#481 <https://github.com/HazyResearch/fonduer/pull/481>`_)
* `@HiromuHota`_: Fix :func:`Visualizer.display_candidates` not to draw rectangles on wrong pages.
  (`#488 <https://github.com/HazyResearch/fonduer/pull/488>`_)
* `@HiromuHota`_: Persist doc only when no error happens during parsing.
  (`#489 <https://github.com/HazyResearch/fonduer/issues/489>`_)
  (`#490 <https://github.com/HazyResearch/fonduer/pull/490>`_)

0.8.2_ - 2020-04-28
-------------------

Deprecated
^^^^^^^^^^

* `@HiromuHota`_: Use of undecorated labeling functions is deprecated and will not be supported as of v0.9.0.
  Please decorate them with ``snorkel.labeling.labeling_function``.

Fixed
^^^^^
* `@HiromuHota`_: Labeling functions can now be decorated with ``snorkel.labeling.labeling_function``.
  (`#400 <https://github.com/HazyResearch/fonduer/issues/400>`_)
  (`#401 <https://github.com/HazyResearch/fonduer/pull/401>`_)

0.8.1_ - 2020-04-13
-------------------

Added
^^^^^
* `@senwu`_: Add `mode` argument in create_task to support `STL` and `MTL`.

.. note::
    Fonduer has a new `mode` argument to support switching between different learning modes
    (e.g., STL or MLT). Example usage:

    .. code:: python

        # Create task for each relation.
        tasks = create_task(
            task_names = TASK_NAMES,
            n_arities = N_ARITIES,
            n_features = N_FEATURES,
            n_classes = N_CLASSES,
            emb_layer = EMB_LAYER,
            model="LogisticRegression",
            mode = MODE,
        )

0.8.0_ - 2020-04-07
-------------------

Changed
^^^^^^^
* `@senwu`_: Switch to Emmental as the default learning engine.

.. note::
    Rather than maintaining a separate learning engine, we switch to Emmental,
    a deep learning framework for multi-task learning. Switching to a more general
    learning framework allows Fonduer to support more applications and
    multi-task learning. Example usage:

    .. code:: python

        # With Emmental, you need do following steps to perform learning:
        # 1. Create task for each relations and EmmentalModel to learn those tasks.
        # 2. Wrap candidates into EmmentalDataLoader for training.
        # 3. Training and inference (prediction).

        import emmental

        # Collect word counter from candidates which is used in LSTM model.
        word_counter = collect_word_counter(train_cands)

        # Initialize Emmental. For customize Emmental, please check here:
        # https://emmental.readthedocs.io/en/latest/user/config.html
        emmental.init(fonduer.Meta.log_path)

        #######################################################################
        # 1. Create task for each relations and EmmentalModel to learn those tasks.
        #######################################################################

        # Generate special tokens which are used for LSTM model to locate mentions.
        # In LSTM model, we pad sentence with special tokens to help LSTM to learn
        # those mentions. Example:
        # Original sentence: Then Barack married Michelle.
        # ->  Then ~~[[1 Barack 1]]~~ married ~~[[2 Michelle 2]]~~.
        arity = 2
        special_tokens = []
        for i in range(arity):
            special_tokens += [f"~~[[{i}", f"{i}]]~~"]

        # Generate word embedding module for LSTM.
        emb_layer = EmbeddingModule(
            word_counter=word_counter, word_dim=300, specials=special_tokens
        )

        # Create task for each relation.
        tasks = create_task(
            ATTRIBUTE,
            2,
            F_train[0].shape[1],
            2,
            emb_layer,
            model="LogisticRegression",
        )

        # Create Emmental model to learn the tasks.
        model = EmmentalModel(name=f"{ATTRIBUTE}_task")

        # Add tasks into model
        for task in tasks:
            model.add_task(task)

        #######################################################################
        # 2. Wrap candidates into EmmentalDataLoader for training.
        #######################################################################

        # Here we only use the samples that have labels, which we filter out the
        # samples that don't have significant marginals.
        diffs = train_marginals.max(axis=1) - train_marginals.min(axis=1)
        train_idxs = np.where(diffs > 1e-6)[0]

        # Create a dataloader with weakly supervisied samples to learn the model.
        train_dataloader = EmmentalDataLoader(
            task_to_label_dict={ATTRIBUTE: "labels"},
            dataset=FonduerDataset(
                ATTRIBUTE,
                train_cands[0],
                F_train[0],
                emb_layer.word2id,
                train_marginals,
                train_idxs,
            ),
            split="train",
            batch_size=100,
            shuffle=True,
        )


        # Create test dataloader to do prediction.
        # Build test dataloader
        test_dataloader = EmmentalDataLoader(
            task_to_label_dict={ATTRIBUTE: "labels"},
            dataset=FonduerDataset(
                ATTRIBUTE, test_cands[0], F_test[0], emb_layer.word2id, 2
            ),
            split="test",
            batch_size=100,
            shuffle=False,
        )


        #######################################################################
        # 3. Training and inference (prediction).
        #######################################################################

        # Learning those tasks.
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, [train_dataloader])

        # Predict based the learned model.
        test_preds = model.predict(test_dataloader, return_preds=True)

* `@HiromuHota`_: Change ABSTAIN to -1 to be compatible with Snorkel of 0.9.X.
  Accordingly, user-defined labels should now be 0-indexed (used to be
  1-indexed).
  (`#310 <https://github.com/HazyResearch/fonduer/issues/310>`_)
  (`#320 <https://github.com/HazyResearch/fonduer/pull/320>`_)
* `@HiromuHota`_: Use executemany_mode="batch" instead of deprecated use_batch_mode=True.
  (`#358 <https://github.com/HazyResearch/fonduer/issues/358>`_)
* `@HiromuHota`_: Use tqdm.notebook.tqdm instead of deprecated tqdm.tqdm_notebook.
  (`#360 <https://github.com/HazyResearch/fonduer/issues/360>`_)
* `@HiromuHota`_: To support ImageMagick7, expand the version range of Wand.
  (`#373 <https://github.com/HazyResearch/fonduer/pull/373>`_)
* `@HiromuHota`_: Comply with PEP 561 for type-checking codes that use Fonduer.
* `@HiromuHota`_: Make UDF.apply of all child classes unaware of the database backend,
  meaning PostgreSQL is not required if UDF.apply is directly used instead of UDFRunner.apply.
  (`#316 <https://github.com/HazyResearch/fonduer/issues/316>`_)
  (`#368 <https://github.com/HazyResearch/fonduer/pull/368>`_)

Fixed
^^^^^
* `@senwu`_: Fix mention extraction to return mention classes instead of data model
  classes.

0.7.1_ - 2019-11-06
-------------------

Added
^^^^^
* `@senwu`_: Refactor `Featurization` to support user defined customized feature
  extractors and rename existing feature extractors' name to match the paper.

.. note::

    Rather than using a fixed multimodal feature library along, we have added an
    interface for users to provide customized feature extractors. Please see our
    full documentation for details.

    .. code:: python

        from fonduer.features import Featurizer, FeatureExtractor

        # Example feature extractor
        def feat_ext(candidates):
            for candidate in candidates:
                yield candidate.id, f"{candidate.id}", 1

        feature_extractors=FeatureExtractor(customize_feature_funcs=[feat_ext])
        featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    Rather than:

    .. code:: python

        from fonduer.features import Featurizer

        featurizer = Featurizer(session, [PartTemp])

* `@HiromuHota`_: Add page argument to get_pdf_dim in case pages have different dimensions.
* `@HiromuHota`_: Add Labeler#upsert_keys.
* `@HiromuHota`_: Add `vizlink` as an argument to `Parser` to be able to plug a custom visual linker.
  Unless otherwise specified, `VisualLinker` will be used by default.

.. note::

    Example usage:

    .. code:: python

        from fonduer.parser.visual_linker import VisualLinker
        class CustomVisualLinker(VisualLinker):
            def __init__(self):
                """Your code"""

            def link(self, document_name: str, sentences: Iterable[Sentence], pdf_path: str) -> Iterable[Sentence]:
                """Your code"""

            def is_linkable(self, filename: str) -> bool:
                """Your code"""

        from fonduer.parser import Parser
        parser = Parser(session, vizlink=CustomVisualLinker())

* `@HiromuHota`_: Add `LingualParser`, which any lingual parser like `Spacy` should inherit from,
  and add `lingual_parser` as an argument to `Parser` to be able to plug a custom lingual parser.
* `@HiromuHota`_: Annotate types to some of the classes incl. preprocesssors and parser/models.
* `@HiromuHota`_: Add table argument to ``Labeler.apply`` (and ``Labeler.update``), which can now be used to annotate gold labels.

.. note::

    Example usage:

    .. code:: python

        # Define a LF for gold labels
        def gold(c: Candidate) -> int:
            if some condition:
                return TRUE
            else:
                return FALSE

        labeler = Labeler(session, [PartTemp, PartVolt])
        # Annotate gold labels
        labeler.apply(docs=docs, lfs=[[gold], [gold]], table=GoldLabel, train=True)
        # A label matrix can be obtained using the name of annotator, "gold" in this case
        L_train_gold = labeler.get_gold_labels(train_cands, annotator="gold")
        # Annotate (noisy) labels
        labeler.apply(split=0, lfs=[[LF1, LF2, LF3], [LF4, LF5]], train=True)

    Note that the method name, "gold" in this example, is referred to as annotator.

Changed
^^^^^^^
* `@HiromuHota`_: Load a spaCy model if possible during `Spacy#__init__`.
* `@HiromuHota`_: Rename Spacy to SpacyParser.
* `@HiromuHota`_: Rename SimpleTokenizer into SimpleParser and let it inherit LingualParser.
* `@HiromuHota`_: Move all ligual parsers into lingual_parser folder.
* `@HiromuHota`_: Make load_lang_model private as a model is internally loaded during init.
* `@HiromuHota`_: Add a unit test for ``Parser`` with tabular=False.
  (`#261 <https://github.com/HazyResearch/fonduer/pull/261>`_)
* `@HiromuHota`_: Now ``longest_match_only`` of ``Union``, ``Intersect``, and ``Inverse`` override that of child matchers.
* `@HiromuHota`_: Use the official name "beautifulsoup4" instead of an alias "bs4".
  (`#306 <https://github.com/HazyResearch/fonduer/issues/306>`_)
* `@HiromuHota`_: Pin PyTorch on 1.1.0 to align with Snorkel of 0.9.X.
* `@HiromuHota`_: Depend on psycopg2 instead of psycopg2-binary as the latter is not recommended for production.
* `@HiromuHota`_: Change the default value for ``delim`` of ``SimpleParser`` from "<NB>" to ".".
  (`#272 <https://github.com/HazyResearch/fonduer/pull/272>`_)

Deprecated
^^^^^^^^^^
* `@HiromuHota`_: Classifier and its subclass disc_models are deprecated, and in v0.8.0 they will be removed.

Removed
^^^^^^^
* `@HiromuHota`_: Remove __repr__ from each mixin class as the referenced attributes are not available.
* `@HiromuHota`_: Remove the dependency on nltk, but ``PorterStemmer()`` can still be used,
  if it is provided as ``DictionaryMatch(stemmer=PorterStemmer())``.
* `@HiromuHota`_: Remove ``_NgramMatcher`` and ``_FigureMatcher`` as they are no longer needed.
* `@HiromuHota`_: Remove the dependency on Pandas and visual_linker._display_links.

Fixed
^^^^^
* `@senwu`_: Fix legacy code bug in ``SymbolTable``.
* `@HiromuHota`_: Fix the type of max_docs.
* `@HiromuHota`_: Associate sentence with section and paragraph no matter what tabular is.
  (`#261 <https://github.com/HazyResearch/fonduer/pull/261>`_)
* `@HiromuHota`_: Add a safeguard that prevents from accessing Meta.engine before it is assigned.
  Also this change allows creating a mention/candidate subclass even before Meta is initialized.
* `@HiromuHota`_: Create an Engine and open a connection in each child process.
  (`#323 <https://github.com/HazyResearch/fonduer/issues/323>`_)
* `@HiromuHota`_: Fix ``featurizer.apply(docs=train_docs)`` fails on clearing.
  (`#250 <https://github.com/HazyResearch/fonduer/issues/250>`_)
* `@HiromuHota`_: Correct abs_char_offsets to make it absolute.
  (`#332 <https://github.com/HazyResearch/fonduer/issues/332>`_)
* `@HiromuHota`_: Fix deadlock error during Labeler.apply and Featurizer.apply.
  (`#328 <https://github.com/HazyResearch/fonduer/issues/328>`_)
* `@HiromuHota`_: Avoid networkx 2.4 so that snorkel-metal does not use the removed API.
* `@HiromuHota`_: Fix the issue that Labeler.apply with docs instead of split fails.
  (`#340 <https://github.com/HazyResearch/fonduer/pull/340>`_)
* `@HiromuHota`_: Make mention/candidate_subclasses and their objects picklable.
* `@HiromuHota`_: Make Visualizer#display_candidates mention-type argnostic.
* `@HiromuHota`_: Ensure labels get updated when LFs are updated.
  (`#336 <https://github.com/HazyResearch/fonduer/issues/336>`_)

0.7.0_ - 2019-06-12
-------------------

Added
^^^^^
* `@HiromuHota`_: Add notes about the current implementation of data models.
* `@HiromuHota`_: Add Featurizer#upsert_keys.
* `@HiromuHota`_: Update the doc for OS X about an external dependency on libomp.
* `@HiromuHota`_: Add test_classifier.py to unit test Classifier and its subclasses.
* `@senwu`_: Add test_simple_tokenizer.py to unit test simple_tokenizer.
* `@HiromuHota`_: Add test_spacy_parser.py to unit test spacy_parser.

Changed
^^^^^^^
* `@HiromuHota`_: Assign a section for mention spaces.
* `@HiromuHota`_: Incorporate entity_confusion_matrix as a first-class citizen and
  rename it to confusion_matrix because it can be used both entity-level
  and mention-level.
* `@HiromuHota`_: Separate Spacy#_split_sentences_by_char_limit to test itself.
* `@HiromuHota`_: Refactor the custom sentence_boundary_detector for readability
  and efficiency.
* `@HiromuHota`_: Remove a redundant argument, document, from Spacy#split_sentences.
* `@HiromuHota`_: Refactor TokenPreservingTokenizer for readability.

Removed
^^^^^^^
* `@HiromuHota`_: Remove ``data_model_utils.tabular.same_document``, which
  always returns True because a candidate can only have mentions from the same
  document under the current implemention of ``CandidateExtractorUDF``.

Fixed
^^^^^
* `@senwu`_: Fix the doc about the PostgreSQL version requirement.

0.6.2_ - 2019-04-01
-------------------

Fixed
^^^^^
* `@lukehsiao`_: Fix Meta initialization bug which would configure logging
  upon import rather than allowing the user to configure logging themselves.

0.6.1_ - 2019-03-29
-------------------

Added
^^^^^
* `@senwu`_: update the spacy version to v2.1.x.
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

    which will create logs within the ``log_folder`` directory. If logging is
    not explicitly initialized, we will provide a default configuration which
    will store logs in a temporary directory.

Changed
^^^^^^^
* `@senwu`_: Update the whole logging strategy.

.. note::
    For the whole logging strategy:

    With this change, the running log is stored ``fonduer.log`` in the
    ``{fonduer.Meta.log_path}/{datetime}`` folder. User can specify it
    using ``fonduer.init_logging()``. It also contains the learning logs init.

    For learning logging strategy:

    Previously, the model checkpoints are stored in the user provided folder
    by ``save_dir`` and the name for checkpoint is
    ``{model_name}.mdl.ckpt.{global_step}``.

    With this change, the model is saved in the subfolder of the same folder
    ``fonduer.Meta.log_path`` with log file file. Each learning run creates a
    subfolder under name ``{datetime}_{model_name}`` with all model checkpoints
    and tensorboard log file init. To use the tensorboard to check the learning
    curve, run ``tensorboard --logdir LOG_FOLDER``.

Fixed
^^^^^
* `@senwu`_: Change the exception condition to make sure parser run end to end.
* `@lukehsiao`_: Fix parser error when text was located in the ``tail`` of an
  LXML table node..
* `@HiromuHota`_: Store lemmas and pos_tags in case they are returned from a
  tokenizer.
* `@HiromuHota`_: Use unidic instead of ipadic for Japanese.
  (`#231 <https://github.com/HazyResearch/fonduer/issues/231>`_)
* `@senwu`_: Use mecab-python3 version 0.7 for Japanese tokenization since
  spaCy only support version 0.7.
* `@HiromuHota`_: Use black 18.9b0 or higher to be consistent with isort.
  (`#225 <https://github.com/HazyResearch/fonduer/issues/225>`_)
* `@HiromuHota`_: Workaround no longer required for Japanese as of spaCy v2.1.0.
  (`#224 <https://github.com/HazyResearch/fonduer/pull/224>`_)
* `@senwu`_: Update the metal version.
* `@senwu`_: Expose the ``b`` and ``pos_label`` in training.
* `@senwu`_: Fix the issue that pdfinfo causes parsing error when it contains
  more than one ``Page``.

0.6.0_ - 2019-02-17
-------------------

Changed
^^^^^^^
* `@lukehsiao`_: improved performance of ``data_model_utils`` through caching
  and simplifying the underlying queries.
  (`#212 <https://github.com/HazyResearch/fonduer/pull/212>`_,
  `#215 <https://github.com/HazyResearch/fonduer/pull/215>`_)
* `@senwu`_: upgrade to PyTorch v1.0.0.
  (`#209 <https://github.com/HazyResearch/fonduer/pull/209>`_)

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

Fixed
^^^^^
* `@senwu`_: Improve type checking in featurization.
* `@lukehsiao`_: Fixed sentence.sentence_num bug in get_neighbor_sentence_ngrams.
* `@lukehsiao`_: Add session synchronization to sqlalchemy delete queries.
  (`#214 <https://github.com/HazyResearch/fonduer/pull/214>`_)
* `@lukehsiao`_: Update PyYAML dependency to patch CVE-2017-18342.
  (`#205 <https://github.com/HazyResearch/fonduer/pull/205>`_)
* `@KenSugimoto`_: Fix max/min in ``visualizer.get_box``

0.5.0_ - 2019-01-01
-------------------

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

Fixed
^^^^^
* `@HiromuHota`_: Modify docstring of functions that return get_sparse_matrix
* `@lukehsiao`_: Fix the behavior of ``get_last_documents`` to return Documents
  that are correctly linked to the database and can be navigated by the user.
  (`#201 <https://github.com/HazyResearch/fonduer/pull/201>`_)
* `@lukehsiao`_: Fix the behavior of MentionExtractor ``clear`` and
  ``clear_all`` to also delete the Candidates that correspond to the Mentions.

0.4.1_ - 2018-12-12
-------------------

Added
^^^^^
* `@senwu`_: Added alpha spacy support for Chinese tokenizer.

Changed
^^^^^^^
* `@lukehsiao`_: Add soft version pinning to avoid failures due to dependency
  API changes.
* `@j-rausch`_: Change ``get_row_ngrams`` and ``get_col_ngrams`` to return
  ``None`` if the passed ``Mention`` argument is not inside a table.
  (`#194 <https://github.com/HazyResearch/fonduer/pull/194>`_)

Fixed
^^^^^
* `@senwu`_: fix non-deterministic issue from get_candidates and get_mentions
  by parallel candidate/mention generation.

0.4.0_ - 2018-11-27
-------------------

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

0.3.6_ - 2018-11-15
-------------------

Fixed
^^^^^
* `@lukehsiao`_: Updated snorkel-metal version requirement to ensure new syntax
  works when a user upgrades Fonduer.
* `@lukehsiao`_: Improve error messages on PostgreSQL connection and update FAQ.

0.3.5_ - 2018-11-04
-------------------

Added
^^^^^
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

0.3.4_ - 2018-10-17
-------------------

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

0.3.3_ - 2018-09-27
-------------------

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

0.3.2_ - 2018-09-20
-------------------

Changed
^^^^^^^
* `@lukehsiao`_: :class:`MentionNgrams` ``split_tokens`` now defaults to an
  empty list and splits on all occurrences, rather than just the first
  occurrence.
* `@j-rausch`_: Parser will now skip documents with parsing errors rather than
  crashing.

Fixed
^^^^^
* `@lukehsiao`_: Fix attribute error when using MentionFigures.

0.3.1_ - 2018-09-18
-------------------

Fixed
^^^^^
* `@lukehsiao`_: Fix the layers module in fonduer.learning.disc_models.layers.

0.3.0_ - 2018-09-18
-------------------

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

0.2.3_ - 2018-07-23
-------------------

Added
^^^^^
* `@lukehsiao`_: Support Figures nested in Cell contexts and Paragraphs in
  Figure contexts.
  (`#84 <https://github.com/HazyResearch/fonduer/pull/84>`_)

0.2.2_ - 2018-07-22
-------------------

.. note::
    Version 0.2.0 and 0.2.1 had to be skipped due to errors in uploading those
    versions to PyPi. Consequently, v0.2.2 is the version directly after
    v0.1.8.

.. warning::
    This release is NOT backwards compatable with v0.1.8. The code has now been
    refactored into submodules, where each submodule corresponds with a phase
    of the Fonduer pipeline. Consequently, you may need to adjust the paths
    of your imports from Fonduer.

Added
^^^^^
* `@senwu`_: Add branding, OSX tests.
  (`#61 <https://github.com/HazyResearch/fonduer/pull/61>`_,
  `#62 <https://github.com/HazyResearch/fonduer/pull/62>`_)
* `@lukehsiao`_: Update the Data Model to include Caption, Section, Paragraph.
  (`#76 <https://github.com/HazyResearch/fonduer/pull/76>`_,
  `#77 <https://github.com/HazyResearch/fonduer/pull/77>`_,
  `#78 <https://github.com/HazyResearch/fonduer/pull/78>`_)

Changed
^^^^^^^
* `@senwu`_: Split up lf_helpers into separate files for each modality.
  (`#81 <https://github.com/HazyResearch/fonduer/pull/81>`_)
* `@lukehsiao`_: Rename to Phrase to Sentence.
  (`#72 <https://github.com/HazyResearch/fonduer/pull/72>`_)
* `@lukehsiao`_: Split models and preprocessors into individual files.
  (`#60 <https://github.com/HazyResearch/fonduer/pull/60>`_,
  `#64 <https://github.com/HazyResearch/fonduer/pull/64>`_)

Removed
^^^^^^^
* `@lukehsiao`_: Remove the futures imports, truly making Fonduer Python 3
  only. Also reorganize the codebase into submodules for each pipeline phase.
  (`#59 <https://github.com/HazyResearch/fonduer/pull/59>`_)

Fixed
^^^^^
* A variety of small bugfixes and code cleanup.
  (`view milestone <https://github.com/HazyResearch/fonduer/milestone/8>`_)

0.1.8_ - 2018-06-01
-------------------

Added
^^^^^
* `@prabh06`_: Extend styles parsing and add regex search
  (`#52 <https://github.com/HazyResearch/fonduer/pull/52>`_)

Removed
^^^^^^^
* `@senwu`_: Remove the Viewer, which is unused in Fonduer
  (`#55 <https://github.com/HazyResearch/fonduer/pull/55>`_)
* `@lukehsiao`_: Remove unnecessary encoding in __repr__
  (`#50 <https://github.com/HazyResearch/fonduer/pull/50>`_)

Fixed
^^^^^
* `@senwu`_: Fix SimpleTokenizer for lingual features are disabled
  (`#53 <https://github.com/HazyResearch/fonduer/pull/53>`_)
* `@lukehsiao`_: Fix LocationMatch NER tags for spaCy
  (`#50 <https://github.com/HazyResearch/fonduer/pull/50>`_)

0.1.7_ - 2018-04-04
-------------------

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

Added
^^^^^
* `@lukehsiao`_: Add lf_helpers to ReadTheDocs
  (`#42 <https://github.com/HazyResearch/fonduer/pull/42>`_)

Removed
^^^^^^^
* `@lukehsiao`_: Remove SQLite code, switch to logging, and absorb snorkel
  codebase directly into the fonduer package for simplicity
  (`#44 <https://github.com/HazyResearch/fonduer/pull/44>`_)
* `@lukehsiao`_: Remove unused package dependencies
  (`#41 <https://github.com/HazyResearch/fonduer/pull/41>`_)

0.1.6_ - 2018-03-31
-------------------

Changed
^^^^^^^
* `@lukehsiao`_: Switch README from Markdown to reStructuredText

Fixed
^^^^^
* `@senwu`_: Fix support for providing a PostgreSQL username and password as
  part of the connection string provided to Meta.init()
  (`#40 <https://github.com/HazyResearch/fonduer/pull/40>`_)

0.1.5_ - 2018-03-31
-------------------
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

Removed
^^^^^^^
* `@lukehsiao`_: Remove reliance on environment vars and remove side-effects of
  importing fonduer (`#36 <https://github.com/HazyResearch/fonduer/pull/36>`_)

Fixed
^^^^^
* `@lukehsiao`_: Bring codebase in PEP8 compliance and add automatic code-style
  checks (`#37 <https://github.com/HazyResearch/fonduer/pull/37>`_)

0.1.4_ - 2018-03-30
-------------------

Changed
^^^^^^^
* `@lukehsiao`_: Separate tutorials into their own repo (`#31
  <https://github.com/HazyResearch/fonduer/pull/31>`_)

0.1.3_ - 2018-03-29
-------------------

Fixed
^^^^^
Minor hotfix to the README formatting for PyPi.

0.1.2_ - 2018-03-29
-------------------

Added
^^^^^
* `@lukehsiao`_: Deploy Fonduer to PyPi using Travis-CI

.. _Unreleased: https://github.com/hazyresearch/fonduer/compare/v0.9.0...master
.. _0.9.0: https://github.com/hazyresearch/fonduer/compare/v0.8.3...v0.9.0
.. _0.8.3: https://github.com/hazyresearch/fonduer/compare/v0.8.2...v0.8.3
.. _0.8.2: https://github.com/hazyresearch/fonduer/compare/v0.8.1...v0.8.2
.. _0.8.1: https://github.com/hazyresearch/fonduer/compare/v0.8.0...v0.8.1
.. _0.8.0: https://github.com/hazyresearch/fonduer/compare/v0.7.1...v0.8.0
.. _0.7.1: https://github.com/hazyresearch/fonduer/compare/v0.7.0...v0.7.1
.. _0.7.0: https://github.com/hazyresearch/fonduer/compare/v0.6.2...v0.7.0
.. _0.6.2: https://github.com/hazyresearch/fonduer/compare/v0.6.1...v0.6.2
.. _0.6.1: https://github.com/hazyresearch/fonduer/compare/v0.6.0...v0.6.1
.. _0.6.0: https://github.com/hazyresearch/fonduer/compare/v0.5.0...v0.6.0
.. _0.5.0: https://github.com/hazyresearch/fonduer/compare/v0.4.1...v0.5.0
.. _0.4.1: https://github.com/hazyresearch/fonduer/compare/v0.4.0...v0.4.1
.. _0.4.0: https://github.com/hazyresearch/fonduer/compare/v0.3.6...v0.4.0
.. _0.3.6: https://github.com/hazyresearch/fonduer/compare/v0.3.5...v0.3.6
.. _0.3.5: https://github.com/hazyresearch/fonduer/compare/v0.3.4...v0.3.5
.. _0.3.4: https://github.com/hazyresearch/fonduer/compare/v0.3.3...v0.3.4
.. _0.3.3: https://github.com/hazyresearch/fonduer/compare/v0.3.2...v0.3.3
.. _0.3.2: https://github.com/hazyresearch/fonduer/compare/v0.3.1...v0.3.2
.. _0.3.1: https://github.com/hazyresearch/fonduer/compare/v0.3.0...v0.3.1
.. _0.3.0: https://github.com/hazyresearch/fonduer/compare/v0.2.3...v0.3.0
.. _0.2.3: https://github.com/hazyresearch/fonduer/compare/v0.2.2...v0.2.3
.. _0.2.2: https://github.com/hazyresearch/fonduer/compare/v0.1.8...v0.2.2
.. _0.1.8: https://github.com/hazyresearch/fonduer/compare/v0.1.7...v0.1.8
.. _0.1.7: https://github.com/hazyresearch/fonduer/compare/v0.1.6...v0.1.7
.. _0.1.6: https://github.com/hazyresearch/fonduer/compare/v0.1.5...v0.1.6
.. _0.1.5: https://github.com/hazyresearch/fonduer/compare/v0.1.4...v0.1.5
.. _0.1.4: https://github.com/hazyresearch/fonduer/compare/v0.1.3...v0.1.4
.. _0.1.3: https://github.com/hazyresearch/fonduer/compare/v0.1.2...v0.1.3
.. _0.1.2: https://github.com/hazyresearch/fonduer/releases/tag/v0.1.2

..
  For convenience, all username links for contributors can be listed here

.. _@YasushiMiyata: https://github.com/YasushiMiyata
.. _@HiromuHota: https://github.com/HiromuHota
.. _@KenSugimoto: https://github.com/KenSugimoto
.. _@j-rausch: https://github.com/j-rausch
.. _@kaikun213: https://github.com/kaikun213
.. _@lukehsiao: https://github.com/lukehsiao
.. _@prabh06: https://github.com/Prabh06
.. _@senwu: https://github.com/senwu
.. _@wajdikhattel: https://github.com/wajdikhattel
