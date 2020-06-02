Packaging
=========

You can package a whole trained Fonduer pipeline model (parsing, extraction, featurization, and classification) and deploy it to a remote place to serve.
To this end, we use `MLflow Model`_ as a storage format.
A packaged Fonduer pipeline model (or simply referred to as a Fonduer model) looks like this:

    .. code-block::
        :caption: Directory written by fonduer.packaging.save_model

        fonduer_model/
        ├── MLmodel
        ├── code
        │   ├── my_subclasses.py
        │   └── my_fonduer_model.py
        ├── conda.yaml
        ├── candidate_classes.pkl
        ├── mention_classes.pkl
        └── model.pkl  # the pickled Fonduer pipeline model.

Currently, two types of classifiers are supported: :class:`EmmentalModel` (aka discriminative model) and :class:`LabelModel` (aka generative model).
The following example shows how to package a Fonduer pipeline model that uses :class:`EmmentalModel` as a classifier.

Example
-------

First, create a class that inherits :class:`FonduerModel` and implements :func:`_classify`.
You can see fully functional examples of such a class at hardware_fonduer_model.py_ and my_fonduer_model.py_.
Then, put this class in a Python module like `my_fonduer_model.py` instead of in a Jupyter notebook or a Python script as this module will be packaged.

    .. code-block:: python
        :caption: my_fonduer_model.py

        class MyFonduerModel(FonduerModel):
            def _classify(self, doc: Document) -> DataFrame:
                # Assume only one candidate class is used.
                candidate_class = self.candidate_extractor.candidate_classes[0]
                # Get a list of candidates for this candidate_class.
                test_cands = getattr(doc, candidate_class.__tablename__ + "s")
                # Get a list of true predictions out of candidates.
                ...
                true_preds = [test_cands[_] for _ in positive[0]]

                # Load the true predictions into a dataframe.
                df = DataFrame()
                for true_pred in true_preds:
                    entity_relation = tuple(m.context.get_span() for m in true_pred.get_mentions())
                    df = df.append(
                        DataFrame([entity_relation],
                        columns=[m.__name__ for m in candidate_class.mentions]
                        )
                    )
                return df

Similarly, put anything that is required for :class:`MentionExtractor` and :class:`CandidateExtractor`, i.e., mention_classes, mention_spaces, matchers, candidate_classes, and throttlers, into another module.

    .. code-block:: python
        :caption: my_subclasses.py

        from fonduer.candidates.models import mention_subclass
        Presidentname = mention_subclass("Presidentname")
        Placeofbirth = mention_subclass("Placeofbirth")

        mention_classes = [Presidentname, Placeofbirth]
        ...
        mention_spaces = [presname_ngrams, placeofbirth_ngrams]
        matchers = [president_name_matcher, place_of_birth_matcher]
        candidate_classes = [PresidentnamePlaceofbirth]
        throttlers = [my_throttler]

Finally, in a Jupyter notebook or a Python script, build and train a pipeline, then save the trained pipeline.

    >>> from fonduer.parser.preprocessors import HTMLDocPreprocessor
    >>> preprocessor = HTMLDocPreprocessor(docs_path)
    >>> ...
    # Import mention_classes, candidate_classes, etc. from my_subclasses.py
    # instead of defining them here.
    >>> from my_subclasses import mention_classes, mention_spaces, matchers
    >>> mention_extractor = MentionExtractor(session, mention_classes, mention_spaces, matchers)
    >>> from my_subclasses import candidate_classes, throttlers
    >>> candidate_extractor = CandidateExtractor(session, candidate_classes, throttlers)
    >>> ...
    >>> from my_fonduer_model import MyFonduerModel
    >>> from fonduer.packaging import save_model
    >>> save_model(
            fonduer_model=MyFonduerModel(),
            path="fonduer_model",
            code_paths=["my_subclasses.py", "my_fonduer_model.py"],
            preprocessor=preprocessor,
            parser=parser,
            mention_extractor=mention_extractor,
            candidate_extractor=candidate_extractor,
            featurizer=featurizer,
            emmental_model=emmental_model,
            word2id=emb_layer.word2id,
        )

Remember to list `my_subclasses.py` and `my_fonduer_model.py` in the ``code_paths`` argument.
Other modules can also be listed if they are required during inference.
Alternatively, you can manually place arbitrary modules or data under `/code` or `/data` directory, respectively.
For further information about MLflow Model, please see `MLflow Model`_.

.. _MLflow Model: https://www.mlflow.org/docs/latest/models.html
.. _hardware_fonduer_model.py: https://github.com/HazyResearch/fonduer/blob/master/tests/shared/hardware_fonduer_model.py
.. _my_fonduer_model.py: https://github.com/HiromuHota/fonduer-mlflow/blob/master/my_fonduer_model.py

MLflow model for Fonduer
------------------------

.. automodule:: fonduer.packaging.fonduer_model
    :members:
    :inherited-members:
    :show-inheritance:
