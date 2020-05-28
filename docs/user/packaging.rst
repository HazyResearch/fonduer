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
        └── model.pkl  # the pickled Fonduer pipeline model.

This directory, conformal to the `MLflow Model`_, can be deployed either locally or on the cloud using MLflow's built-in deployment tools.

Example
-------

Currently, two types of classifiers are supported: :class:`EmmentalModel` (aka discriminative model) and :class:`LabelModel` (aka generative model).
The following example shows how to package a Fonduer pipeline model that uses :class:`EmmentalModel` as a classifier.

    >>> from fonduer.parser.preprocessors import HTMLDocPreprocessor
    >>> preprocessor = HTMLDocPreprocessor(docs_path)
    >>> ...
    >>> # Initialize other components like parser, mention_extractor.
    >>> # Then, train them if they can be trained, e.g., featurizer, emmental_model.
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

    .. code-block:: python
        :caption: my_fonduer_model.py

        import my_subclasses

        class MyFonduerModel(FonduerModel):
            def _classify(self, doc: Document) -> DataFrame:
                # My implementation

If you are familiar with MLflow, you may wonder why the `/code` directory should include these Python files.

To successfully package a Fonduer model, there are things to understand:

1. `my_fonduer_model.py` should import `my_subclasses`.
2. Depending on your application, some other python files may need to be included `code_paths`.


.. _MLflow Model: https://www.mlflow.org/docs/latest/models.html

MLflow model for Fonduer
------------------------

.. automodule:: fonduer.packaging.fonduer_model
    :members:
    :inherited-members:
    :show-inheritance:
