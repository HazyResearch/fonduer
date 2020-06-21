Multimodal Featurization
========================
The third stage of Fonduer_'s pipeline is to featurize each Candidate with
multimodal features.

Feature Model Classes
---------------------

The following describes the Feature element.

.. automodule:: fonduer.features.models
    :members:
    :inherited-members:
    :show-inheritance:

Core Objects
------------

These are Fonduer_'s core objects used for featurization.

.. automodule:: fonduer.features
    :members:
    :inherited-members:
    :show-inheritance:

Multimodal features
-------------------

Fonduer_ includes a basic multimodal feature library based on its rich data model.
In addition, users can provide their own feature extractors to use with their
applications.

.. automodule:: fonduer.features.feature_libs
    :members:
    :inherited-members:
    :show-inheritance:

Configuration Settings
----------------------

Visit the `Configuring Fonduer`_ page to see how to provide configuration
parameters to Fonduer_ via ``.fonduer-config.yaml``.

The different featurization parameters are explained in this section::

    featurization:
      # settings of textual-based features
      textual:
        # settings for window features
        window_feature:
          size: 3
          combinations: True
          isolated: True
        # settings for word window usd to extract features from surrounding words
        word_feature:
          window: 7
      # settings of tabular-based features
      tabular:
        # unary feture settings
        unary_features:
          # type of attributes
          attrib:
            - words
          # number of gram for features extract in cells
          get_cell_ngrams:
            max: 2
          # number of gram for features extract in headers
          get_head_ngrams:
            max: 2
          # number of gram for features extract in rows
          get_row_ngrams:
            max: 2
          # number of gram for features extract in columns
          get_col_ngrams:
            max: 2
        # binary feature settings
        multinary_features:
          # minimal difference in rows to check
          min_row_diff:
            absolute: False
          # minimal difference in cols to check
          min_col_diff:
            absolute: False

.. _Fonduer: https://github.com/HazyResearch/fonduer
.. _Configuring Fonduer: config.html
