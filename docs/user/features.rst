Multimodal Featurization
========================
The third stage of Fonduer_'s pipeline is to featurize each Candidate with
multimodal features.

Feature Model Classes
---------------------

The following describes the Feature element.

.. automodule:: fonduer.features.models
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Core Objects
------------

These are Fonduer_'s core objects used for featurization.

.. automodule:: fonduer.features
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Configuration Settings
----------------------

Visit the `Configuring Fonduer`_ page to see how to provide configuration
parameters to Fonduer_ via ``.fonduer-config.yaml``.

The different featurization parameters are explained in this section.

[TODO] give descriptions for the following::

    featurization:
      content:
        window_feature:
          size: 3
          combinations: True
          isolated: True
        word_feature:
          window: 7
      table:
        unary_features:
          attrib:
            - words
          get_cell_ngrams:
            max: 2
          get_head_ngrams:
            max: 2
          get_row_ngrams:
            max: 2
          get_col_ngrams:
            max: 2
        binary_features:
          min_row_diff:
            absolute: False
          min_col_diff:
            absolute: False

.. _Fonduer: https://github.com/HazyResearch/fonduer
.. _Configuring Fonduer: config.html
