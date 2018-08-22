Multimodal Featurization
========================
The third stage of Fonduer_'s pipeline is to featurize each Candidate with
multimodal features.

Feature Model Classes
---------------------

The following describes the Feature element.

.. automodule:: fonduer.features.models
    :members:

Core Objects
------------

These are Fonduer_'s core objects used for featurization.

.. automodule:: fonduer.features
    :members:

Configuration Settings
----------------------

By default, Fonduer looks for ``.fonduer-config.yaml`` starting from the
current working directory, allowing you to have multiple configuration files
for different directories or projects. If it's not there, it looks in parent
directories. If no file is found, a default configuration will be used.

Fonduer will only ever use one ``.fonduer-config.yaml`` file. It does not look
for multiple files and will not compose configuration settings from different
files. Thus, all configuration options MUST be specified in the config file.

The default configuration for featurization is shown below::

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
