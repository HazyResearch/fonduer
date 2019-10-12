Configuring Fonduer
===================

By default, Fonduer_ looks for ``.fonduer-config.yaml`` starting from the
current working directory, allowing you to have multiple configuration files
for different directories or projects. If it's not there, it looks in parent
directories. If no file is found, a default configuration will be used.

Fonduer will only ever use one ``.fonduer-config.yaml`` file. It does not look
for multiple files and will not compose configuration settings from different
files.

The default ``.fonduer-config.yaml`` configuration file is shown below::

    featurization:
      textual:
        window_feature:
          size: 3
          combinations: True
          isolated: True
        word_feature:
          window: 7
      tabular:
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

    learning:
      LSTM:
        emb_dim: 100
        hidden_dim: 100
        attention: True
        dropout: 0.1
        bidirectional: True
        host_device: "CPU"
        max_sentence_length: 100
      LogisticRegression:
        bias: False
      SparseLSTM:
        emb_dim: 100
        hidden_dim: 100
        attention: True
        dropout: 0.1
        bidirectional: True
        host-device: "CPU"
        max_sentence_length: 100
        bias: False
      SparseLogisticRegression:
        bias: False

.. _Fonduer: https://github.com/HazyResearch/fonduer
