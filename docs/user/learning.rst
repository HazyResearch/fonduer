Learning
========
The final stage of Fonduer_'s pipeline is to use machine learning models to
model the noise between supervision sources to generate probabilistic labels as
training data, and then classify each Candidate.

Core Learning Objects
---------------------

These are Fonduer_'s core objects used for learning.

.. automodule:: fonduer.learning
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :special-members: __init__

Learning Utilities
------------------

These utilities can be used during error analysis to provide additional
insights.

.. automodule:: fonduer.learning.utils
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :special-members: __init__

Configuration Settings
----------------------

Visit the `Configuring Fonduer`_ page to see how to provide configuration
parameters to Fonduer_ via ``.fonduer-config.yaml``.

The different learning parameters are described below::

    learning:
      LSTM:
        # Word embedding dimension size
        emb_dim: 100
        # The number of features in the LSTM hidden state
        hidden_dim: 100
        # Use attention or not (Options: True or False)
        attention: True
        # Dropout parameter
        dropout: 0.1
        # Use bidirectional LSTM or not (Options: True or False)
        bidirectional: True
        # Prefered host device (Options: CPU or GPU)
        host_device: "CPU"
        # Maximum sentence length of LSTM input
        max_sentence_length: 100

.. _Configuring Fonduer: config.html
.. _Fonduer: https://github.com/HazyResearch/fonduer
