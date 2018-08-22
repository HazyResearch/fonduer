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

Learning Utilities
------------------

These utilities can be used during error analysis to provide additional
insights.

.. automodule:: fonduer.learning.utils
    :members:

Configuration Settings
----------------------

Visit the `Configuring Fonduer`_ page to see how to provide configuration
parameters to Fonduer_ via ``.fonduer-config.yaml``.

The different learning parameters are explained in this section.

[TODO] give descriptions for the following::

    learning:
      LSTM:
        emb_dim: 100
        hidden_dim: 100
        attention: True
        dropout: 0.1
        bidirectional: True
        host_device: "CPU"
        max_sentence_length: 100

.. _Configuring Fonduer: config.html
.. _Fonduer: https://github.com/HazyResearch/fonduer
