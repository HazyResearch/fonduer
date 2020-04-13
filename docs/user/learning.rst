Learning
========
The final stage of Fonduer_'s pipeline is to use machine learning models to
model the noise between supervision sources to generate probabilistic labels as
training data, and then classify each Candidate. Rather than maintaining a separate
learning engine, we switch to Emmental_, a deep learning framework for multi-task
learning. Switching to a more general learning framework allows Fonduer_ to
support more applications and multi-task learning. With Emmental, you need do
following steps to perform learning:

  #. Create task for each relations and EmmentalModel to learn those tasks.
  #. Wrap candidates into EmmentalDataLoader for training.
  #. Training and inference (prediction).


Core Learning Objects
---------------------

These are Fonduer_'s core objects used for learning. First, we describe how to
create Emmental task for each relation.

.. automodule:: fonduer.learning.task
    :members:
    :inherited-members:
    :show-inheritance:

Then, we describe how to wrap candidates into an EmmentalDataLoader.

.. automodule:: fonduer.learning.dataset
    :members:
    :inherited-members:
    :show-inheritance:

Learning Utilities
------------------

These utilities can be used during error analysis to provide additional
insights.

.. automodule:: fonduer.learning.utils
    :members:
    :inherited-members:
    :show-inheritance:

Configuration Settings
----------------------

Visit the `Configuring Fonduer`_ page to see how to provide configuration
parameters to Fonduer_ via ``.fonduer-config.yaml``.

The learning parameters of different models are described below::

    learning:
      # LSTM model
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
      # Logistic Regression model
      LogisticRegression:
        # The number of features in the LogisticRegression hidden state
        hidden_dim: 100
        # bias term
        bias: False

.. _Configuring Fonduer: config.html
.. _Fonduer: https://github.com/HazyResearch/fonduer
.. _Emmental: https://github.com/SenWu/emmental
