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

Some of the multimodal features used by Fonduer_ can be configured by editing
``fonduer/features/settings.json``. The default configuration is shown below.

.. literalinclude:: ../../fonduer/features/settings.json
    :language: json

.. _Fonduer: https://github.com/HazyResearch/fonduer
