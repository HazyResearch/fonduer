Getting Started
===============

This document will show you how to get up and running with Fonduer. We'll show
you how to get everything installed and your machine so that you can walk
through real examples by checking out our Tutorials_.

Installing Non-Python Dependencies
----------------------------------

Fonduer relies on a couple of non-Python applications. You'll need to install
these and be sure are on your ``PATH``.

For OS X using homebrew_::

    $ brew install poppler
    $ brew install postgresql@10
    $ brew install libpng freetype pkg-config
    $ brew install libomp #https://github.com/pytorch/pytorch/issues/20030
    $ brew install imagemagick

On Debian-based distros::

    $ sudo apt update
    $ sudo apt install libxml2-dev libxslt-dev python3-dev libpq-dev
    $ sudo apt build-dep python-matplotlib
    $ sudo apt install poppler-utils
    $ sudo apt install postgresql
    $ sudo apt install libmagickwand-dev

.. note::
    Fonduer requires PostgreSQL version 9.6 or higher.

.. note::
    Fonduer requires ``poppler-utils`` to be version 0.36.0 or later.
    Otherwise, the ``-bbox-layout`` option is not available for ``pdftotext``
    (`see changelog`_).
    It is recommended to use ``poppler-utils`` version 0.48.0 or later
    to avoid `a known bug <https://bugs.freedesktop.org/show_bug.cgi?id=97399>`_.

.. note::
    Use Wand (>=0.5.0) with ImageMagick7 as Wand (<0.5.0) does not support ImageMagick7.

Installing the Fonduer Package
------------------------------

Then, install Fonduer by running::

    $ pip install fonduer

.. note::
    Fonduer only supports Python 3. Python 2 is not supported.

.. tip::
  For the Python dependencies, we recommend using a virtualenv_, which will
  allow you to install Fonduer and its python dependencies in an isolated
  Python environment. Once you have virtualenv installed, you can create a
  Python 3 virtual environment as follows.::

      $ virtualenv -p python3.6 .venv

  Once the virtual environment is created, activate it by running::

      $ source .venv/bin/activate

  Any Python libraries installed will now be contained within this virtual
  environment. To deactivate the environment, simply run::

      $ deactivate

Downloading spaCy language models
---------------------------------

Language models introduced recently cannot be downloaded by Fonduer.
Those models should be downloaded and their shortcuts should be created as below::

    $ python -m spacy download ja_core_news_sm
    $ python -m spacy link ja_core_news_sm ja
    $ python -m spacy download zh_core_web_sm
    $ python -m spacy link zh_core_web_sm zh

The Fonduer Pipeline
--------------------

The Fonduer pipeline can be broken into five phases.

  #. Parsing
      In this first stage, an input corpus of richly formatted documents is
      parsed into Fonduer's data model.
  #. Mention and Candidate Extraction
      Here, we initialize the knowledge base with the user's target schema.
      Users define Mentions using Matchers_, and then combine Mentions to
      create Candidates. Throttlers can also (optionally) be added to filter
      out invalid Candidates to achieve better class balance.
  #. Multimodal Featurization
      Fonduer then featurizes each candidate with features from multiple
      modalities.
  #. Supervision
      Next, users provide labeling functions (which can leverage our
      `data model utilities`_) to provide weak supervision.
  #. Classification
      Finally, Fonduer provides machine learning models which are used to
      classify each Candidate.

To demonstrate how to set up and use Fonduer in your applications, we walk
through each of these phases in real-world examples in our Tutorials_.

Check out the `Fonduer paper`_ for more details about the system.


.. _Fonduer paper: https://arxiv.org/abs/1703.05028
.. _Tutorials: https://github.com/HazyResearch/fonduer-tutorials
.. _data model utilities: data_model_utils.html
.. _homebrew: https://brew.sh
.. _Matchers: candidates.html#matchers
.. _preprocessors: preprocessors.html
.. _see changelog: https://poppler.freedesktop.org/releases.html
.. _virtualenv: https://virtualenv.pypa.io/en/stable/
