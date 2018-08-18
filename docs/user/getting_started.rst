Getting Started
===============

This document will show you how to get up and running with Fonduer. We'll show
you how to get everything installed and your machine so that you can walk
through real examples by checking out our Tutorials_.

Installing External Dependencies
--------------------------------

Fonduer relies on a couple of external applications. You'll need to install
these and be sure are on your ``PATH``.

For OS X using homebrew_::

    $ brew install poppler
    $ brew install postgresql

On Debian-based distros::

    $ sudo apt update
    $ sudo apt install libxml2-dev libxslt-dev python3-dev
    $ sudo apt install poppler-utils
    $ sudo apt install postgresql

.. note::
    Fonduer requires PostgreSQL version 9.6 or above.

.. note::
    Fonduer requires ``poppler-utils`` to be version 0.36.0 or above.
    Otherwise, the ``-bbox-layout`` option is not available for ``pdftotext``
    (`see changelog`_).

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


The Fonduer Pipeline
--------------------

The Fonduer pipeline can be broken into three main phases. Each of these phases
requires input from you as you build your Fonduer application.

  #. Knowledge Base Initialization
      We initialize the target knowledge base. Fonduer requires users to
      specify a schema, as well as preprocessors_ to be used while parsing your
      corpus.
  #. Candidate Extraction
      Next, users provide matchers_ to define which each type in the schema
      looks like. Throttlers can also (optionally) be added to filter out
      invalid candidates to achieve better class balance.
  #. Multimodal Featurization, Supervision, and Classification
      Finally, users provide labeling functions (which can leverage our
      `data model utilities`_) to provide weak supervision.

To demonstrate how to set up and use Fonduer in your applications, we walk
through each of these phases in real-world examples in our Tutorials_.

Check out the `Fonduer paper`_ for more details about the system.


.. _Fonduer paper: https://arxiv.org/abs/1703.05028
.. _Tutorials: https://github.com/HazyResearch/fonduer-tutorials
.. _data model utilities: data_model_utils.html
.. _homebrew: https://brew.sh
.. _matchers: matchers.html
.. _preprocessors: preprocessors.html
.. _see changelog: https://poppler.freedesktop.org/releases.html
.. _virtualenv: https://virtualenv.pypa.io/en/stable/
