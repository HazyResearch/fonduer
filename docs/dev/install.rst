Installation
============

We are following `Semantic Versioning 2.0.0 <https://semver.org/>`__
conventions. The maintainers will create a git tag for each release and
increment the version number found in `fonduer/\_version.py`_ accordingly. We
deploy tags to PyPI automatically using Travis-CI.

To install locally, you'll need to install ``pandoc``:

.. code:: bash

    sudo apt-get install pandoc

which is used to create the reStructuredText file from the Markdown-formatted
README.md that ``setuptools`` expects.

To test changes in the package, you install it in `editable mode`_ locally in
your virtualenv by running:

.. code:: bash

    make dev

.. _fonduer/\_version.py: https://github.com/HazyResearch/fonduer/blob/master/fonduer/_version.py
.. _editable mode: https://packaging.python.org/tutorials/distributing-packages/#working-in-development-mode 
