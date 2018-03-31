Installation
============

We are following `Semantic Versioning 2.0.0 <https://semver.org/>`__
conventions. The maintainers will create a git tag for each release and
increment the version number found in `fonduer/\_version.py`_ accordingly. We
deploy tags to PyPI automatically using Travis-CI.

To test changes in the package, you install it in `editable mode`_ locally in
your virtualenv by running::

    $ make dev

.. tip::
    If you need to install a locally edited version of fonduer in a separate location,
    such as an application, you can directly install your locally modified version::
        
        $ pip install -e path/to/fonduer/

    in the virtualenv of your application.

.. _fonduer/\_version.py: https://github.com/HazyResearch/fonduer/blob/master/fonduer/_version.py
.. _editable mode: https://packaging.python.org/tutorials/distributing-packages/#working-in-development-mode 
