Code Style
==========

For code consistency, we have a `pre-commit`_ configuration file so that you
can easily install pre-commit hooks to run style checks before you commit your
files. You can setup our pre-commit hooks by running::

    $ pip install -r requirements-dev.txt
    $ pre-commit install

Or, just run::

    $ make dev

Now, each time you commit, checks will be run using the packages explained
below.

We use `black`_ as our Python code formatter with its default settings. Black
helps minimize the line diffs and allows you to not worry about formatting
during your own development. Just run black on each of your files before
committing them.

.. tip::
    Whatever editor you use, we recommend checking out `black editor
    integrations`_ to help make the code formatting process just a few
    keystrokes.

For sorting imports, we reply on `isort`_. Our repository already includes a
`.isort.cfg` that is compatible with black. You can run a code style check on
your local machine by running our checks::

    $ make check

.. _pre-commit: https://pre-commit.com/
.. _isort: https://github.com/timothycrosley/isort
.. _black editor integrations: https://github.com/ambv/black#editor-integration
.. _black: https://github.com/ambv/black


Docstring format
^^^^^^^^^^^^^^^^

We use Sphinx_ to build documentation.
While Sphinx's ``autodoc`` extension, which automatically generates documentation from
docstring, supports several docstring formats, we use `Sphinx docstring format`_.
A typical Sphinx docstring looks like the following:

.. code-block:: python
    :caption: A typical Sphinx docstring

    """[Summary]

    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """

where `:type` and `:rtype` can be omitted.
Since we annotate functions with types (`PEP 484`_), `:type` and `:rtype` in the docstring
are redundant and hence not allowed in Fonduer. An example Fonduer docstring looks like:

.. code-block:: python
    :caption: An example Fonduer docstring

    def greeting(name: str) -> str:
        """Return a greeting to a person.

        :param name: a person's name to greet.
        :return: a greeting string.
        """
        return 'Hello ' + name

Note that `:type` and `:rtype` are omitted and that the function is annotated with types.

.. _Sphinx: https://www.sphinx-doc.org
.. _Sphinx docstring format: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format
.. _PEP 484: https://www.python.org/dev/peps/pep-0484/
