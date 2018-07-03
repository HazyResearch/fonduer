Code Style
==========

For code consistency, we use `black`_ as our Python code formatter with its
default settings. Black helps minimize the line diffs and allows you to not
worry about formatting during your own development. Just run black on each of
your files before committing them. You can install black by running::

    $ pip install black

.. tip::
    Whatever editor you use, we recommend checking out `black editor
    integrations`_ to help make the code formatting process just a few
    keystrokes.

We are using flake8_ to enforce general code style standards. This style check
is part of our Travis-CI tests which is required before merging. You can check
this on your local machine by installing flake8::

    $ pip install flake8
    $ make check 


.. _flake8: https://flake8.pycqa.org/en/latest/ 
.. _black editor integrations: https://github.com/ambv/black#editor-integration
.. _black: https://github.com/ambv/black 
