Testing
=======

We use pytest_ to run our tests. Our tests are all located in the ``tests``
directory in the repo, and are meant to be run *after* installing_ Fonduer
locally.

In order to run the tests, you will need to create the local databases used
by the tests::

    $ createdb parser_test
    $ createdb e2e_test
    $ createdb cand_test
    $ createdb meta_test
    $ createdb inc_test

You'll also need to download the external test data::

    $ cd tests/
    $ ./download_data.sh
    $ cd ..

Then, you'll be able to run our tests::

    $ make test

.. _pytest: https://docs.pytest.org/en/latest/
.. _installing: install.html
