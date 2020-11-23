Testing
=======

We use pytest_ to run our tests. Our tests are all located in the ``tests``
directory in the repo, and are meant to be run *after* installing_ Fonduer
locally.

You need PostgreSQL running locally with `trust authentication`_ enabled.
You'll also need to download the external test data::

    $ cd tests/
    $ ./download_data.sh
    $ cd ..

Then, you'll be able to run our tests::

    $ make test

.. _pytest: https://docs.pytest.org/en/latest/
.. _installing: install.html
.. _trust authentication: https://www.postgresql.org/docs/current/static/auth-methods.html#AUTH-TRUST
