Frequently Asked Questions (FAQs)
=================================

Here are a collection of troubleshooting questions we've seen asked. If you
run into anything not covered in this section, feel free to open an Issue_.

.. _Issue: https://github.com/hazyresearch/fonduer/issues

When I try to createdb, or use psql, I get FATAL: role "<username>" does not exist.
-----------------------------------------------------------------------------------
If you just installed PostgreSQL, you probably need to add users. You will need
sudo privileges to do this.

We recommend using createuser_ to define a new PostgreSQL user account::

  $ sudo -u postgres createuser [options] [username]

.. _createuser: https://www.postgresql.org/docs/current/static/app-createuser.html

How do I connect to PostgreSQL? I'm getting "fe\_sendauth no password supplied".
--------------------------------------------------------------------------------
There are `four main ways`_ to deal with entering passwords when you connect to
your PostgreSQL database:

1. Set the ``PGPASSWORD`` environment variable ``PGPASSWORD=<pass> psql -h
   <host> -U <user>``
2. Using a `.pgpass file to store the password`_.
3. Setting the users to `trust authentication`_ in the pg\_hba.conf file. This
   makes local development easy, but probably isn't suitable for multiuser
   environments. You can find your hba file location by running::

    $ sudo -u postgres psql -c "SHOW hba_file;"

4. Put the username and password in the connection URI:
   ``postgresql://<user>:<pw>@<host>:<port>/<database_name>``

.. _.pgpass file to store the password: http://www.postgresql.org/docs/current/static/libpq-pgpass.html
.. _four main ways: https://dba.stackexchange.com/questions/14740/how-to-use-psql-with-no-password-prompt
.. _trust authentication: https://www.postgresql.org/docs/current/static/auth-methods.html#AUTH-TRUST

I'm getting a CalledProcessError for command 'pdftotext -f 1 -l 1 -bbox-layout'?
--------------------------------------------------------------------------------

Are you using Ubuntu 14.04 (or older)? Fonduer requires ``poppler-utils`` to be
version ``0.36.0`` or greater. Otherwise, the ``-bbox-layout`` option is not
available for ``pdftotext`` (`see changelog`_).

If you must use Ubuntu 14.04, you can `install manually`_. As an example, to
install ``0.53.0``::

    $ sudo apt install build-essential checkinstall
    $ wget poppler.freedesktop.org/poppler-0.53.0.tar.xz
    $ tar -xf ./poppler-0.53.0.tar.xz
    $ cd poppler-0.53.0
    $ ./configure
    $ make
    $ sudo checkinstall

We highly recommend using at least Ubuntu 16.04 though, as we haven't done
testing on 14.04 or older.

.. _see changelog: https://poppler.freedesktop.org/releases.html
.. _install manually: https://poppler.freedesktop.org

How can I use use Fonduer for documents in Languages other than English?
------------------------------------------------------------------------

If available, Fonduer uses languages supported by spaCy for tokenization and
its NLP pipeline (`see spacy language support`_). We also started adding
languages with spaCy alpha support for tokenization (`see spacy alpha
languages`_). Currently, only Chinese and Japanese are supported.

If you would like to use Fonduer for Japanese documents, you can use
``pip install fonduer[spacy_ja]`` to install Fonduer with Japanese language support.

If you would like to use Fonduer for Chinese documents, you can use
``pip install fonduer[spacy_zh]`` to install Fonduer with Chinese language support.

If you would like to use other languages with spaCy alpha support, which are
not yet integrated in Fonduer, feel free to submit a `Pull Request`_ or open an Issue_.

.. _see spacy language support: https://spacy.io/usage/models#languages
.. _see spacy alpha languages: https://spacy.io/usage/models#alpha-support
.. _see mecab on PyPI: https://pypi.org/project/mecab-python3/
.. _see jieba on PyPI: https://pypi.org/project/jieba/
.. _Pull Request: https://github.com/hazyresearch/fonduer/pulls
.. _Issue: https://github.com/hazyresearch/fonduer/issues
