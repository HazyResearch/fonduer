# Fonduer

**_v0.0.1_**

[![Build Status](https://travis-ci.com/SenWu/fonduer.svg?token=T3shSHjcJk8kMbzHEY7Z&branch=master)](https://travis-ci.com/SenWu/fonduer)

`Fonduer` is a framework for building KBC applications from _richy formatted
data_ and is implemented as a library on top of a modified version of
[Snorkel](https://hazyresearch.github.io/snorkel/).

## Installation / dependencies

We recommend using a [virtualenv](https://virtualenv.pypa.io/en/stable/). Once
you have cloned the repository, change directories to the root of the repository
and run

```
virtualenv -p python3 .venv
```

Once the virtual environment is created, activate it by running

```
source .venv/bin/activate
```

Any Python libraries installed will now be contained within this virtual environment.
To deactivate the environment, simply run `deactivate`.

`Fonduer` adds some additional python packages to the default Snorkel
installation which can be installed using `pip`:

```bash
pip install -r python-package-requirement.txt
```

By default (e.g. in the tutorials, etc.) we also use use
[`poppler`](https://poppler.freedesktop.org/) utilities for working with PDFs
along with [PhantomJS](http://phantomjs.org/). You will also be prompted to
install both of these when you run `run.sh`.

## Running

After installing Fonduer, and the additional python dependencies, just run:

```
./run.sh
```

which will finish installing the external libraries we use.

## Learning how to use `Fonduer`

The [`Fonduer`
tutorials](https://github.com/SenWu/fonduer/tree/master/tutorials) cover the
`Fonduer` workflow, showing how to extract relations from hardware datasheets
and scientific literature.

The tutorials are available in the following directory:

```
tutorials/
```

## For Developers

### Testing

You can run unit tests locally by running

```
source ./set_env.sh
pytest tests -rs
```

## FAQs

<details><summary>How do I connect to PostgreSQL? I'm getting "fe_sendauth no
password supplied".</summary><br>

There are [four main
ways](https://dba.stackexchange.com/questions/14740/how-to-use-psql-with-no-password-prompt)
to deal with entering passwords when you connect to your PostgreSQL database:

1. Set the `PGPASSWORD` environment variable
   ```
   PGPASSWORD=<pass> psql -h <host> -U <user>
   ```
2. Using a [.pgpass file to store the
   password](http://www.postgresql.org/docs/current/static/libpq-pgpass.html).
3. Setting the users to [trust
   authentication](https://www.postgresql.org/docs/current/static/auth-methods.html#AUTH-TRUST)
   in the pg_hba.conf file. This makes local development easy, but probably
   isn't suitable for multiuser environments. You can find your hba file
   location by running `psql`, then querying
   ```
   SHOW hba_file;
   ```
4. Put the username and password in the connection URI:
   ```
   postgres://user:pw@localhost:5432/...
   ```

</details>

<details><summary>I'm seeing errors during the poppler
installation.</summary><br>

You may run into errors that look like this:

```
checking for FONTCONFIG... no
configure: error: in `/home/lwhsiao/repos/fonduer/poppler':
configure: error: The pkg-config script could not be found or is too old.  Make
sure it is in your PATH or set the PKG_CONFIG environment variable to the full
path to pkg-config.
```

or this:

```
checking for FONTCONFIG... no
configure: error: Package requirements (fontconfig >= 2.0.0) were not met:

No package 'fontconfig' found

Consider adjusting the PKG_CONFIG_PATH environment variable if you
installed software in a non-standard prefix.
```

Fear not. You just need to make sure these packages are installed:

```
sudo apt-get install pkg-config libfontconfig1-dev
```

</details>
