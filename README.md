# Fonduer

 **_v0.0.1_**

 [![Build Status](https://travis-ci.com/SenWu/fonduer.svg?token=3mETsMEqfcpP23yhpsr4&branch=master)](https://travis-ci.com/SenWu/fonduer)

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

By default (e.g. in the tutorials, etc.) we also use [Stanford
CoreNLP](http://stanfordnlp.github.io/CoreNLP/) for pre-processing text; you
will be prompted to install this when you run `run.sh`. In addition, we also
use [`poppler`](https://poppler.freedesktop.org/) utilities for working with
PDFs along with [PhantomJS](http://phantomjs.org/). You will also be prompted
to install both of these when you run `run.sh`.

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
