# Fonduer

[![Travis](https://img.shields.io/travis/HazyResearch/fonduer.svg)](https://travis-ci.org/HazyResearch/fonduer)
[![Coveralls github](https://img.shields.io/coveralls/github/HazyResearch/fonduer.svg)](https://coveralls.io/github/HazyResearch/fonduer)
[![Read the Docs](https://img.shields.io/readthedocs/fonduer.svg)](https://fonduer.readthedocs.io/)
[![GitHub issues](https://img.shields.io/github/issues/HazyResearch/fonduer.svg)](https://github.com/HazyResearch/fonduer/issues)
[![PyPI](https://img.shields.io/pypi/v/fonduer.svg)](https://pypi.org/project/fonduer/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fonduer.svg)](https://pypi.org/project/fonduer/)
[![GitHub stars](https://img.shields.io/github/stars/HazyResearch/fonduer.svg)](https://github.com/HazyResearch/fonduer/stargazers)
[![GitHub license](https://img.shields.io/github/license/HazyResearch/fonduer.svg)](https://github.com/HazyResearch/fonduer/blob/master/LICENSE)

`Fonduer` is a framework for building knowledge base construction (KBC)
applications from _richy formatted data_ and is implemented as a library on
top of a modified version of [Snorkel](https://hazyresearch.github.io/snorkel/).

_Note that Fonduer is still actively under development, so feedback and
contributions are welcome. Let us know in the
[Issues](https://github.com/HazyResearch/fonduer/issues) section or feel free
to submit your contributions as a pull request._

## Reference

[Fonduer: Knowledge Base Construction from Richly Formatted Data](https://arxiv.org/abs/1703.05028)

```
@article{wu2017fonduer,
  title={Fonduer: Knowledge Base Construction from Richly Formatted Data},
  author={Wu, Sen and Hsiao, Luke and Cheng, Xiao and Hancock, Braden and Rekatsinas, Theodoros and Levis, Philip and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:1703.05028},
  year={2017}
}
```

## Installation

### Dependencies

We use a few applications that you'll need to install and be sure are on your
PATH.

For OS X using [homebrew](https://brew.sh):

```bash
brew install poppler
brew install postgresql
```

On Debian-based distros:

```bash
sudo apt-get install poppler-utils
sudo apt-get install postgresql
```

For the Python dependencies, we recommend using a
[virtualenv](https://virtualenv.pypa.io/en/stable/). Once you have cloned the
repository, change directories to the root of the repository and run

```bash
virtualenv -p python3 .venv
```

Once the virtual environment is created, activate it by running

```bash
source .venv/bin/activate
```

Any Python libraries installed will now be contained within this virtual
environment. To deactivate the environment, simply run `deactivate`.

Then, install `Fonduer` by running

```bash
pip install fonduer
```

## Learning how to use `Fonduer`

The [`Fonduer`
tutorials](https://github.com/hazyresearch/fonduer-tutorials) cover the
`Fonduer` workflow, showing how to extract relations from hardware datasheets
and scientific literature.
