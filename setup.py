"""For pip."""
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

exec(open("src/fonduer/_version.py").read())
setup(
    name="fonduer",
    version=__version__,
    description="Knowledge base construction system for richly formatted data.",
    long_description=open("README.rst").read(),
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    install_requires=[
        "bs4>=0.0.1, <0.1.0",
        "editdistance>=0.5.2, <0.6.0",
        "lxml>=4.2.5, <5.0.0",
        "nltk>=3.3, <4.0",
        "numpy>=1.11, <2.0",
        "pandas>=0.23.4, <0.24.0",
        "pyyaml>=3.13, <4.0",
        "scipy>=1.1.0, <2.0.0",
        "snorkel-metal>=0.3.3, <0.4.0",
        "spacy>=2.0.12, <3.0.0",
        "sqlalchemy[postgresql_psycopg2binary]>=1.2.12, <2.0.0",
        "torch>=0.4.1, <0.5.0",
        "tqdm>=4.26.0, <5.0.0",
        "treedlib>=0.1.1, <0.2.0",
        "wand>=0.4.4, <0.5.0",
    ],
    extras_require={
        "spacy_ja": ["mecab-python3>=0.7, <0.8"],
        "spacy_zh": ["jieba>=0.39, <0.40"],
    },
    keywords=["fonduer", "knowledge base construction", "richly formatted data"],
    include_package_data=True,
    url="https://github.com/HazyResearch/fonduer",
    classifiers=[  # https://pypi.python.org/pypi?:action=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
    ],
    project_urls={
        "Tracker": "https://github.com/HazyResearch/fonduer/issues",
        "Source": "https://github.com/HazyResearch/fonduer",
    },
    python_requires=">=3.6",
    author="Hazy Research",
    author_email="senwu@cs.stanford.edu",
    license="MIT",
)
