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
        "beautifulsoup4>=4.8.0",
        "editdistance>=0.5.2, <0.6.0",
        "snorkel>=0.9.5, <0.10.0",
        "emmental>=0.0.6, <0.1.0",
        "lxml>=4.2.5, <5.0.0",
        "mlflow>=1.1.0, <2.0.0",
        "numpy>=1.11, <2.0",
        "pyyaml>=5.1, <6.0",
        "scipy>=1.1.0, <2.0.0",
        "spacy>=2.1.3, <2.4.0",
        "sqlalchemy[postgresql]>=1.3.7, <2.0.0",
        "torch>=1.3.1,<2.0.0",
        "tqdm>=4.36.0, <5.0.0",
        "treedlib>=0.1.3, <0.2.0",
        "wand>=0.4.4, <0.6.0",
        "ipython",
        "deprecation",
    ],
    extras_require={
        "spacy_ja": ["fugashi[unidic-lite]>=0.2.3"],
        "spacy_zh": ["jieba>=0.39, <0.40"],
    },
    keywords=["fonduer", "knowledge base construction", "richly formatted data"],
    package_data={"fonduer": ["py.typed"]},
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
