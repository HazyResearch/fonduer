"""For pip."""
from setuptools import setup
from setuptools import find_packages

try:
    from pypandoc import convert
    long_description = convert('README.md', 'rst')
except (IOError, ImportError):
    print("warning: pypandoc module not found. Could not convert MD to RST.")
    long_description = open('README.md').read()

exec(open('fonduer/_version.py').read())
setup(
    name='fonduer',
    version=__version__,
    description="Knowledge base construction system for richly formatted data.",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        'bs4',
        'editdistance',
        'future',
        'ipywidgets>=7.0',
        'jupyter',
        'lxml==3.6.4',
        'matplotlib',
        'nltk',
        'numba',
        'numbskull',
        'numpy>=1.11',
        'pandas',
        'pdftotree',
        'psycopg2-binary',
        'pyyaml',
        'requests',
        'scipy>=0.18',
        'six',
        'spacy>=2.0.7',
        'sqlalchemy>=1.0.14',
        'tensorflow>=1.0',
        'treedlib',
        'wand',
    ],
    keywords=['pdf', 'parsing', 'html'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=3.4.0'],
    include_package_data=True,
    url="https://github.com/HazyResearch/fonduer",
    classifiers=[  # https://pypi.python.org/pypi?:action=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3 :: Only",
    ],
    project_urls={
        'Tracker': 'https://github.com/HazyResearch/fonduer/issues',
        'Source': 'https://github.com/HazyResearch/fonduer',
    },
    python_requires='>3',
    author="Hazy Research",
    author_email="senwu@cs.stanford.edu",
    license='MIT',
)
