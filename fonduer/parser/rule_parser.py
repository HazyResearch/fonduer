from __future__ import absolute_import, division, unicode_literals

import re
from pathlib import Path

import pkg_resources

try:
    import spacy
    from spacy.cli import download
    from spacy import util
except Exception as e:
    raise Exception("spacy not installed. Use `pip install spacy`.")


class Tokenizer(object):
    '''
    Interface for rule-based tokenizers
    '''

    def apply(self, s):
        raise NotImplementedError()


class RegexTokenizer(Tokenizer):
    '''
    Regular expression tokenization.
    '''

    def __init__(self, rgx="\s+"):
        super(RegexTokenizer, self).__init__()
        self.rgx = re.compile(rgx)

    def apply(self, s):
        '''

        :param s:
        :return:
        '''
        tokens = []
        offset = 0
        # keep track of char offsets
        for t in self.rgx.split(s):
            while t < len(s) and t != s[offset:len(t)]:
                offset += 1
            tokens += [(t, offset)]
            offset += len(t)
        return tokens


class SpacyTokenizer(Tokenizer):
    '''
    Only use spaCy's tokenizer functionality
    '''

    def __init__(self, lang='en'):
        super(SpacyTokenizer, self).__init__()
        self.lang = lang
        self.model = SpacyTokenizer.load_lang_model(lang)

    def apply(self, s):
        doc = self.model.tokenizer(s)
        return [(t.text, t.idx) for t in doc]

    @staticmethod
    def is_package(name):
        """Check if string maps to a package installed via pip.
        name (unicode): Name of package.
        RETURNS (bool): True if installed package, False if not.

        From https://github.com/explosion/spaCy/blob/master/spacy/util.py

        """
        name = name.lower()  # compare package name against lowercase name
        packages = pkg_resources.working_set.by_key.keys()
        for package in packages:
            if package.lower().replace('-', '_') == name:
                return True
        return False

    @staticmethod
    def model_installed(name):
        '''
        Check if spaCy language model is installed

        From https://github.com/explosion/spaCy/blob/master/spacy/util.py

        :param name:
        :return:
        '''
        data_path = util.get_data_path()
        if not data_path or not data_path.exists():
            raise IOError("Can't find spaCy data path: %s" % str(data_path))
        if name in set([d.name for d in data_path.iterdir()]):
            return True
        if spacy.is_package(name):  # installed as package
            return True
        if Path(name).exists():  # path to model data directory
            return True
        return False

    @staticmethod
    def load_lang_model(lang):
        '''
        Load spaCy language model or download if
        model is available and not installed

        Currenty supported spaCy languages

        en English (50MB)
        de German (645MB)
        fr French (1.33GB)
        es Spanish (377MB)

        :param lang:
        :return:
        '''
        if SpacyTokenizer.model_installed:
            model = spacy.load(lang)
        else:
            download(lang)
            model = spacy.load(lang)
        return model
