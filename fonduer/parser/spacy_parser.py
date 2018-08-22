import logging
import random
import string
from collections import defaultdict
from pathlib import Path

import pkg_resources

from fonduer.parser.models import construct_stable_id

try:
    import spacy
    from spacy.cli import download
    from spacy import util
except Exception as e:
    raise Exception("spaCy not installed. Use `pip install spacy`.")


class Spacy(object):
    """
    spaCy
    https://spacy.io/

    Models for each target language needs to be downloaded using the
    following command:

    python -m spacy download en

    Default named entity types

    PERSON      People, including fictional.
    NORP        Nationalities or religious or political groups.
    FACILITY    Buildings, airports, highways, bridges, etc.
    ORG         Companies, agencies, institutions, etc.
    GPE         Countries, cities, states.
    LOC         Non-GPE locations, mountain ranges, bodies of water.
    PRODUCT     Objects, vehicles, foods, etc. (Not services.)
    EVENT       Named hurricanes, battles, wars, sports events, etc.
    WORK_OF_ART Titles of books, songs, etc.
    LANGUAGE    Any named language.

    DATE        Absolute or relative dates or periods.
    TIME        Times smaller than a day.
    PERCENT     Percentage, including "%".
    MONEY       Monetary values, including unit.
    QUANTITY    Measurements, as of weight or distance.
    ORDINAL     "first", "second", etc.
    CARDINAL    Numerals that do not fall under another type.

    """

    def __init__(
        self,
        # annotators=["tagger", "parser", "entity"],
        lang="en",
        num_threads=1,
        verbose=False,
    ):
        self.logger = logging.getLogger(__name__)
        self.name = "spacy"
        self.model = Spacy.load_lang_model(lang)
        self.num_threads = num_threads

        # self.pipeline = [proc for _, proc in self.model.__dict__["pipeline"]]

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
            if package.lower().replace("-", "_") == name:
                return True
        return False

    @staticmethod
    def model_installed(name):
        """
        Check if spaCy language model is installed

        From https://github.com/explosion/spaCy/blob/master/spacy/util.py

        :param name:
        :return:
        """
        data_path = util.get_data_path()
        if not data_path or not data_path.exists():
            raise IOError("Can't find spaCy data path: %s" % str(data_path))
        if name in {d.name for d in data_path.iterdir()}:
            return True
        if Spacy.is_package(name):  # installed as package
            return True
        if Path(name).exists():  # path to model data directory
            return True
        return False

    @staticmethod
    def load_lang_model(lang):
        """
        Load spaCy language model or download if
        model is available and not installed

        Currenty supported spaCy languages

        en English (50MB)
        de German (645MB)
        fr French (1.33GB)
        es Spanish (377MB)

        :param lang:
        :return:
        """
        if not Spacy.model_installed(lang):
            download(lang)
        return spacy.load(lang)

    def custom_boundary_funct(self, separator_str):
        def set_custom_boundary(doc):
            for token in doc[:-1]:
                if token.text == separator_str:
                    doc[token.i + 1].is_sent_start = True
                    doc[token.i].is_sent_start = True
                else:
                    doc[token.i + 1].is_sent_start = False
            return doc

        return set_custom_boundary

    def parse(self, all_sentences):
        """
        Transform spaCy output to match CoreNLP's default format
        :param document:
        :param text:
        :return:
        """

        # doc = self.model.tokenizer(merged_sentences)

        if self.model.has_pipe("sbd"):
            self.model.remove_pipe("sbd")
            self.logger.debug(
                "removed 'sbd' from model. Now in pipeline: {}".format(
                    self.model.pipe_names
                )
            )

        # Create random, (most likely) unique string to separate sentences
        separator_str = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=10)
        )
        spaced_separator_str = " " + separator_str + " "

        all_sentence_strings = [x.text for x in all_sentences]
        merged_sentences = spaced_separator_str.join(all_sentence_strings)
        self.model.Defaults.stop_words.add(spaced_separator_str)
        self.model.Defaults.stop_words.add(separator_str)
        custom_boundary_fct = self.custom_boundary_funct(separator_str)

        self.model.add_pipe(custom_boundary_fct, before="parser")
        doc = self.model(merged_sentences)

        try:
            assert doc.is_parsed
        except Exception:
            self.logger.exception("{} was not parsed".format(doc))

        self.logger.debug(
            "number of input sentences: {} vs number from pipeline:\
             {}\n with subtracted: {}".format(
                len(all_sentences),
                len(list(doc.sents)),
                len(list(doc.sents)) - (len(all_sentences) - 1),
            )
        )

        # skipped_sentences = 0
        sentence_nr = -1
        for sent in doc.sents:
            if separator_str in sent.text:
                continue
            sentence_nr += 1
            parts = defaultdict(list)

            for i, token in enumerate(sent):
                if str(token) == separator_str:
                    self.logger.warning("Found separator string in a sentence token.")
                    continue

                parts["lemmas"].append(token.lemma_)
                parts["pos_tags"].append(token.tag_)
                parts["ner_tags"].append(token.ent_type_ if token.ent_type_ else "O")
                head_idx = 0 if token.head is token else token.head.i - sent[0].i + 1
                parts["dep_parents"].append(head_idx)
                parts["dep_labels"].append(token.dep_)
            current_sentence_obj = all_sentences[sentence_nr]
            current_sentence_obj.pos_tags = parts["pos_tags"]
            current_sentence_obj.lemmas = parts["lemmas"]
            current_sentence_obj.ner_tags = parts["ner_tags"]
            current_sentence_obj.dep_parents = parts["dep_parents"]
            current_sentence_obj.dep_labels = parts["dep_labels"]

    def split_sentences(self, document, text):
        """
        Split input text into sentences that match CoreNLP's
         default format, but are not yet processed
        :param document:
        :param text:
        :return:
        """
        if not self.model.has_pipe("sbd"):
            sbd = self.model.create_pipe("sbd")  # add sentencizer
            self.model.add_pipe(sbd)
        doc = self.model(text, disable=["parser", "tagger", "ner"])
        position = 0
        for sent in doc.sents:
            parts = defaultdict(list)
            text = sent.text

            for i, token in enumerate(sent):
                parts["words"].append(str(token))
                parts["lemmas"].append("")  # placeholder for later NLP parsing
                parts["pos_tags"].append("")  # placeholder for later NLP parsing
                parts["ner_tags"].append("")  # placeholder for later NLP parsing
                parts["char_offsets"].append(token.idx)
                parts["abs_char_offsets"].append(token.idx)
                parts["dep_parents"].append(0)  # placeholder for later NLP parsing
                parts["dep_labels"].append("")  # placeholder for later NLP parsing

            # Add null entity array (matching null for CoreNLP)
            parts["entity_cids"] = ["O" for _ in parts["words"]]
            parts["entity_types"] = ["O" for _ in parts["words"]]

            # make char_offsets relative to start of sentence
            parts["char_offsets"] = [
                p - parts["char_offsets"][0] for p in parts["char_offsets"]
            ]
            parts["position"] = position

            # Link the sentence to its parent document object
            parts["document"] = document
            parts["text"] = text

            # Add null entity array (matching null for CoreNLP)
            parts["entity_cids"] = ["O" for _ in parts["words"]]
            parts["entity_types"] = ["O" for _ in parts["words"]]

            # Assign the stable id as document's stable id plus absolute
            # character offset
            abs_sent_offset = parts["abs_char_offsets"][0]
            abs_sent_offset_end = (
                abs_sent_offset + parts["char_offsets"][-1] + len(parts["words"][-1])
            )
            if document:
                parts["stable_id"] = construct_stable_id(
                    document, "sentence", abs_sent_offset, abs_sent_offset_end
                )

            position += 1

            yield parts
