import importlib
import logging
from collections import defaultdict
from pathlib import Path
from string import whitespace

import pkg_resources

try:
    import spacy
    from spacy.cli import download
    from spacy import util
    from spacy.tokens import Doc
except Exception:
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

    def __init__(self, lang):
        self.logger = logging.getLogger(__name__)
        self.name = "spacy"
        self.languages = ["en", "de", "es", "pt", "fr", "it", "nl", "xx"]
        self.alpha_languages = {"ja": "Japanese"}

        self.lang = lang
        self.model = None

        # self.model = self.load_lang_model()

    def has_tokenizer_support(self):
        return self.lang and (
            self.has_NLP_support() or self.lang in self.alpha_languages
        )

    def has_NLP_support(self):
        return self.lang and (self.lang in self.languages)

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
        """Check if spaCy language model is installed.

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

    def load_lang_model(self):
        """
        Load spaCy language model or download if model is available and not
        installed.

        Currenty supported spaCy languages

        en English (50MB)
        de German (645MB)
        fr French (1.33GB)
        es Spanish (377MB)

        :return:
        """
        if self.lang in self.languages:
            if not Spacy.model_installed(self.lang):
                download(self.lang)
            model = spacy.load(self.lang)
        elif self.lang in self.alpha_languages:
            language_module = importlib.import_module("spacy.lang.{}".format(self.lang))
            language_method = getattr(language_module, self.alpha_languages[self.lang])
            model = language_method()
            """ TODO: Depending on OS (Linux/macOS) and on the sentence to be parsed,
            UnicodeDecodeError or ValueError happens at the first use when lang='ja'.
            As a workaround, the model parses some sentence before actually being used.
            """
            if self.lang == "ja":
                try:
                    model("初期化")
                except (UnicodeDecodeError, ValueError):
                    pass
        self.model = model

    def sentence_list_separator_function(self, all_sentence_objs):
        start_token_marker = []
        total_nr_input_words = 0
        for sentence in all_sentence_objs:
            if len(sentence.words) > 0:
                start_token_marker += [True] + [False] * (len(sentence.words) - 1)
                total_nr_input_words += len(sentence.words)

        def set_custom_boundary(doc):
            output_tokens = list(doc)
            total_nr_output_words = len(output_tokens)
            try:
                assert total_nr_input_words == total_nr_output_words
            except AssertionError:
                self.logger.error(
                    "input token number ({}) not same as output token"
                    " nr ({})".format(total_nr_input_words, total_nr_output_words)
                )
                raise

            for token_nr, token in enumerate(doc):
                if start_token_marker[token_nr] is True:
                    doc[token.i].is_sent_start = True
                else:
                    doc[token.i].is_sent_start = False
            return doc

        return set_custom_boundary

    def enrich_sentences_with_NLP(self, all_sentences):
        """
        Enrich a list of fonduer Sentence objects with NLP features. We merge
        and process the text of all Sentences for higher efficiency.

        :param all_sentences: List of fonduer Sentence objects for one document
        :return:
        """
        if self.lang in self.alpha_languages:
            raise NotImplementedError(
                "Language {} not available in "
                "spacy beyond tokenization".format(self.lang)
            )

        if len(all_sentences) == 0:
            return  # Nothing to parse

        if self.model.has_pipe("sbd"):
            self.model.remove_pipe("sbd")
            self.logger.debug(
                "Removed sentencizer ('sbd') from model. Now in pipeline: {}".format(
                    self.model.pipe_names
                )
            )

        batch_char_limit = self.model.max_length
        sentence_batches = [[]]
        num_chars = 0
        for sentence in all_sentences:
            if num_chars + len(sentence.text) >= batch_char_limit:
                sentence_batches.append([sentence])
                num_chars = len(sentence.text)
            else:
                sentence_batches[-1].append(sentence)
                num_chars += len(sentence.text)

        # TODO: We could do this in parallel. Test speedup in the future
        for sentence_batch in sentence_batches:
            batch_sentence_strings = [x.text for x in sentence_batch]

            if self.model.has_pipe("sentence_boundary_detector"):
                self.model.remove_pipe(name="sentence_boundary_detector")

            sentence_separator_fct = self.sentence_list_separator_function(
                sentence_batch
            )
            self.model.add_pipe(
                sentence_separator_fct,
                before="parser",
                name="sentence_boundary_detector",
            )

            custom_tokenizer = TokenPreservingTokenizer(
                self.model.vocab, sentence_batch
            )
            # we circumvent redundant tokenization by using a custom
            # tokenizer that directly uses the already separated words
            # of each sentence as tokens
            doc = custom_tokenizer()
            for name, proc in self.model.pipeline:  # iterate over components in order
                doc = proc(doc)

            try:
                assert doc.is_parsed
            except Exception:
                self.logger.exception("{} was not parsed".format(doc))

            batch_parsed_sentences = list(doc.sents)
            try:
                assert len(batch_sentence_strings) == len(batch_parsed_sentences)
            except AssertionError:
                self.logger.error(
                    "Number of parsed spacy sentences doesnt match input sentences:"
                    " input {}, output: {}, document: {}".format(
                        len(batch_sentence_strings),
                        len(batch_parsed_sentences),
                        sentence_batch[0].document,
                    )
                )
                raise

            sentence_nr = -1
            for sent in batch_parsed_sentences:
                sentence_nr += 1
                parts = defaultdict(list)

                for i, token in enumerate(sent):
                    parts["lemmas"].append(token.lemma_)
                    parts["pos_tags"].append(token.tag_)
                    parts["ner_tags"].append(
                        token.ent_type_ if token.ent_type_ else "O"
                    )
                    head_idx = (
                        0 if token.head is token else token.head.i - sent[0].i + 1
                    )
                    parts["dep_parents"].append(head_idx)
                    parts["dep_labels"].append(token.dep_)
                current_sentence_obj = sentence_batch[sentence_nr]
                current_sentence_obj.pos_tags = parts["pos_tags"]
                current_sentence_obj.lemmas = parts["lemmas"]
                current_sentence_obj.ner_tags = parts["ner_tags"]
                current_sentence_obj.dep_parents = parts["dep_parents"]
                current_sentence_obj.dep_labels = parts["dep_labels"]
                yield current_sentence_obj

    def split_sentences(self, document, text):
        """
        Split input text into sentences that match CoreNLP's default format,
        but are not yet processed.

        :param document: The Document context
        :param text: The text of the parent paragraph of the sentences
        :return:
        """

        if self.model.has_pipe("sentence_boundary_detector"):
            self.model.remove_pipe(name="sentence_boundary_detector")

        if not self.model.has_pipe("sbd"):
            sbd = self.model.create_pipe("sbd")  # add sentencizer
            self.model.add_pipe(sbd)
        try:
            doc = self.model(text, disable=["parser", "tagger", "ner"])
        except ValueError:
            # temporary increase character limit of spacy
            # 'Probably save' according to spacy, as no parser or NER is used
            previous_max_length = self.model.max_length
            self.model.max_length = 100000000
            self.logger.warning(
                "Temporarily increased spacy maximum "
                " character limit to split sentences.".format(self.model.max_length)
            )
            doc = self.model(text, disable=["parser", "tagger", "ner"])
            self.model.max_length = previous_max_length
            self.logger.warning(
                "Spacy maximum"
                " character limit set back to.".format(self.model.max_length)
            )

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

            # make char_offsets relative to start of sentence
            parts["char_offsets"] = [
                p - parts["char_offsets"][0] for p in parts["char_offsets"]
            ]
            parts["position"] = position

            # Link the sentence to its parent document object
            parts["document"] = document
            parts["text"] = text

            position += 1

            yield parts


class TokenPreservingTokenizer(object):
    """
    This custom tokenizer simply preserves the tokenization that was already
    performed during sentence splitting. It will output a list of space
    separated tokens, whereas each token is a single word from the list of
    sentences.

    :param vocab: The vocab attribute of the respective spacy language object
    :param tokenized_sentences: A list of sentences that was previously
        tokenized/split by spacy
    :return:
    """

    def __init__(self, vocab, tokenized_sentences):
        self.logger = logging.getLogger(__name__)
        self.all_input_tokens = []
        self.vocab = vocab
        self.all_spaces = []
        for sentence in tokenized_sentences:
            words_in_sentence = sentence.words
            if len(words_in_sentence) > 0:
                self.all_input_tokens += sentence.words
                current_sentence_pos = 0
                spaces_list = [True] * len(words_in_sentence)
                # Last word in sentence always assumed to be followed by space
                for i, word in enumerate(words_in_sentence[:-1]):
                    current_sentence_pos = sentence.text.find(
                        word, current_sentence_pos
                    )
                    if current_sentence_pos == -1:
                        raise AttributeError(
                            "Could not find token in its parent sentence"
                        )
                    current_sentence_pos += len(word)
                    if not any(
                        sentence.text[current_sentence_pos:].startswith(s)
                        for s in whitespace
                    ):
                        spaces_list[i] = False
                self.all_spaces += spaces_list

    def __call__(self):
        return Doc(self.vocab, words=self.all_input_tokens, spaces=self.all_spaces)
