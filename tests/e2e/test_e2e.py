#!/usr/bin/env python
import csv
import logging
import os
import pickle
import re
from builtins import range

import pytest

from fonduer import (
    BatchFeatureAnnotator,
    BatchLabelAnnotator,
    CandidateExtractor,
    DictionaryMatch,
    Document,
    GenerativeModel,
    HTMLPreprocessor,
    Intersect,
    LambdaFunctionMatcher,
    Meta,
    OmniParser,
    Phrase,
    RegexMatchSpan,
    SparseLogisticRegression,
    Union,
    candidate_subclass,
    load_gold_labels,
)
from fonduer.supervision.lf_helpers import *
from hardware_spaces import OmniNgramsPart, OmniNgramsTemp
from hardware_utils import entity_level_f1, load_hardware_labels

logger = logging.getLogger(__name__)
ATTRIBUTE = "stg_temp_max"
DB = "e2e_test"


#  @pytest.mark.skipif("CI" not in os.environ, reason="Only run e2e on Travis")
def test_e2e(caplog):
    """Run an end-to-end test on 20 documents of the hardware domain."""
    caplog.set_level(logging.INFO)
    PARALLEL = 2
    max_docs = 12

    session = Meta.init("postgres://localhost:5432/" + DB).Session()

    Part_Attr = candidate_subclass("Part_Attr", ["part", "attr"])

    docs_path = "tests/e2e/data/html/"
    pdf_path = "tests/e2e/data/pdf/"

    doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser = OmniParser(
        structural=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

    num_docs = session.query(Document).count()
    logger.info("Docs: {}".format(num_docs))
    assert num_docs == max_docs

    num_phrases = session.query(Phrase).count()
    logger.info("Phrases: {}".format(num_phrases))
    #  assert num_phrases == 20

    # Divide into test and train
    docs = session.query(Document).order_by(Document.name).all()
    ld = len(docs)

    train_docs = set()
    dev_docs = set()
    test_docs = set()
    splits = (0.5, 0.75)
    data = [(doc.name, doc) for doc in docs]
    data.sort(key=lambda x: x[0])
    for i, (doc_name, doc) in enumerate(data):
        if i < splits[0] * ld:
            train_docs.add(doc)
        elif i < splits[1] * ld:
            dev_docs.add(doc)
        else:
            test_docs.add(doc)
    logger.info([x.name for x in train_docs])

    attr_matcher = RegexMatchSpan(rgx=r"(?:[1][5-9]|20)[05]", longest_match_only=False)

    ### Transistor Naming Conventions as Regular Expressions ###
    eeca_rgx = r"([ABC][A-Z][WXYZ]?[0-9]{3,5}(?:[A-Z]){0,5}[0-9]?[A-Z]?(?:-[A-Z0-9]{1,7})?(?:[-][A-Z0-9]{1,2})?(?:\/DG)?)"
    jedec_rgx = r"(2N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)"
    jis_rgx = r"(2S[ABCDEFGHJKMQRSTVZ]{1}[\d]{2,4})"
    others_rgx = r"((?:NSVBC|SMBT|MJ|MJE|MPS|MRF|RCA|TIP|ZTX|ZT|ZXT|TIS|TIPL|DTC|MMBT|SMMBT|PZT|FZT|STD|BUV|PBSS|KSC|CXT|FCX|CMPT){1}[\d]{2,4}[A-Z]{0,5}(?:-[A-Z0-9]{0,6})?(?:[-][A-Z0-9]{0,1})?)"

    part_rgx = "|".join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx])
    part_rgx_matcher = RegexMatchSpan(rgx=part_rgx, longest_match_only=True)

    def get_digikey_parts_set(path):
        """
        Reads in the digikey part dictionary and yeilds each part.
        """
        all_parts = set()
        with open(path, "r") as csvinput:
            reader = csv.reader(csvinput)
            for line in reader:
                (part, url) = line
                all_parts.add(part)
        return all_parts

    ### Dictionary of known transistor parts ###
    dict_path = "tests/e2e/data/digikey_part_dictionary.csv"
    part_dict_matcher = DictionaryMatch(d=get_digikey_parts_set(dict_path))

    def common_prefix_length_diff(str1, str2):
        for i in range(min(len(str1), len(str2))):
            if str1[i] != str2[i]:
                return min(len(str1), len(str2)) - i
        return 0

    def part_file_name_conditions(attr):
        file_name = attr.sentence.document.name
        if len(file_name.split("_")) != 2:
            return False
        if attr.get_span()[0] == "-":
            return False
        name = attr.get_span().replace("-", "")
        return (
            any(char.isdigit() for char in name)
            and any(char.isalpha() for char in name)
            and common_prefix_length_diff(file_name.split("_")[1], name) <= 2
        )

    add_rgx = "^[A-Z0-9\-]{5,15}$"

    part_file_name_lambda_matcher = LambdaFunctionMatcher(
        func=part_file_name_conditions
    )
    part_file_name_matcher = Intersect(
        RegexMatchSpan(rgx=add_rgx, longest_match_only=True),
        part_file_name_lambda_matcher,
    )

    part_matcher = Union(part_rgx_matcher, part_dict_matcher, part_file_name_matcher)

    part_ngrams = OmniNgramsPart(parts_by_doc=None, n_max=3)
    attr_ngrams = OmniNgramsTemp(n_max=2)

    def stg_temp_filter(c):
        (part, attr) = c
        if same_table((part, attr)):
            return is_horz_aligned((part, attr)) or is_vert_aligned((part, attr))
        return True

    candidate_filter = stg_temp_filter

    candidate_extractor = CandidateExtractor(
        Part_Attr,
        [part_ngrams, attr_ngrams],
        [part_matcher, attr_matcher],
        candidate_filter=candidate_filter,
    )

    candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)

    train_cands = session.query(Part_Attr).filter(Part_Attr.split == 0).all()
    logger.info("Number of candidates: {}".format(len(train_cands)))

    for i, docs in enumerate([dev_docs, test_docs]):
        candidate_extractor.apply(docs, split=i + 1)
        logger.info(
            "Number of candidates: {}".format(
                session.query(Part_Attr).filter(Part_Attr.split == i + 1).count()
            )
        )

    featurizer = BatchFeatureAnnotator(Part_Attr)
    F_train = featurizer.apply(split=0, replace_key_set=True, parallelism=PARALLEL)
    logger.info(F_train.shape)
    F_dev = featurizer.apply(split=1, replace_key_set=False, parallelism=PARALLEL)
    logger.info(F_dev.shape)
    F_test = featurizer.apply(split=2, replace_key_set=False, parallelism=PARALLEL)
    logger.info(F_test.shape)

    gold_file = "tests/e2e/data/hardware_tutorial_gold.csv"
    load_hardware_labels(
        session, Part_Attr, gold_file, ATTRIBUTE, annotator_name="gold"
    )

    def LF_storage_row(c):
        return 1 if "storage" in get_row_ngrams(c.attr) else 0

    def LF_temperature_row(c):
        return 1 if "temperature" in get_row_ngrams(c.attr) else 0

    def LF_operating_row(c):
        return 1 if "operating" in get_row_ngrams(c.attr) else 0

    def LF_tstg_row(c):
        return 1 if overlap(["tstg", "stg", "ts"], list(get_row_ngrams(c.attr))) else 0

    def LF_to_left(c):
        return 1 if "to" in get_left_ngrams(c.attr, window=2) else 0

    def LF_negative_number_left(c):
        return (
            1
            if any(
                [
                    re.match(r"-\s*\d+", ngram)
                    for ngram in get_left_ngrams(c.attr, window=4)
                ]
            )
            else 0
        )

    stg_temp_lfs = [
        LF_storage_row,
        LF_operating_row,
        LF_temperature_row,
        LF_tstg_row,
        LF_to_left,
        LF_negative_number_left,
    ]

    labeler = BatchLabelAnnotator(Part_Attr, lfs=stg_temp_lfs)
    L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)
    logger.info(L_train.shape)

    L_gold_train = load_gold_labels(session, annotator_name="gold", split=0)

    gen_model = GenerativeModel()
    gen_model.train(
        L_train, epochs=500, decay=0.9, step_size=0.001 / L_train.shape[0], reg_param=0
    )
    logger.info("LF Accuracy: {}".format(gen_model.weights.lf_accuracy))

    L_gold_dev = load_gold_labels(session, annotator_name="gold", split=1)

    train_marginals = gen_model.marginals(L_train)

    disc_model = SparseLogisticRegression()
    disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)

    L_gold_test = load_gold_labels(session, annotator_name="gold", split=2)

    test_candidates = [F_test.get_candidate(session, i) for i in range(F_test.shape[0])]
    test_score = disc_model.predictions(F_test)
    true_pred = [test_candidates[_] for _ in np.nditer(np.where(test_score > 0))]

    pickle_file = "tests/e2e/data/parts_by_doc_dict.pkl"
    with open(pickle_file, "rb") as f:
        parts_by_doc = pickle.load(f)

    (TP, FP, FN) = entity_level_f1(
        true_pred, gold_file, ATTRIBUTE, test_docs, parts_by_doc=parts_by_doc
    )

    tp_len = len(TP)
    fp_len = len(FP)
    fn_len = len(FN)
    prec = tp_len / (tp_len + fp_len) if tp_len + fp_len > 0 else float("nan")
    rec = tp_len / (tp_len + fn_len) if tp_len + fn_len > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")

    logger.info("prec: {}".format(prec))
    logger.info("rec: {}".format(rec))
    logger.info("f1: {}".format(f1))

    assert f1 < 0.7 and f1 > 0.4

    def LF_test_condition_aligned(c):
        return (
            -1
            if overlap(["test", "condition"], list(get_aligned_ngrams(c.attr)))
            else 0
        )

    def LF_collector_aligned(c):
        return (
            -1
            if overlap(
                [
                    "collector",
                    "collector-current",
                    "collector-base",
                    "collector-emitter",
                ],
                list(get_aligned_ngrams(c.attr)),
            )
            else 0
        )

    def LF_current_aligned(c):
        return (
            -1
            if overlap(["current", "dc", "ic"], list(get_aligned_ngrams(c.attr)))
            else 0
        )

    def LF_voltage_row_temp(c):
        return (
            -1
            if overlap(
                ["voltage", "cbo", "ceo", "ebo", "v"], list(get_aligned_ngrams(c.attr))
            )
            else 0
        )

    def LF_voltage_row_part(c):
        return (
            -1
            if overlap(
                ["voltage", "cbo", "ceo", "ebo", "v"], list(get_aligned_ngrams(c.attr))
            )
            else 0
        )

    def LF_typ_row(c):
        return -1 if overlap(["typ", "typ."], list(get_row_ngrams(c.attr))) else 0

    def LF_complement_left_row(c):
        return (
            -1
            if (
                overlap(
                    ["complement", "complementary"],
                    chain.from_iterable(
                        [get_row_ngrams(c.part), get_left_ngrams(c.part, window=10)]
                    ),
                )
            )
            else 0
        )

    def LF_too_many_numbers_row(c):
        num_numbers = list(get_row_ngrams(c.attr, attrib="ner_tags")).count("number")
        return -1 if num_numbers >= 3 else 0

    def LF_temp_on_high_page_num(c):
        return -1 if c.attr.get_attrib_tokens("page")[0] > 2 else 0

    def LF_temp_outside_table(c):
        return -1 if not c.attr.sentence.is_tabular() is None else 0

    def LF_not_temp_relevant(c):
        return (
            -1
            if not overlap(
                ["storage", "temperature", "tstg", "stg", "ts"],
                list(get_aligned_ngrams(c.attr)),
            )
            else 0
        )

    stg_temp_lfs_2 = [
        LF_test_condition_aligned,
        LF_collector_aligned,
        LF_current_aligned,
        LF_voltage_row_temp,
        LF_voltage_row_part,
        LF_typ_row,
        LF_complement_left_row,
        LF_too_many_numbers_row,
        LF_temp_on_high_page_num,
        LF_temp_outside_table,
        LF_not_temp_relevant,
    ]

    labeler = BatchLabelAnnotator(Part_Attr, lfs=stg_temp_lfs_2)
    L_train = labeler.apply(
        split=0, clear=False, update_keys=True, update_values=True, parallelism=PARALLEL
    )
    gen_model = GenerativeModel()
    gen_model.train(
        L_train, epochs=500, decay=0.9, step_size=0.001 / L_train.shape[0], reg_param=0
    )
    train_marginals = gen_model.marginals(L_train)

    disc_model = SparseLogisticRegression()
    disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)

    test_candidates = [F_test.get_candidate(session, i) for i in range(F_test.shape[0])]
    test_score = disc_model.predictions(F_test)
    true_pred = [test_candidates[_] for _ in np.nditer(np.where(test_score > 0))]

    (TP, FP, FN) = entity_level_f1(
        true_pred, gold_file, ATTRIBUTE, test_docs, parts_by_doc=parts_by_doc
    )

    tp_len = len(TP)
    fp_len = len(FP)
    fn_len = len(FN)
    prec = tp_len / (tp_len + fp_len) if tp_len + fp_len > 0 else float("nan")
    rec = tp_len / (tp_len + fn_len) if tp_len + fn_len > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")

    logger.info("prec: {}".format(prec))
    logger.info("rec: {}".format(rec))
    logger.info("f1: {}".format(f1))

    assert f1 > 0.7
