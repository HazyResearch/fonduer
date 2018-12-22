#! /usr/bin/env python
import logging
import os
import pickle

import numpy as np
import pytest
from metal.label_model import LabelModel

from fonduer import Meta
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.models import Candidate, candidate_subclass, mention_subclass
from fonduer.features import Featurizer
from fonduer.features.models import Feature, FeatureKey
from fonduer.learning import (
    LSTM,
    LogisticRegression,
    SparseLogisticRegression,
    SparseLSTM,
)
from fonduer.parser import Parser
from fonduer.parser.models import Document, Sentence
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.supervision import Labeler
from fonduer.supervision.models import GoldLabel, Label, LabelKey
from tests.shared.hardware_lfs import (
    LF_bad_keywords_in_row,
    LF_collector_aligned,
    LF_complement_left_row,
    LF_current_aligned,
    LF_current_in_row,
    LF_negative_number_left,
    LF_non_ce_voltages_in_row,
    LF_not_temp_relevant,
    LF_operating_row,
    LF_storage_row,
    LF_temp_on_high_page_num,
    LF_temp_outside_table,
    LF_temperature_row,
    LF_test_condition_aligned,
    LF_to_left,
    LF_too_many_numbers_row,
    LF_tstg_row,
    LF_typ_row,
    LF_voltage_row_part,
    LF_voltage_row_temp,
)
from tests.shared.hardware_matchers import part_matcher, temp_matcher, volt_matcher
from tests.shared.hardware_spaces import (
    MentionNgramsPart,
    MentionNgramsTemp,
    MentionNgramsVolt,
)
from tests.shared.hardware_throttlers import temp_throttler, volt_throttler
from tests.shared.hardware_utils import entity_level_f1, load_hardware_labels

logger = logging.getLogger(__name__)
ATTRIBUTE = "stg_temp_max"
DB = "e2e_test"


@pytest.mark.skipif("CI" not in os.environ, reason="Only run incremental on Travis")
def test_incremental(caplog):
    """Run an end-to-end test on incremental additions."""
    caplog.set_level(logging.INFO)

    PARALLEL = 1

    max_docs = 1

    session = Meta.init("postgresql://localhost:5432/" + DB).Session()

    docs_path = "tests/data/html/dtc114w.html"
    pdf_path = "tests/data/pdf/dtc114w.pdf"

    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser = Parser(
        session,
        parallelism=PARALLEL,
        structural=True,
        lingual=True,
        visual=True,
        pdf_path=pdf_path,
    )
    corpus_parser.apply(doc_preprocessor)

    num_docs = session.query(Document).count()
    logger.info("Docs: {}".format(num_docs))
    assert num_docs == max_docs

    docs = corpus_parser.get_documents()
    last_docs = corpus_parser.get_documents()

    assert len(docs[0].sentences) == len(last_docs[0].sentences)

    # Mention Extraction
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
    temp_ngrams = MentionNgramsTemp(n_max=2)

    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")

    mention_extractor = MentionExtractor(
        session, [Part, Temp], [part_ngrams, temp_ngrams], [part_matcher, temp_matcher]
    )

    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Part).count() == 11
    assert session.query(Temp).count() == 9

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])

    candidate_extractor = CandidateExtractor(
        session, [PartTemp], throttlers=[temp_throttler]
    )

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    assert session.query(PartTemp).filter(PartTemp.split == 0).count() == 78
    assert session.query(Candidate).count() == 78

    # Grab candidate lists
    train_cands = candidate_extractor.get_candidates(split=0)
    assert len(train_cands) == 1
    assert len(train_cands[0]) == 78

    # Featurization
    featurizer = Featurizer(session, [PartTemp])

    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 78
    assert session.query(FeatureKey).count() == 496

    F_train = featurizer.get_feature_matrices(train_cands)
    assert F_train[0].shape == (78, 496)
    assert len(featurizer.get_keys()) == 496

    # Test Dropping FeatureKey
    featurizer.drop_keys(["CORE_e1_LENGTH_1"])
    assert session.query(FeatureKey).count() == 495

    stg_temp_lfs = [
        LF_storage_row,
        LF_operating_row,
        LF_temperature_row,
        LF_tstg_row,
        LF_to_left,
        LF_negative_number_left,
    ]

    labeler = Labeler(session, [PartTemp])

    labeler.apply(split=0, lfs=[stg_temp_lfs], train=True, parallelism=PARALLEL)
    assert session.query(Label).count() == 78

    # Only 5 because LF_operating_row doesn't apply to the first test doc
    assert session.query(LabelKey).count() == 5
    L_train = labeler.get_label_matrices(train_cands)
    assert L_train[0].shape == (78, 5)
    assert len(labeler.get_keys()) == 5

    docs_path = "tests/data/html/112823.html"
    pdf_path = "tests/data/pdf/112823.pdf"

    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser.apply(doc_preprocessor, pdf_path=pdf_path, clear=False)

    assert len(corpus_parser.get_documents()) == 2

    new_docs = corpus_parser.get_last_documents()

    assert len(new_docs) == 1
    assert new_docs[0].name == "112823"

    # Get mentions from just the new docs
    mention_extractor.apply(new_docs, parallelism=PARALLEL, clear=False)

    assert session.query(Part).count() == 81
    assert session.query(Temp).count() == 33

    # Just run candidate extraction and assign to split 0
    candidate_extractor.apply(new_docs, split=0, parallelism=PARALLEL, clear=False)

    # Grab candidate lists
    train_cands = candidate_extractor.get_candidates(split=0)
    assert len(train_cands) == 1
    assert len(train_cands[0]) == 1574

    # Update features
    featurizer.update(new_docs, parallelism=PARALLEL)
    assert session.query(Feature).count() == 1574
    assert session.query(FeatureKey).count() == 2425
    F_train = featurizer.get_feature_matrices(train_cands)
    assert F_train[0].shape == (1574, 2425)
    assert len(featurizer.get_keys()) == 2425

    # Update Labels
    labeler.update(new_docs, lfs=[stg_temp_lfs], parallelism=PARALLEL)
    assert session.query(Label).count() == 1574
    assert session.query(LabelKey).count() == 6
    L_train = labeler.get_label_matrices(train_cands)
    assert L_train[0].shape == (1574, 6)

    # Test clear
    featurizer.clear(train=True)
    assert session.query(FeatureKey).count() == 0


@pytest.mark.skipif("CI" not in os.environ, reason="Only run e2e on Travis")
def test_e2e(caplog):
    """Run an end-to-end test on documents of the hardware domain."""
    caplog.set_level(logging.INFO)

    PARALLEL = 4

    max_docs = 12

    session = Meta.init("postgresql://localhost:5432/" + DB).Session()

    docs_path = "tests/data/html/"
    pdf_path = "tests/data/pdf/"

    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser = Parser(
        session,
        parallelism=PARALLEL,
        structural=True,
        lingual=True,
        visual=True,
        pdf_path=pdf_path,
    )
    corpus_parser.apply(doc_preprocessor)
    assert session.query(Document).count() == max_docs

    num_docs = session.query(Document).count()
    logger.info("Docs: {}".format(num_docs))
    assert num_docs == max_docs

    num_sentences = session.query(Sentence).count()
    logger.info("Sentences: {}".format(num_sentences))

    # Divide into test and train
    docs = sorted(corpus_parser.get_documents())
    last_docs = sorted(corpus_parser.get_last_documents())

    ld = len(docs)
    assert ld == len(last_docs)
    assert len(docs[0].sentences) == len(last_docs[0].sentences)

    assert len(docs[0].sentences) == 799
    assert len(docs[1].sentences) == 663
    assert len(docs[2].sentences) == 784
    assert len(docs[3].sentences) == 661
    assert len(docs[4].sentences) == 513
    assert len(docs[5].sentences) == 700
    assert len(docs[6].sentences) == 528
    assert len(docs[7].sentences) == 161
    assert len(docs[8].sentences) == 228
    assert len(docs[9].sentences) == 511
    assert len(docs[10].sentences) == 331
    assert len(docs[11].sentences) == 528

    # Check table numbers
    assert len(docs[0].tables) == 9
    assert len(docs[1].tables) == 9
    assert len(docs[2].tables) == 14
    assert len(docs[3].tables) == 11
    assert len(docs[4].tables) == 11
    assert len(docs[5].tables) == 10
    assert len(docs[6].tables) == 10
    assert len(docs[7].tables) == 2
    assert len(docs[8].tables) == 7
    assert len(docs[9].tables) == 10
    assert len(docs[10].tables) == 6
    assert len(docs[11].tables) == 9

    # Check figure numbers
    assert len(docs[0].figures) == 32
    assert len(docs[1].figures) == 11
    assert len(docs[2].figures) == 38
    assert len(docs[3].figures) == 31
    assert len(docs[4].figures) == 7
    assert len(docs[5].figures) == 38
    assert len(docs[6].figures) == 10
    assert len(docs[7].figures) == 31
    assert len(docs[8].figures) == 4
    assert len(docs[9].figures) == 27
    assert len(docs[10].figures) == 5
    assert len(docs[11].figures) == 27

    # Check caption numbers
    assert len(docs[0].captions) == 0
    assert len(docs[1].captions) == 0
    assert len(docs[2].captions) == 0
    assert len(docs[3].captions) == 0
    assert len(docs[4].captions) == 0
    assert len(docs[5].captions) == 0
    assert len(docs[6].captions) == 0
    assert len(docs[7].captions) == 0
    assert len(docs[8].captions) == 0
    assert len(docs[9].captions) == 0
    assert len(docs[10].captions) == 0
    assert len(docs[11].captions) == 0

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

    # NOTE: With multi-relation support, return values of getting candidates,
    # mentions, or sparse matrices are formatted as a list of lists. This means
    # that with a single relation, we need to index into the list of lists to
    # get the candidates/mentions/sparse matrix for a particular relation or
    # mention.

    # Mention Extraction
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
    temp_ngrams = MentionNgramsTemp(n_max=2)
    volt_ngrams = MentionNgramsVolt(n_max=1)

    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")
    Volt = mention_subclass("Volt")

    mention_extractor = MentionExtractor(
        session,
        [Part, Temp, Volt],
        [part_ngrams, temp_ngrams, volt_ngrams],
        [part_matcher, temp_matcher, volt_matcher],
    )

    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Part).count() == 299
    assert session.query(Temp).count() == 147
    assert session.query(Volt).count() == 140
    assert len(mention_extractor.get_mentions()) == 3
    assert len(mention_extractor.get_mentions()[0]) == 299
    assert (
        len(
            mention_extractor.get_mentions(
                docs=[session.query(Document).filter(Document.name == "112823").first()]
            )[0]
        )
        == 70
    )

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])
    PartVolt = candidate_subclass("PartVolt", [Part, Volt])

    candidate_extractor = CandidateExtractor(
        session, [PartTemp, PartVolt], throttlers=[temp_throttler, volt_throttler]
    )

    for i, docs in enumerate([train_docs, dev_docs, test_docs]):
        candidate_extractor.apply(docs, split=i, parallelism=PARALLEL)

    assert session.query(PartTemp).filter(PartTemp.split == 0).count() == 3684
    assert session.query(PartTemp).filter(PartTemp.split == 1).count() == 72
    assert session.query(PartTemp).filter(PartTemp.split == 2).count() == 448
    assert session.query(PartVolt).count() == 4282

    # Grab candidate lists
    train_cands = candidate_extractor.get_candidates(split=0, sort=True)
    dev_cands = candidate_extractor.get_candidates(split=1, sort=True)
    test_cands = candidate_extractor.get_candidates(split=2, sort=True)
    assert len(train_cands) == 2
    assert len(train_cands[0]) == 3684
    assert (
        len(
            candidate_extractor.get_candidates(
                docs=[session.query(Document).filter(Document.name == "112823").first()]
            )[0]
        )
        == 1496
    )

    # Featurization
    featurizer = Featurizer(session, [PartTemp, PartVolt])

    # Test that FeatureKey is properly reset
    featurizer.apply(split=1, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 225
    assert session.query(FeatureKey).count() == 1179

    # Test Dropping FeatureKey
    # Should force a row deletion
    featurizer.drop_keys(["DDL_e1_W_LEFT_POS_3_[NFP NN NFP]"])
    assert session.query(FeatureKey).count() == 1178

    # Should only remove the part_volt as a relation and leave part_temp
    assert set(
        session.query(FeatureKey)
        .filter(FeatureKey.name == "DDL_e1_LEMMA_SEQ_[bc182]")
        .one()
        .candidate_classes
    ) == {"part_temp", "part_volt"}
    featurizer.drop_keys(["DDL_e1_LEMMA_SEQ_[bc182]"], candidate_classes=[PartVolt])
    assert session.query(FeatureKey).filter(
        FeatureKey.name == "DDL_e1_LEMMA_SEQ_[bc182]"
    ).one().candidate_classes == ["part_temp"]
    assert session.query(FeatureKey).count() == 1178
    # Removing the last relation from a key should delete the row
    featurizer.drop_keys(["DDL_e1_LEMMA_SEQ_[bc182]"], candidate_classes=[PartTemp])
    assert session.query(FeatureKey).count() == 1177
    session.query(Feature).delete()
    session.query(FeatureKey).delete()

    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 6669
    assert session.query(FeatureKey).count() == 4161
    F_train = featurizer.get_feature_matrices(train_cands)
    assert F_train[0].shape == (3684, 4161)
    assert F_train[1].shape == (2985, 4161)
    assert len(featurizer.get_keys()) == 4161

    featurizer.apply(split=1, parallelism=PARALLEL)
    assert session.query(Feature).count() == 6894
    assert session.query(FeatureKey).count() == 4161
    F_dev = featurizer.get_feature_matrices(dev_cands)
    assert F_dev[0].shape == (72, 4161)
    assert F_dev[1].shape == (153, 4161)

    featurizer.apply(split=2, parallelism=PARALLEL)
    assert session.query(Feature).count() == 8486
    assert session.query(FeatureKey).count() == 4161
    F_test = featurizer.get_feature_matrices(test_cands)
    assert F_test[0].shape == (448, 4161)
    assert F_test[1].shape == (1144, 4161)

    gold_file = "tests/data/hardware_tutorial_gold.csv"
    load_hardware_labels(session, PartTemp, gold_file, ATTRIBUTE, annotator_name="gold")
    assert session.query(GoldLabel).count() == 4204
    load_hardware_labels(session, PartVolt, gold_file, ATTRIBUTE, annotator_name="gold")
    assert session.query(GoldLabel).count() == 8486

    stg_temp_lfs = [
        LF_storage_row,
        LF_operating_row,
        LF_temperature_row,
        LF_tstg_row,
        LF_to_left,
        LF_negative_number_left,
    ]

    ce_v_max_lfs = [
        LF_bad_keywords_in_row,
        LF_current_in_row,
        LF_non_ce_voltages_in_row,
    ]

    labeler = Labeler(session, [PartTemp, PartVolt])

    with pytest.raises(ValueError):
        labeler.apply(split=0, lfs=stg_temp_lfs, train=True, parallelism=PARALLEL)

    labeler.apply(
        split=0, lfs=[stg_temp_lfs, ce_v_max_lfs], train=True, parallelism=PARALLEL
    )
    assert session.query(Label).count() == 6669
    assert session.query(LabelKey).count() == 9
    L_train = labeler.get_label_matrices(train_cands)
    assert L_train[0].shape == (3684, 9)
    assert L_train[1].shape == (2985, 9)
    assert len(labeler.get_keys()) == 9

    L_train_gold = labeler.get_gold_labels(train_cands)
    assert L_train_gold[0].shape == (3684, 1)

    L_train_gold = labeler.get_gold_labels(train_cands, annotator="gold")
    assert L_train_gold[0].shape == (3684, 1)

    gen_model = LabelModel(k=2)
    gen_model.train_model(L_train[0], n_epochs=500, print_every=100)

    train_marginals = gen_model.predict_proba(L_train[0])[:, 1]

    disc_model = LogisticRegression()
    disc_model.train(
        (train_cands[0], F_train[0]), train_marginals, n_epochs=20, lr=0.001
    )

    test_score = disc_model.predictions((test_cands[0], F_test[0]), b=0.6)
    true_pred = [test_cands[0][_] for _ in np.nditer(np.where(test_score > 0))]

    pickle_file = "tests/data/parts_by_doc_dict.pkl"
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

    assert f1 < 0.7 and f1 > 0.3

    stg_temp_lfs_2 = [
        LF_to_left,
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
    labeler.update(split=0, lfs=[stg_temp_lfs_2, ce_v_max_lfs], parallelism=PARALLEL)
    assert session.query(Label).count() == 6669
    assert session.query(LabelKey).count() == 16
    L_train = labeler.get_label_matrices(train_cands)
    assert L_train[0].shape == (3684, 16)

    gen_model = LabelModel(k=2)
    gen_model.train_model(L_train[0], n_epochs=500, print_every=100)

    train_marginals = gen_model.predict_proba(L_train[0])[:, 1]

    disc_model = LogisticRegression()
    disc_model.train(
        (train_cands[0], F_train[0]), train_marginals, n_epochs=20, lr=0.001
    )

    test_score = disc_model.predictions((test_cands[0], F_test[0]), b=0.6)
    true_pred = [test_cands[0][_] for _ in np.nditer(np.where(test_score > 0))]

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

    # Testing LSTM
    disc_model = LSTM()
    disc_model.train(
        (train_cands[0], F_train[0]), train_marginals, n_epochs=5, lr=0.001
    )

    test_score = disc_model.predictions((test_cands[0], F_test[0]), b=0.6)
    true_pred = [test_cands[0][_] for _ in np.nditer(np.where(test_score > 0))]

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

    # Testing Sparse Logistic Regression
    disc_model = SparseLogisticRegression()
    disc_model.train(
        (train_cands[0], F_train[0]), train_marginals, n_epochs=20, lr=0.001
    )

    test_score = disc_model.predictions((test_cands[0], F_test[0]), b=0.6)
    true_pred = [test_cands[0][_] for _ in np.nditer(np.where(test_score > 0))]

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

    # Testing Sparse LSTM
    disc_model = SparseLSTM()
    disc_model.train(
        (train_cands[0], F_train[0]), train_marginals, n_epochs=5, lr=0.001
    )

    test_score = disc_model.predictions((test_cands[0], F_test[0]), b=0.6)
    true_pred = [test_cands[0][_] for _ in np.nditer(np.where(test_score > 0))]

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
