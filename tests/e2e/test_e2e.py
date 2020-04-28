import logging
import os
import pickle

import emmental
import numpy as np
import pytest
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.modules.embedding_module import EmbeddingModule
from snorkel.labeling.model import LabelModel

import fonduer
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.features import Featurizer
from fonduer.features.models import Feature, FeatureKey
from fonduer.learning.dataset import FonduerDataset
from fonduer.learning.task import create_task
from fonduer.learning.utils import collect_word_counter
from fonduer.parser import Parser
from fonduer.parser.models import Document, Sentence
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.supervision import Labeler
from fonduer.supervision.models import GoldLabel, Label, LabelKey
from tests.shared.hardware_lfs import (
    TRUE,
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
from tests.shared.hardware_subclasses import Part, PartTemp, PartVolt, Temp, Volt
from tests.shared.hardware_throttlers import temp_throttler, volt_throttler
from tests.shared.hardware_utils import entity_level_f1, gold

logger = logging.getLogger(__name__)
ATTRIBUTE = "stg_temp_max"
DB = "e2e_test"
if "CI" in os.environ:
    CONN_STRING = (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        + f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{DB}"
    )
else:
    CONN_STRING = f"postgresql://127.0.0.1:5432/{DB}"


@pytest.mark.skipif("CI" not in os.environ, reason="Only run e2e on GitHub Actions")
def test_e2e():
    """Run an end-to-end test on documents of the hardware domain."""
    # GitHub Actions gives 2 cores
    # help.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners
    PARALLEL = 2

    max_docs = 12

    fonduer.init_logging(
        format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
        level=logging.DEBUG,
    )

    session = fonduer.Meta.init(CONN_STRING).Session()

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
    logger.info(f"Docs: {num_docs}")
    assert num_docs == max_docs

    num_sentences = session.query(Sentence).count()
    logger.info(f"Sentences: {num_sentences}")

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

    mention_extractor = MentionExtractor(
        session,
        [Part, Temp, Volt],
        [part_ngrams, temp_ngrams, volt_ngrams],
        [part_matcher, temp_matcher, volt_matcher],
    )

    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Part).count() == 299
    assert session.query(Temp).count() == 138
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
    candidate_extractor = CandidateExtractor(
        session, [PartTemp, PartVolt], throttlers=[temp_throttler, volt_throttler]
    )

    for i, docs in enumerate([train_docs, dev_docs, test_docs]):
        candidate_extractor.apply(docs, split=i, parallelism=PARALLEL)

    assert session.query(PartTemp).filter(PartTemp.split == 0).count() == 3493
    assert session.query(PartTemp).filter(PartTemp.split == 1).count() == 61
    assert session.query(PartTemp).filter(PartTemp.split == 2).count() == 416
    assert session.query(PartVolt).count() == 4282

    # Grab candidate lists
    train_cands = candidate_extractor.get_candidates(split=0, sort=True)
    dev_cands = candidate_extractor.get_candidates(split=1, sort=True)
    test_cands = candidate_extractor.get_candidates(split=2, sort=True)

    # Candidate lists should be deterministically sorted.
    assert (
        "112823::implicit_span_mention:11059:11065:part_expander:0"
        == train_cands[0][0][0].context.get_stable_id()
    )
    assert (
        "112823::implicit_span_mention:2752:2754:temp_expander:0"
        == train_cands[0][0][1].context.get_stable_id()
    )

    assert len(train_cands) == 2
    assert len(train_cands[0]) == 3493
    assert (
        len(
            candidate_extractor.get_candidates(
                docs=[session.query(Document).filter(Document.name == "112823").first()]
            )[0]
        )
        == 1432
    )

    # Featurization
    featurizer = Featurizer(session, [PartTemp, PartVolt])

    # Test that FeatureKey is properly reset
    featurizer.apply(split=1, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 214
    assert session.query(FeatureKey).count() == 1260

    # Test Dropping FeatureKey
    # Should force a row deletion
    featurizer.drop_keys(["DDL_e1_W_LEFT_POS_3_[NNP NN IN]"])
    assert session.query(FeatureKey).count() == 1259

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
    assert session.query(FeatureKey).count() == 1259

    # Inserting the removed key
    featurizer.upsert_keys(
        ["DDL_e1_LEMMA_SEQ_[bc182]"], candidate_classes=[PartTemp, PartVolt]
    )
    assert set(
        session.query(FeatureKey)
        .filter(FeatureKey.name == "DDL_e1_LEMMA_SEQ_[bc182]")
        .one()
        .candidate_classes
    ) == {"part_temp", "part_volt"}
    assert session.query(FeatureKey).count() == 1259
    # Removing the key again
    featurizer.drop_keys(["DDL_e1_LEMMA_SEQ_[bc182]"], candidate_classes=[PartVolt])

    # Removing the last relation from a key should delete the row
    featurizer.drop_keys(["DDL_e1_LEMMA_SEQ_[bc182]"], candidate_classes=[PartTemp])
    assert session.query(FeatureKey).count() == 1258
    session.query(Feature).delete(synchronize_session="fetch")
    session.query(FeatureKey).delete(synchronize_session="fetch")

    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 6478
    assert session.query(FeatureKey).count() == 4538
    F_train = featurizer.get_feature_matrices(train_cands)
    assert F_train[0].shape == (3493, 4538)
    assert F_train[1].shape == (2985, 4538)
    assert len(featurizer.get_keys()) == 4538

    featurizer.apply(split=1, parallelism=PARALLEL)
    assert session.query(Feature).count() == 6692
    assert session.query(FeatureKey).count() == 4538
    F_dev = featurizer.get_feature_matrices(dev_cands)
    assert F_dev[0].shape == (61, 4538)
    assert F_dev[1].shape == (153, 4538)

    featurizer.apply(split=2, parallelism=PARALLEL)
    assert session.query(Feature).count() == 8252
    assert session.query(FeatureKey).count() == 4538
    F_test = featurizer.get_feature_matrices(test_cands)
    assert F_test[0].shape == (416, 4538)
    assert F_test[1].shape == (1144, 4538)

    gold_file = "tests/data/hardware_tutorial_gold.csv"

    labeler = Labeler(session, [PartTemp, PartVolt])

    # This should raise an error, since gold labels are not yet loaded.
    with pytest.raises(ValueError):
        _ = labeler.get_gold_labels(train_cands, annotator="gold")

    labeler.apply(
        docs=last_docs,
        lfs=[[gold], [gold]],
        table=GoldLabel,
        train=True,
        parallelism=PARALLEL,
    )
    assert session.query(GoldLabel).count() == 8252

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

    with pytest.raises(ValueError):
        labeler.apply(split=0, lfs=stg_temp_lfs, train=True, parallelism=PARALLEL)

    labeler.apply(
        docs=train_docs,
        lfs=[stg_temp_lfs, ce_v_max_lfs],
        train=True,
        parallelism=PARALLEL,
    )
    assert session.query(Label).count() == 6478
    assert session.query(LabelKey).count() == 9
    L_train = labeler.get_label_matrices(train_cands)
    assert L_train[0].shape == (3493, 9)
    assert L_train[1].shape == (2985, 9)
    assert len(labeler.get_keys()) == 9

    # Test Dropping LabelerKey
    labeler.drop_keys(["LF_storage_row"])
    assert len(labeler.get_keys()) == 8

    # Test Upserting LabelerKey
    labeler.upsert_keys(["LF_storage_row"])
    assert "LF_storage_row" in [label.name for label in labeler.get_keys()]

    L_train_gold = labeler.get_gold_labels(train_cands)
    assert L_train_gold[0].shape == (3493, 1)

    L_train_gold = labeler.get_gold_labels(train_cands, annotator="gold")
    assert L_train_gold[0].shape == (3493, 1)

    label_model = LabelModel(cardinality=2)
    label_model.fit(L_train=L_train[0], n_epochs=500, seed=1234, log_freq=100)

    train_marginals = label_model.predict_proba(L_train[0])

    # Collect word counter
    word_counter = collect_word_counter(train_cands)

    emmental.init(fonduer.Meta.log_path)

    # Training config
    config = {
        "meta_config": {"verbose": False},
        "model_config": {"model_path": None, "device": 0, "dataparallel": False},
        "learner_config": {
            "n_epochs": 5,
            "optimizer_config": {"lr": 0.001, "l2": 0.0},
            "task_scheduler": "round_robin",
        },
        "logging_config": {
            "evaluation_freq": 1,
            "counter_unit": "epoch",
            "checkpointing": False,
            "checkpointer_config": {
                "checkpoint_metric": {f"{ATTRIBUTE}/{ATTRIBUTE}/train/loss": "min"},
                "checkpoint_freq": 1,
                "checkpoint_runway": 2,
                "clear_intermediate_checkpoints": True,
                "clear_all_checkpoints": True,
            },
        },
    }
    emmental.Meta.update_config(config=config)

    # Generate word embedding module
    arity = 2
    # Geneate special tokens
    specials = []
    for i in range(arity):
        specials += [f"~~[[{i}", f"{i}]]~~"]

    emb_layer = EmbeddingModule(
        word_counter=word_counter, word_dim=300, specials=specials
    )

    diffs = train_marginals.max(axis=1) - train_marginals.min(axis=1)
    train_idxs = np.where(diffs > 1e-6)[0]

    train_dataloader = EmmentalDataLoader(
        task_to_label_dict={ATTRIBUTE: "labels"},
        dataset=FonduerDataset(
            ATTRIBUTE,
            train_cands[0],
            F_train[0],
            emb_layer.word2id,
            train_marginals,
            train_idxs,
        ),
        split="train",
        batch_size=100,
        shuffle=True,
    )

    tasks = create_task(
        ATTRIBUTE, 2, F_train[0].shape[1], 2, emb_layer, model="LogisticRegression"
    )

    model = EmmentalModel(name=f"{ATTRIBUTE}_task")

    for task in tasks:
        model.add_task(task)

    emmental_learner = EmmentalLearner()
    emmental_learner.learn(model, [train_dataloader])

    test_dataloader = EmmentalDataLoader(
        task_to_label_dict={ATTRIBUTE: "labels"},
        dataset=FonduerDataset(
            ATTRIBUTE, test_cands[0], F_test[0], emb_layer.word2id, 2
        ),
        split="test",
        batch_size=100,
        shuffle=False,
    )

    test_preds = model.predict(test_dataloader, return_preds=True)
    positive = np.where(np.array(test_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.6)
    true_pred = [test_cands[0][_] for _ in positive[0]]

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

    logger.info(f"prec: {prec}")
    logger.info(f"rec: {rec}")
    logger.info(f"f1: {f1}")

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
    assert session.query(Label).count() == 6478
    assert session.query(LabelKey).count() == 16
    L_train = labeler.get_label_matrices(train_cands)
    assert L_train[0].shape == (3493, 16)

    label_model = LabelModel(cardinality=2)
    label_model.fit(L_train=L_train[0], n_epochs=500, seed=1234, log_freq=100)

    train_marginals = label_model.predict_proba(L_train[0])

    diffs = train_marginals.max(axis=1) - train_marginals.min(axis=1)
    train_idxs = np.where(diffs > 1e-6)[0]

    train_dataloader = EmmentalDataLoader(
        task_to_label_dict={ATTRIBUTE: "labels"},
        dataset=FonduerDataset(
            ATTRIBUTE,
            train_cands[0],
            F_train[0],
            emb_layer.word2id,
            train_marginals,
            train_idxs,
        ),
        split="train",
        batch_size=100,
        shuffle=True,
    )

    valid_dataloader = EmmentalDataLoader(
        task_to_label_dict={ATTRIBUTE: "labels"},
        dataset=FonduerDataset(
            ATTRIBUTE,
            train_cands[0],
            F_train[0],
            emb_layer.word2id,
            np.argmax(train_marginals, axis=1),
            train_idxs,
        ),
        split="valid",
        batch_size=100,
        shuffle=False,
    )

    # Testing STL LogisticRegression
    emmental.Meta.reset()
    emmental.init(fonduer.Meta.log_path)
    emmental.Meta.update_config(config=config)

    tasks = create_task(
        ATTRIBUTE,
        2,
        F_train[0].shape[1],
        2,
        emb_layer,
        model="LogisticRegression",
        mode="STL",
    )

    model = EmmentalModel(name=f"{ATTRIBUTE}_task")

    for task in tasks:
        model.add_task(task)

    emmental_learner = EmmentalLearner()
    emmental_learner.learn(model, [train_dataloader, valid_dataloader])

    test_preds = model.predict(test_dataloader, return_preds=True)
    positive = np.where(np.array(test_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.7)
    true_pred = [test_cands[0][_] for _ in positive[0]]

    (TP, FP, FN) = entity_level_f1(
        true_pred, gold_file, ATTRIBUTE, test_docs, parts_by_doc=parts_by_doc
    )

    tp_len = len(TP)
    fp_len = len(FP)
    fn_len = len(FN)
    prec = tp_len / (tp_len + fp_len) if tp_len + fp_len > 0 else float("nan")
    rec = tp_len / (tp_len + fn_len) if tp_len + fn_len > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")

    logger.info(f"prec: {prec}")
    logger.info(f"rec: {rec}")
    logger.info(f"f1: {f1}")

    assert f1 > 0.7

    # MLflow model
    from tests.shared.hardware_fonduer_model import HardwareFonduerModel

    code_paths = [
        "tests",
    ]

    from fonduer.utils import fonduer_model

    fonduer_model.save_model(
        HardwareFonduerModel(),
        "fonduer_disc_model",
        code_paths=code_paths,
        preprocessor=doc_preprocessor,
        parser=corpus_parser,
        mention_extractor=mention_extractor,
        candidate_extractor=candidate_extractor,
        featurizer=featurizer,
        disc_model=model,
        word2id=emb_layer.word2id,
    )

    # Testing STL LSTM
    emmental.Meta.reset()
    emmental.init(fonduer.Meta.log_path)
    emmental.Meta.update_config(config=config)

    tasks = create_task(
        ATTRIBUTE, 2, F_train[0].shape[1], 2, emb_layer, model="LSTM", mode="STL"
    )

    model = EmmentalModel(name=f"{ATTRIBUTE}_task")

    for task in tasks:
        model.add_task(task)

    emmental_learner = EmmentalLearner()
    emmental_learner.learn(model, [train_dataloader])

    test_preds = model.predict(test_dataloader, return_preds=True)
    positive = np.where(np.array(test_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.7)
    true_pred = [test_cands[0][_] for _ in positive[0]]

    (TP, FP, FN) = entity_level_f1(
        true_pred, gold_file, ATTRIBUTE, test_docs, parts_by_doc=parts_by_doc
    )

    tp_len = len(TP)
    fp_len = len(FP)
    fn_len = len(FN)
    prec = tp_len / (tp_len + fp_len) if tp_len + fp_len > 0 else float("nan")
    rec = tp_len / (tp_len + fn_len) if tp_len + fn_len > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")

    logger.info(f"prec: {prec}")
    logger.info(f"rec: {rec}")
    logger.info(f"f1: {f1}")

    assert f1 > 0.7

    # Testing MTL LogisticRegression
    emmental.Meta.reset()
    emmental.init(fonduer.Meta.log_path)
    emmental.Meta.update_config(config=config)

    tasks = create_task(
        ATTRIBUTE,
        2,
        F_train[0].shape[1],
        2,
        emb_layer,
        model="LogisticRegression",
        mode="MTL",
    )

    model = EmmentalModel(name=f"{ATTRIBUTE}_task")

    for task in tasks:
        model.add_task(task)

    emmental_learner = EmmentalLearner()
    emmental_learner.learn(model, [train_dataloader, valid_dataloader])

    test_preds = model.predict(test_dataloader, return_preds=True)
    positive = np.where(np.array(test_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.7)
    true_pred = [test_cands[0][_] for _ in positive[0]]

    (TP, FP, FN) = entity_level_f1(
        true_pred, gold_file, ATTRIBUTE, test_docs, parts_by_doc=parts_by_doc
    )

    tp_len = len(TP)
    fp_len = len(FP)
    fn_len = len(FN)
    prec = tp_len / (tp_len + fp_len) if tp_len + fp_len > 0 else float("nan")
    rec = tp_len / (tp_len + fn_len) if tp_len + fn_len > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")

    logger.info(f"prec: {prec}")
    logger.info(f"rec: {rec}")
    logger.info(f"f1: {f1}")

    assert f1 > 0.7

    # Testing LSTM
    emmental.Meta.reset()
    emmental.init(fonduer.Meta.log_path)
    emmental.Meta.update_config(config=config)

    tasks = create_task(
        ATTRIBUTE, 2, F_train[0].shape[1], 2, emb_layer, model="LSTM", mode="MTL"
    )

    model = EmmentalModel(name=f"{ATTRIBUTE}_task")

    for task in tasks:
        model.add_task(task)

    emmental_learner = EmmentalLearner()
    emmental_learner.learn(model, [train_dataloader])

    test_preds = model.predict(test_dataloader, return_preds=True)
    positive = np.where(np.array(test_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.7)
    true_pred = [test_cands[0][_] for _ in positive[0]]

    (TP, FP, FN) = entity_level_f1(
        true_pred, gold_file, ATTRIBUTE, test_docs, parts_by_doc=parts_by_doc
    )

    tp_len = len(TP)
    fp_len = len(FP)
    fn_len = len(FN)
    prec = tp_len / (tp_len + fp_len) if tp_len + fp_len > 0 else float("nan")
    rec = tp_len / (tp_len + fn_len) if tp_len + fn_len > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")

    logger.info(f"prec: {prec}")
    logger.info(f"rec: {rec}")
    logger.info(f"f1: {f1}")

    assert f1 > 0.7
