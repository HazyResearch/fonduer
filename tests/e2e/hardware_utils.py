from __future__ import division

import codecs
import csv
import logging
from builtins import range

from fonduer.supervision.lf_helpers import *
from fonduer.models import GoldLabel, GoldLabelKey
from fonduer.utils import ProgressBar

logger = logging.getLogger(__name__)


def get_gold_dict(
    filename, doc_on=True, part_on=True, val_on=True, attribute=None, docs=None
):
    with codecs.open(filename, encoding="utf-8") as csvfile:
        gold_reader = csv.reader(csvfile)
        gold_dict = set()
        for row in gold_reader:
            (doc, part, attr, val) = row
            if docs is None or doc.upper() in docs:
                if attribute and attr != attribute:
                    continue
                if not val:
                    continue
                else:
                    key = []
                    if doc_on:
                        key.append(doc.upper())
                    if part_on:
                        key.append(part.upper())
                    if val_on:
                        key.append(val.upper())
                    gold_dict.add(tuple(key))
    return gold_dict


def load_hardware_labels(
    session, candidate_class, filename, attrib, annotator_name="gold"
):
    ak = session.query(GoldLabelKey).filter(GoldLabelKey.name == annotator_name).first()
    if ak is None:
        ak = GoldLabelKey(name=annotator_name)
        session.add(ak)
        session.commit()

    candidates = session.query(candidate_class).all()
    gold_dict = get_gold_dict(filename, attribute=attrib)
    cand_total = len(candidates)
    logger.info("Loading {} candidate labels".format(cand_total))
    pb = ProgressBar(cand_total)
    labels = []
    for i, c in enumerate(candidates):
        pb.bar(i)
        doc = (c[0].sentence.document.name).upper()
        part = (c[0].get_span()).upper()
        val = ("".join(c[1].get_span().split())).upper()
        context_stable_ids = "~~".join([i.stable_id for i in c.get_contexts()])
        label = (
            session.query(GoldLabel)
            .filter(GoldLabel.key == ak)
            .filter(GoldLabel.candidate == c)
            .first()
        )
        if label is None:
            if (doc, part, val) in gold_dict:
                label = GoldLabel(candidate=c, key=ak, value=1)
            else:
                label = GoldLabel(candidate=c, key=ak, value=-1)
            session.add(label)
            labels.append(label)
    session.commit()
    pb.close()

    session.commit()
    logger.info("AnnotatorLabels created: %s" % (len(labels),))


def entity_confusion_matrix(pred, gold):
    if not isinstance(pred, set):
        pred = set(pred)
    if not isinstance(gold, set):
        gold = set(gold)
    TP = pred.intersection(gold)
    FP = pred.difference(gold)
    FN = gold.difference(pred)
    return (TP, FP, FN)


def entity_level_f1(
    candidates, gold_file, attribute=None, corpus=None, parts_by_doc=None
):
    """Checks entity-level recall of candidates compared to gold.

    Turns a CandidateSet into a normal set of entity-level tuples
    (doc, part, [attribute_value])
    then compares this to the entity-level tuples found in the gold.

    Example Usage:
        from hardware_utils import entity_level_total_recall
        candidates = # CandidateSet of all candidates you want to consider
        gold_file = os.environ['FONDUERHOME'] + '/tutorials/tables/data/hardware/hardware_gold.csv'
        entity_level_total_recall(candidates, gold_file, 'stg_temp_min')
    """
    docs = [(doc.name).upper() for doc in corpus] if corpus else None
    val_on = attribute is not None
    gold_set = get_gold_dict(
        gold_file,
        docs=docs,
        doc_on=True,
        part_on=True,
        val_on=val_on,
        attribute=attribute,
    )
    if len(gold_set) == 0:
        logger.info("Gold File: {}\n Attribute: {}".format(gold_file, attribute))
        logger.error("Gold set is empty.")
        return
    # Turn CandidateSet into set of tuples
    logger.info("Preparing candidates...")
    pb = ProgressBar(len(candidates))
    entities = set()
    for i, c in enumerate(candidates):
        pb.bar(i)
        part = c[0].get_span()
        doc = c[0].sentence.document.name.upper()
        if attribute:
            val = c[1].get_span()
        for p in get_implied_parts(part, doc, parts_by_doc):
            if attribute:
                entities.add((doc, p, val))
            else:
                entities.add((doc, p))
    pb.close()

    (TP_set, FP_set, FN_set) = entity_confusion_matrix(entities, gold_set)
    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    prec = TP / (TP + FP) if TP + FP > 0 else float("nan")
    rec = TP / (TP + FN) if TP + FN > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")
    logger.info("========================================")
    logger.info("Scoring on Entity-Level Gold Data")
    logger.info("========================================")
    logger.info("Corpus Precision {:.3}".format(prec))
    logger.info("Corpus Recall    {:.3}".format(rec))
    logger.info("Corpus F1        {:.3}".format(f1))
    logger.info("----------------------------------------")
    logger.info("TP: {} | FP: {} | FN: {}".format(TP, FP, FN))
    logger.info("========================================\n")
    return [sorted(list(x)) for x in [TP_set, FP_set, FN_set]]


def get_implied_parts(part, doc, parts_by_doc):
    yield part
    if parts_by_doc:
        for p in parts_by_doc[doc]:
            if p.startswith(part) and len(part) >= 4:
                yield p


def entity_to_candidates(entity, candidate_subset):
    matches = []
    for c in candidate_subset:
        c_entity = tuple(
            [c[0].sentence.document.name.upper()]
            + [c[i].get_span().upper() for i in range(len(c))]
        )
        c_entity = tuple([str(x) for x in c_entity])
        if c_entity == entity:
            matches.append(c)
    return matches
