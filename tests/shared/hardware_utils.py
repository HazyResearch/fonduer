import codecs
import csv
import logging
from builtins import range

from fonduer.supervision.models import GoldLabel, GoldLabelKey
from fonduer.utils.utils import get_entity

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


logger = logging.getLogger(__name__)

# Define labels
ABSTAIN = 0
FALSE = 1
TRUE = 2


def get_gold_dict(
    filename, session, doc_on=True, part_on=True, val_on=True, attribute=None, docs=None
):
    with codecs.open(filename, encoding="utf-8") as csvfile:
        gold_reader = csv.reader(csvfile)
        gold_dict = set()
        for row in gold_reader:
            (doc, part, attr, val) = row
            if docs is None or doc.upper() in docs:
                if attribute and attr != attribute:
                    continue
                if val == TRUE:
                    continue
                else:
                    key = []
                    if doc_on:
                        entity = get_entity(session, doc.upper())
                        key.append(entity)
                    if part_on:
                        entity = get_entity(session, part.upper())
                        key.append(entity)
                    if val_on:
                        entity = get_entity(session, val.upper())
                        key.append(entity)
                    gold_dict.add(tuple(key))
    return gold_dict


def load_hardware_labels(
    session, candidate_classes, filename, attrib, annotator_name="gold"
):
    """Bulk insert hardware GoldLabels.

    :param session: The database session to use.
    :param candidate_classes: Which candidate_classes to load labels for.
    :param filename: Path to the CSV file containing gold labels.
    :param attrib: Which attributes to load labels for (e.g. "stg_temp_max").
    """
    # Check that candidate_classes is iterable
    candidate_classes = (
        candidate_classes
        if isinstance(candidate_classes, (list, tuple))
        else [candidate_classes]
    )

    ak = session.query(GoldLabelKey).filter(GoldLabelKey.name == annotator_name).first()
    # Add the gold key
    if ak is None:
        ak = GoldLabelKey(
            name=annotator_name,
            candidate_classes=[_.__tablename__ for _ in candidate_classes],
        )
        session.add(ak)
        session.commit()

    # Bulk insert candidate labels
    candidates = []
    for candidate_class in candidate_classes:
        candidates.extend(session.query(candidate_class).all())

    gold_dict = get_gold_dict(filename, session, attribute=attrib)
    cand_total = len(candidates)
    logger.info(f"Loading {cand_total} candidate labels")
    labels = 0

    cands = []
    values = []
    for i, c in enumerate(tqdm(candidates)):
        doc = (c[0].context.sentence.document.name).upper()
        doc_entity = get_entity(session, doc)
        part = (c[0].context.get_span()).upper()
        part_entity = get_entity(session, part)
        val = ("".join(c[1].context.get_span().split())).upper()
        val_entity = get_entity(session, val)

        label = session.query(GoldLabel).filter(GoldLabel.candidate == c).first()
        if label is None:
            if (doc_entity, part_entity, val_entity) in gold_dict:
                values.append(TRUE)
            else:
                values.append(FALSE)

            cands.append(c)
            labels += 1

    # Only insert the labels which were not already present
    session.bulk_insert_mappings(
        GoldLabel,
        [
            {"candidate_id": cand.id, "keys": [annotator_name], "values": [val]}
            for (cand, val) in zip(cands, values)
        ],
    )
    session.commit()

    logger.info(f"GoldLabels created: {labels}")


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
    candidates, gold_file, session, attribute=None, corpus=None, parts_by_doc=None
):
    """Checks entity-level recall of candidates compared to gold.

    Turns a CandidateSet into a normal set of entity-level tuples
    (doc, part, [attribute_value])
    then compares this to the entity-level tuples found in the gold.

    Example Usage:
        from hardware_utils import entity_level_total_recall
        candidates = # CandidateSet of all candidates you want to consider
        gold_file = 'tutorials/tables/data/hardware/hardware_gold.csv'
        entity_level_total_recall(candidates, gold_file, 'stg_temp_min')
    """
    docs = [(doc.name).upper() for doc in corpus] if corpus else None
    val_on = attribute is not None
    gold_set = get_gold_dict(
        gold_file,
        session,
        docs=docs,
        doc_on=True,
        part_on=True,
        val_on=val_on,
        attribute=attribute,
    )
    if len(gold_set) == 0:
        logger.info(f"Gold File: {gold_file}\n Attribute: {attribute}")
        logger.error("Gold set is empty.")
        return
    # Turn CandidateSet into set of tuples
    logger.info("Preparing candidates...")
    entities = set()
    for i, c in enumerate(tqdm(candidates)):
        part = c[0].context.get_span()
        doc = c[0].context.sentence.document.name.upper()
        doc_entity = get_entity(session, doc)
        if attribute:
            val = c[1].context.get_span()
            val_entity = get_entity(session, val)
        for p in get_implied_parts(part, doc, parts_by_doc):
            part_entity = get_entity(session, p)
            if attribute:
                entities.add((doc_entity, part_entity, val_entity))
            else:
                entities.add((doc_entity, part_entity))

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
    logger.info(f"Corpus Precision {prec:.3}")
    logger.info(f"Corpus Recall    {rec:.3}")
    logger.info(f"Corpus F1        {f1:.3}")
    logger.info("----------------------------------------")
    logger.info(f"TP: {TP} | FP: {FP} | FN: {FN}")
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
            [c[0].context.sentence.document.name.upper()]
            + [c[i].context.get_span().upper() for i in range(len(c))]
        )
        c_entity = tuple([str(x) for x in c_entity])
        if c_entity == entity:
            matches.append(c)
    return matches
