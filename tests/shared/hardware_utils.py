import codecs
import csv
import logging
from builtins import range

from snorkel.labeling import labeling_function

from fonduer.candidates.models import Candidate
from fonduer.learning.utils import confusion_matrix

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


logger = logging.getLogger(__name__)

ABSTAIN = -1
FALSE = 0
TRUE = 1


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
                if val == TRUE:
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


gold_dict = get_gold_dict(
    "tests/data/hardware_tutorial_gold.csv", attribute="stg_temp_max"
)


@labeling_function()
def gold(c: Candidate) -> int:
    doc = (c[0].context.sentence.document.name).upper()
    part = (c[0].context.get_span()).upper()
    val = ("".join(c[1].context.get_span().split())).upper()

    if (doc, part, val) in gold_dict:
        return TRUE
    else:
        return FALSE


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
        gold_file = 'tutorials/tables/data/hardware/hardware_gold.csv'
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
        logger.info(f"Gold File: {gold_file}\n Attribute: {attribute}")
        logger.error("Gold set is empty.")
        return
    # Turn CandidateSet into set of tuples
    logger.info("Preparing candidates...")
    entities = set()
    for i, c in enumerate(tqdm(candidates)):
        part = c[0].context.get_span()
        doc = c[0].context.sentence.document.name.upper()
        if attribute:
            val = c[1].context.get_span()
        for p in get_implied_parts(part, doc, parts_by_doc):
            if attribute:
                entities.add((doc, p, val))
            else:
                entities.add((doc, p))

    (TP_set, FP_set, FN_set) = confusion_matrix(entities, gold_set)
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
