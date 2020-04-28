from emmental.data import EmmentalDataLoader
import numpy as np
from pandas import DataFrame
import pickle

from fonduer.learning.dataset import FonduerDataset
from fonduer.parser.models import Document
from fonduer.utils.fonduer_model import F_matrix, FonduerModel, L_matrix
from tests.shared.hardware_subclasses import Part, PartTemp, PartVolt, Temp, Volt
from tests.shared.hardware_lfs import TRUE
from tests.shared.hardware_utils import get_implied_parts

ATTRIBUTE = "stg_temp_max"


class HardwareFonduerModel(FonduerModel):
    def _classify(self, doc: Document) -> DataFrame:
        # Only one candidate class is used.
        candidate_class = self.candidate_extractor.candidate_classes[0]
        test_cands = getattr(doc, candidate_class.__tablename__ + "s")

        features_list = self.featurizer.apply(doc)
        # Convert features into a sparse matrix
        F_test = F_matrix(features_list[0], self.key_names)

        test_dataloader = EmmentalDataLoader(
            task_to_label_dict={ATTRIBUTE: "labels"},
            dataset=FonduerDataset(
                ATTRIBUTE, test_cands, F_test, self.word2id, 2
            ),
            split="test",
            batch_size=100,
            shuffle=False,
        )

        test_preds = self.disc_model.predict(test_dataloader, return_preds=True)
        positive = np.where(np.array(test_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.7)
        true_preds = [test_cands[_] for _ in positive[0]]

        pickle_file = "tests/data/parts_by_doc_dict.pkl"
        with open(pickle_file, "rb") as f:
            parts_by_doc = pickle.load(f)

        df = DataFrame()
        for c in true_preds:
            part = c[0].context.get_span()
            doc = c[0].context.sentence.document.name.upper()
            val = c[1].context.get_span()
            for p in get_implied_parts(part, doc, parts_by_doc):
                entity_relation = (doc, p, val)
                df = df.append(
                    DataFrame([entity_relation],
                    columns=["doc", "part", "val"]
                    )
                )
        return df
