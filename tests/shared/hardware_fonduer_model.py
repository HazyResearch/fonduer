from emmental.data import EmmentalDataLoader
from pandas import DataFrame

from fonduer.learning.dataset import FonduerDataset
from fonduer.parser.models import Document
from fonduer.utils.fonduer_model import F_matrix, FonduerModel, L_matrix
from tests.shared.hardware_subclasses import Part, PartTemp, PartVolt, Temp, Volt

ATTRIBUTE = "stg_temp_max"


class HardwareFonduerModel(FonduerModel):
    def _classify(self, doc: Document) -> DataFrame:
        test_dataloader = EmmentalDataLoader(
            task_to_label_dict={ATTRIBUTE: "labels"},
            dataset=FonduerDataset(
                ATTRIBUTE, test_cands[0], F_test[0], self.word2id, 2
            ),
            split="test",
            batch_size=100,
            shuffle=False,
        )
        df = DataFrame()
        return df
