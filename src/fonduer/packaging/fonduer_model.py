"""Customized MLflow model for Fonduer."""
import logging
import os
import sys
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Union

import cloudpickle as pickle
import emmental
import numpy as np
import torch
import yaml
from emmental.model import EmmentalModel
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.model_utils import _get_flavor_configuration
from pandas import DataFrame
from scipy.sparse import csr_matrix
from snorkel.labeling.model import LabelModel

from fonduer import init_logging
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.candidates import CandidateExtractorUDF
from fonduer.candidates.mentions import MentionExtractorUDF
from fonduer.candidates.models import (
    Candidate,
    Mention,
    candidate_subclass,
    mention_subclass,
)
from fonduer.candidates.models.mention import mention_subclasses
from fonduer.features.feature_extractors import FeatureExtractor
from fonduer.features.featurizer import Featurizer, FeaturizerUDF
from fonduer.parser import Parser
from fonduer.parser.models import Document
from fonduer.parser.parser import ParserUDF
from fonduer.parser.preprocessors import DocPreprocessor
from fonduer.supervision.labeler import Labeler, LabelerUDF
from fonduer.utils.utils_udf import _convert_mappings_to_matrix, unshift_label_matrix

logger = logging.getLogger(__name__)

MODEL_TYPE = "model_type"


class FonduerModel(pyfunc.PythonModel):
    """A custom MLflow model for Fonduer.

    This class is intended to be subclassed.
    """

    def _classify(self, doc: Document) -> DataFrame:
        """Classify candidates by an Emmental model (or by a label model)."""
        raise NotImplementedError()

    def predict(self, model_input: DataFrame) -> DataFrame:
        """Take html_path (and pdf_path) as input and return extracted information.

        This method is required and its signature is defined by the MLflow's convention.
        See MLflow_ for more details.

        .. _MLflow:
            https://www.mlflow.org/docs/latest/models.html#python-function-python-function

        :param model_input: Pandas DataFrame with rows as docs and colums as params.
            params should include "html_path" and can optionally include "pdf_path".
        :return: Pandas DataFrame containing the output from :func:`_classify`, which
            depends on how it is implemented by a subclass.
        """
        df = DataFrame()
        for index, row in model_input.iterrows():
            output = self._process(
                row["html_path"], row["pdf_path"] if "pdf_path" in row.keys() else None
            )
            output["html_path"] = row["html_path"]
            df = df.append(output)
        return df

    def _process(self, html_path: str, pdf_path: Optional[str] = None) -> DataFrame:
        """Run the whole pipeline of Fonduer.

        :param html_path: a path of an HTML file or a directory containing files.
        :param pdf_path: a path of a PDF file or a directory containing files.
        """
        if not os.path.exists(html_path):
            raise ValueError("html_path should be a file/directory path")
        # Parse docs
        doc = next(
            self.preprocessor._parse_file(html_path, os.path.basename(html_path))
        )

        logger.info(f"Parsing {html_path}")
        doc = self.parser.apply(doc, pdf_path=pdf_path)

        logger.info(f"Extracting mentions from {html_path}")
        doc = self.mention_extractor.apply(doc)

        logger.info(f"Extracting candidates from {html_path}")
        doc = self.candidate_extractor.apply(doc, split=2)

        logger.info(f"Classifying candidates from {html_path}")
        df = self._classify(doc)
        return df

    @staticmethod
    def convert_features_to_matrix(
        features: List[Dict[str, Any]], keys: List[str]
    ) -> csr_matrix:
        """Convert features (the output from FeaturizerUDF.apply) into a sparse matrix.

        :param features: a list of feature mapping (key: key, value=feature).
        :param keys: a list of all keys.
        """
        return _convert_mappings_to_matrix(features, keys)

    @staticmethod
    def convert_labels_to_matrix(
        labels: List[Dict[str, Any]], keys: List[str]
    ) -> np.ndarray:
        """Convert labels (the output from LabelerUDF.apply) into a dense matrix.

        Note that the input labels are 0-indexed (``{0, 1, ..., k}``),
        while the output labels are -1-indexed (``{-1, 0, ..., k-1}``).

        :param labels: a list of label mapping (key: key, value=label).
        :param keys: a list of all keys.
        """
        return unshift_label_matrix(_convert_mappings_to_matrix(labels, keys))


def _load_pyfunc(model_path: str) -> Any:
    """Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``."""
    # Load mention_classes
    _load_mention_classes(model_path)
    # Load candiate_classes
    _load_candidate_classes(model_path)
    # Load a pickled model
    model = pickle.load(open(os.path.join(model_path, "model.pkl"), "rb"))
    fonduer_model = model["fonduer_model"]
    fonduer_model.preprocessor = model["preprosessor"]
    fonduer_model.parser = ParserUDF(**model["parser"])
    fonduer_model.mention_extractor = MentionExtractorUDF(**model["mention_extractor"])
    fonduer_model.candidate_extractor = CandidateExtractorUDF(
        **model["candidate_extractor"]
    )

    # Configure logging for Fonduer
    init_logging(log_dir="logs")

    pyfunc_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME
    )
    candidate_classes = fonduer_model.candidate_extractor.candidate_classes

    fonduer_model.model_type = pyfunc_conf.get(MODEL_TYPE, "emmental")
    if fonduer_model.model_type == "emmental":
        emmental.init()
        fonduer_model.featurizer = FeaturizerUDF(candidate_classes, FeatureExtractor())
        fonduer_model.key_names = model["feature_keys"]
        fonduer_model.word2id = model["word2id"]
        fonduer_model.emmental_model = _load_emmental_model(model["emmental_model"])
    else:
        fonduer_model.labeler = LabelerUDF(candidate_classes)
        fonduer_model.key_names = model["labeler_keys"]
        fonduer_model.lfs = model["lfs"]
        fonduer_model.label_models = []
        for state_dict in model["label_models_state_dict"]:
            label_model = LabelModel()
            label_model.__dict__.update(state_dict)
            fonduer_model.label_models.append(label_model)
    return fonduer_model


def log_model(
    fonduer_model: FonduerModel,
    artifact_path: str,
    preprocessor: DocPreprocessor,
    parser: Parser,
    mention_extractor: MentionExtractor,
    candidate_extractor: CandidateExtractor,
    conda_env: Optional[Union[Dict, str]] = None,
    code_paths: Optional[List[str]] = None,
    model_type: Optional[str] = "emmental",
    labeler: Optional[Labeler] = None,
    lfs: Optional[List[List[Callable]]] = None,
    label_models: Optional[List[LabelModel]] = None,
    featurizer: Optional[Featurizer] = None,
    emmental_model: Optional[EmmentalModel] = None,
    word2id: Optional[Dict] = None,
) -> None:
    """Log a Fonduer model as an MLflow artifact for the current run.

    :param fonduer_model: Fonduer model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param preprocessor: the doc preprocessor.
    :param parser: self-explanatory
    :param mention_extractor: self-explanatory
    :param candidate_extractor: self-explanatory
    :param conda_env: Either a dictionary representation of a Conda environment
        or the path to a Conda environment yaml file.
    :param code_paths: A list of local filesystem paths to Python file dependencies,
        or directories containing file dependencies. These files are prepended to the
        system path when the model is loaded.
    :param model_type: the model type, either "emmental" or "label",
        defaults to "emmental".
    :param labeler: a labeler, defaults to None.
    :param lfs: a list of list of labeling functions.
    :param label_models: a list of label models, defaults to None.
    :param featurizer: a featurizer, defaults to None.
    :param emmental_model: an Emmental model, defaults to None.
    :param word2id: a word embedding map.
    """
    Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        fonduer_model=fonduer_model,
        preprocessor=preprocessor,
        parser=parser,
        mention_extractor=mention_extractor,
        candidate_extractor=candidate_extractor,
        conda_env=conda_env,
        code_paths=code_paths,
        model_type=model_type,
        labeler=labeler,
        lfs=lfs,
        label_models=label_models,
        featurizer=featurizer,
        emmental_model=emmental_model,
        word2id=word2id,
    )


def save_model(
    fonduer_model: FonduerModel,
    path: str,
    preprocessor: DocPreprocessor,
    parser: Parser,
    mention_extractor: MentionExtractor,
    candidate_extractor: CandidateExtractor,
    mlflow_model: Model = Model(),
    conda_env: Optional[Union[Dict, str]] = None,
    code_paths: Optional[List[str]] = None,
    model_type: Optional[str] = "emmental",
    labeler: Optional[Labeler] = None,
    lfs: Optional[List[List[Callable]]] = None,
    label_models: Optional[List[LabelModel]] = None,
    featurizer: Optional[Featurizer] = None,
    emmental_model: Optional[EmmentalModel] = None,
    word2id: Optional[Dict] = None,
) -> None:
    """Save a Fonduer model to a path on the local file system.

    :param fonduer_model: Fonduer model to be saved.
    :param path: the path on the local file system.
    :param preprocessor: the doc preprocessor.
    :param parser: self-explanatory
    :param mention_extractor: self-explanatory
    :param candidate_extractor: self-explanatory
    :param mlflow_model: model configuration.
    :param conda_env: Either a dictionary representation of a Conda environment
        or the path to a Conda environment yaml file.
    :param code_paths: A list of local filesystem paths to Python file dependencies,
        or directories containing file dependencies. These files are prepended to the
        system path when the model is loaded.
    :param model_type: the model type, either "emmental" or "label",
        defaults to "emmental".
    :param labeler: a labeler, defaults to None.
    :param lfs: a list of list of labeling functions.
    :param label_models: a list of label models, defaults to None.
    :param featurizer: a featurizer, defaults to None.
    :param emmental_model: an Emmental model, defaults to None.
    :param word2id: a word embedding map.
    """
    os.makedirs(path)
    model_code_path = os.path.join(path, pyfunc.CODE)
    os.makedirs(model_code_path)

    # Save mention_classes and candidate_classes
    _save_mention_classes(mention_extractor.udf_init_kwargs["mention_classes"], path)
    _save_candidate_classes(
        candidate_extractor.udf_init_kwargs["candidate_classes"], path
    )
    # Makes lfs unpicklable w/o the module (ie fonduer_lfs.py)
    # https://github.com/cloudpipe/cloudpickle/issues/206#issuecomment-555939172
    modules = []
    if model_type == "label":
        for _ in lfs:
            for lf in _:
                modules.append(lf.__module__)
                lf.__module__ = "__main__"

    # Note that instances of ParserUDF and other UDF theselves are not picklable.
    # https://stackoverflow.com/a/52026025
    model = {
        "fonduer_model": fonduer_model,
        "preprosessor": preprocessor,
        "parser": parser.udf_init_kwargs,
        "mention_extractor": mention_extractor.udf_init_kwargs,
        "candidate_extractor": candidate_extractor.udf_init_kwargs,
    }
    if model_type == "emmental":
        key_names = [key.name for key in featurizer.get_keys()]
        model["feature_keys"] = key_names
        model["word2id"] = word2id
        model["emmental_model"] = _save_emmental_model(emmental_model)
    else:
        key_names = [key.name for key in labeler.get_keys()]
        model["labeler_keys"] = key_names
        model["lfs"] = lfs
        model["label_models_state_dict"] = [
            label_model.__dict__ for label_model in label_models
        ]
    pickle.dump(model, open(os.path.join(path, "model.pkl"), "wb"))

    # Restore __module__ back to the original
    if model_type == "label":
        for _ in lfs:
            for lf in _:
                lf.__module__ = modules.pop()

    # Create a conda yaml file.
    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = _get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Copy code_paths.
    if code_paths is not None:
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=model_code_path)

    mlflow_model.add_flavor(
        pyfunc.FLAVOR_NAME,
        code=pyfunc.CODE,
        loader_module=__name__,
        model_type=model_type,
        env=conda_env_subpath,
    )
    mlflow_model.save(os.path.join(path, "MLmodel"))


def _get_default_conda_env() -> Optional[Dict[str, Any]]:
    """Get default Conda environment.

    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import torch

    import fonduer

    return _mlflow_conda_env(
        additional_conda_deps=[
            "pytorch={}".format(torch.__version__),
            "psycopg2",
            "pip",
        ],
        additional_pip_deps=["fonduer=={}".format(fonduer.__version__)],
        additional_conda_channels=["pytorch"],
    )


def _save_emmental_model(emmental_model: EmmentalModel) -> bytes:
    buffer = BytesIO()
    torch.save(emmental_model, buffer)
    buffer.seek(0)
    return buffer.read()


def _load_emmental_model(b: bytes) -> EmmentalModel:
    buffer = BytesIO()
    buffer.write(b)
    buffer.seek(0)
    return torch.load(buffer)


def _save_mention_classes(mention_classes: List[Mention], path: str) -> None:
    pickle.dump(
        [
            {
                "class_name": mention_class.__name__,
                "cardinality": mention_class.cardinality,
                "values": mention_class.values,
                "table_name": mention_class.__tablename__,
            }
            for mention_class in mention_classes
        ],
        open(os.path.join(path, "mention_classes.pkl"), "wb"),
    )


def _load_mention_classes(path: str) -> None:
    for kwargs in pickle.load(open(os.path.join(path, "mention_classes.pkl"), "rb")):
        mention_subclass(**kwargs)


def _save_candidate_classes(candidate_classes: List[Candidate], path: str) -> None:
    pickle.dump(
        [
            {
                "class_name": candidate_class.__name__,
                "mention_class_names": [
                    candidate_class.__name__
                    for candidate_class in candidate_class.mentions
                ],
                "table_name": candidate_class.__tablename__,
                "cardinality": candidate_class.cardinality,
                "values": candidate_class.values,
            }
            for candidate_class in candidate_classes
        ],
        open(os.path.join(path, "candidate_classes.pkl"), "wb"),
    )


def _load_candidate_classes(path: str) -> None:
    for kwargs in pickle.load(open(os.path.join(path, "candidate_classes.pkl"), "rb")):
        # Convert the classnames of mention to mention_classes
        kwargs["args"] = [
            mention_subclasses[mention_class_name][0]
            for mention_class_name in kwargs.pop("mention_class_names")
        ]
        candidate_subclass(**kwargs)
