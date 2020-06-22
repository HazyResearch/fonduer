"""Fonduer MLflow unit tests."""
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import emmental.meta
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
import yaml
from emmental.model import EmmentalModel
from packaging import version
from snorkel.labeling.model import LabelModel

from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.models import candidate_subclass, mention_subclass
from fonduer.candidates.models.candidate import candidate_subclasses
from fonduer.candidates.models.mention import mention_subclasses
from fonduer.features.featurizer import Featurizer
from fonduer.features.models import FeatureKey
from fonduer.packaging import FonduerModel, log_model, save_model
from fonduer.packaging.fonduer_model import (
    _get_default_conda_env,
    _load_candidate_classes,
    _load_mention_classes,
    _save_candidate_classes,
    _save_mention_classes,
)
from fonduer.parser import Parser
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.supervision.labeler import Labeler
from fonduer.supervision.models import LabelKey
from tests.shared.hardware_fonduer_model import HardwareFonduerModel
from tests.shared.hardware_lfs import LF_storage_row
from tests.shared.hardware_matchers import part_matcher, temp_matcher
from tests.shared.hardware_spaces import MentionNgramsPart, MentionNgramsTemp
from tests.shared.hardware_subclasses import Part, PartTemp, Temp
from tests.shared.hardware_throttlers import temp_throttler

artifact_path = "fonduer_model"


@pytest.fixture
def setup_common_components():
    """Set up mention/candidate extractor."""
    preprocessor = HTMLDocPreprocessor("tests/data/html/")
    parser = Parser(None)
    mention_extractor = MentionExtractor(
        None,
        [Part, Temp],
        [MentionNgramsPart(parts_by_doc=None, n_max=3), MentionNgramsTemp(n_max=2)],
        [part_matcher, temp_matcher],
    )
    candidate_extractor = CandidateExtractor(None, [PartTemp], [temp_throttler])
    return {
        "preprocessor": preprocessor,
        "parser": parser,
        "mention_extractor": mention_extractor,
        "candidate_extractor": candidate_extractor,
    }


def test_convert_features_to_matrix():
    """Test _convert_features_to_matrix."""
    features: List[Dict[str, Any]] = [
        {"keys": ["key1", "key2"], "values": [0.0, 0.1]},
        {"keys": ["key1", "key2"], "values": [1.0, 1.1]},
    ]
    key_names: List[str] = ["key1", "key2"]

    F = FonduerModel.convert_features_to_matrix(features, key_names)
    D = np.array([[0.0, 0.1], [1.0, 1.1]])
    assert (F.todense() == D).all()


def test_convert_features_to_matrix_limited_keys():
    """Test _convert_features_to_matrix with limited keys."""
    features: List[Dict[str, Any]] = [
        {"keys": ["key1", "key2"], "values": [0.0, 0.1]},
        {"keys": ["key1", "key2"], "values": [1.0, 1.1]},
    ]

    F = FonduerModel.convert_features_to_matrix(features, ["key1"])
    D = np.array([[0.0], [1.0]])
    assert (F.todense() == D).all()


def test_convert_labels_to_matrix():
    """Test _convert_labels_to_matrix."""
    labels: List[Dict[str, Any]] = [
        {"keys": ["key1", "key2"], "values": [0, 1]},
        {"keys": ["key1", "key2"], "values": [1, 2]},
    ]
    key_names: List[str] = ["key1", "key2"]

    L = FonduerModel.convert_labels_to_matrix(labels, key_names)
    D = np.array([[-1, 0], [0, 1]])
    assert (L == D).all()


@pytest.mark.dependency()
def test_save_subclasses():
    """Test if subclasses can be saved."""
    mention_class = mention_subclass("test_mention_class")
    _save_mention_classes([mention_class], "./")
    assert os.path.exists("./mention_classes.pkl")

    candidate_class = candidate_subclass("test_candidate_class", [mention_class])
    _save_candidate_classes([candidate_class], "./")
    assert os.path.exists("./candidate_classes.pkl")


@pytest.mark.dependency(depends=["test_save_subclasses"])
def test_load_subclasses():
    """Test if subclasses can be loaded."""
    _load_mention_classes("./")
    assert "test_mention_class" in mention_subclasses
    mention_class, _ = mention_subclasses["test_mention_class"]

    _load_candidate_classes("./")
    assert "test_candidate_class" in candidate_subclasses
    candidate_class, _ = candidate_subclasses["test_candidate_class"]
    assert candidate_class.mentions[0] == mention_class


@pytest.mark.dependency()
def test_save_model(tmp_path: Path, setup_common_components: Dict):
    """Test if a Fonduer model can be saved."""
    kwargs = setup_common_components
    featurizer = Featurizer(None, [PartTemp])
    # Mock the get_keys()
    featurizer.get_keys = MagicMock(return_value=[FeatureKey(name="key1")])
    emmental.meta.init_config()
    save_model(
        HardwareFonduerModel(),
        os.path.join(tmp_path, artifact_path),
        **kwargs,
        code_paths=[
            "tests"
        ],  # pass a directory name to preserver the directory hierarchy
        featurizer=featurizer,
        emmental_model=EmmentalModel(),
        word2id={"foo": 1},
    )
    assert os.path.exists(os.path.join(tmp_path, artifact_path))

    log_model(
        HardwareFonduerModel(),
        artifact_path,
        **kwargs,
        code_paths=[
            "tests"
        ],  # pass a directory name to preserver the directory hierarchy
        featurizer=featurizer,
        emmental_model=EmmentalModel(),
        word2id={"foo": 1},
    )


@pytest.mark.dependency(depends=["test_save_model"])
def test_load_model(tmp_path: Path):
    """Test if a saved model can be loaded."""
    # Load from a saved model
    mlflow.pyfunc.load_model(
        os.path.join(tmp_path, "../test_save_model0", artifact_path)
    )


@pytest.mark.dependency()
def test_save_label_model(tmp_path: Path, setup_common_components: Dict):
    """Test if a Fonduer model with a LabelModel as a classifier."""
    kwargs = setup_common_components
    labeler = Labeler(None, [PartTemp])
    # Mock the get_keys()
    labeler.get_keys = MagicMock(return_value=[LabelKey(name="key1")])
    lfs = [[LF_storage_row]]
    label_models = [LabelModel()]
    save_model(
        HardwareFonduerModel(),
        os.path.join(tmp_path, artifact_path),
        **kwargs,
        code_paths=[
            "tests"
        ],  # pass a directory name to preserver the directory hierarchy
        model_type="label",
        labeler=labeler,
        lfs=lfs,
        label_models=label_models,
    )
    assert os.path.exists(os.path.join(tmp_path, artifact_path))


@pytest.mark.dependency(depends=["test_save_label_model"])
def test_load_label_model(tmp_path: Path):
    """Test if a saved model can be loaded."""
    # Load from a saved model
    mlflow.pyfunc.load_model(
        os.path.join(tmp_path, "../test_save_label_model0", artifact_path)
    )


def test_save_with_conda_yaml(tmp_path: Path, setup_common_components: Dict):
    """Test if a model can be saved with a conda yaml file."""
    kwargs = setup_common_components
    labeler = Labeler(None, [PartTemp])
    # Mock the get_keys()
    labeler.get_keys = MagicMock(return_value=[LabelKey(name="key1")])
    lfs = [[LF_storage_row]]
    label_models = [LabelModel()]

    # Create a conda yaml file
    with open(tmp_path.joinpath("my_conda.yaml"), "w") as f:
        yaml.dump(_get_default_conda_env(), f)

    # Save a model with a conda yaml file.
    save_model(
        HardwareFonduerModel(),
        os.path.join(tmp_path, artifact_path),
        **kwargs,
        conda_env=tmp_path.joinpath("my_conda.yaml"),
        code_paths=[
            "tests"
        ],  # pass a directory name to preserver the directory hierarchy
        model_type="label",
        labeler=labeler,
        lfs=lfs,
        label_models=label_models,
    )
    # Your conda yaml file is saved as "conda.yaml".
    assert os.path.exists(os.path.join(tmp_path, artifact_path, "conda.yaml"))


def test_predict(mocker, setup_common_components: Dict):
    """Test if a Fonduer model can predict."""
    kwargs = setup_common_components
    featurizer = Featurizer(None, [PartTemp])
    # Mock the get_keys()
    featurizer.get_keys = MagicMock(return_value=[FeatureKey(name="key1")])
    emmental.meta.init_config()

    # Log the model with FonduerModel()
    log_model(
        FonduerModel(),
        artifact_path,
        **kwargs,
        code_paths=[
            "tests"
        ],  # pass a directory name to preserver the directory hierarchy
        featurizer=featurizer,
        emmental_model=EmmentalModel(),
        word2id={"foo": 1},
    )
    # Load the model
    fonduer_model = mlflow.pyfunc.load_model(
        os.path.join(mlflow.active_run().info.artifact_uri, artifact_path)
    )
    with pytest.raises(NotImplementedError):
        _ = fonduer_model.predict(
            pd.DataFrame(data={"html_path": ["tests/data/html/112823.html"]})
        )

    # Log the model with HardwareFonduerModel()
    log_model(
        HardwareFonduerModel(),
        artifact_path,
        **kwargs,
        code_paths=[
            "tests"
        ],  # pass a directory name to preserver the directory hierarchy
        featurizer=featurizer,
        emmental_model=EmmentalModel(),
        word2id={"foo": 1},
    )
    # Load the model
    fonduer_model = mlflow.pyfunc.load_model(
        os.path.join(mlflow.active_run().info.artifact_uri, artifact_path)
    )

    # Mock the _classify as we don't test the implementation of _classify here.
    mock_output = pd.DataFrame(data={"col1": ["val1"], "col2": ["val2"]})

    # Input both html_path and pdf_html
    if version.parse(mlflow.__version__) >= version.parse("1.9.0"):
        fonduer_model._model_impl._classify = MagicMock(return_value=mock_output)
        spy = mocker.spy(fonduer_model._model_impl, "_process")
    else:
        fonduer_model._classify = MagicMock(return_value=mock_output)
        spy = mocker.spy(fonduer_model, "_process")
    output = fonduer_model.predict(
        pd.DataFrame(
            data={
                "html_path": ["tests/data/html/112823.html"],
                "pdf_path": ["tests/data/pdf/112823.pdf"],
            }
        )
    )
    spy.assert_called_once_with(
        "tests/data/html/112823.html", "tests/data/pdf/112823.pdf"
    )
    assert output.equals(
        pd.DataFrame(
            data={
                "col1": ["val1"],
                "col2": ["val2"],
                "html_path": ["tests/data/html/112823.html"],
            }
        )
    )

    # Input only html_path
    spy.reset_mock()
    output = fonduer_model.predict(
        pd.DataFrame(data={"html_path": ["tests/data/html/112823.html"]})
    )
    spy.assert_called_once_with("tests/data/html/112823.html", None)
    assert output.equals(
        pd.DataFrame(
            data={
                "col1": ["val1"],
                "col2": ["val2"],
                "html_path": ["tests/data/html/112823.html"],
            }
        )
    )

    # Input html_path that does not exist
    spy.reset_mock()

    with pytest.raises(ValueError):
        _ = fonduer_model.predict(
            pd.DataFrame(data={"html_path": ["tests/data/html/foo.html"]})
        )

    # Test when _classify produces multiple relations per doc.
    mock_output = pd.DataFrame(data={"col0": ["00", "10"], "col1": ["01", "11"]})
    if version.parse(mlflow.__version__) >= version.parse("1.9.0"):
        fonduer_model._model_impl._classify = MagicMock(return_value=mock_output)
    else:
        fonduer_model._classify = MagicMock(return_value=mock_output)
    output = fonduer_model.predict(
        pd.DataFrame(data={"html_path": ["tests/data/html/112823.html"]})
    )
    assert output.equals(
        pd.DataFrame(
            data={
                "col0": ["00", "10"],
                "col1": ["01", "11"],
                "html_path": [
                    "tests/data/html/112823.html",
                    "tests/data/html/112823.html",
                ],
            }
        )
    )
