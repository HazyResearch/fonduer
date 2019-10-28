import os

from fonduer.utils.config import get_config


def test_load_config():
    """Simple sanity check for loading feature config."""
    # Check that default is loaded
    defaults = get_config()
    assert defaults["featurization"]["textual"]["window_feature"]["size"] == 3
    assert defaults["learning"]["LSTM"]["emb_dim"] == 100
    assert defaults["learning"]["LSTM"]["host_device"] == "CPU"

    # Check that file is loaded if present
    settings = get_config(os.path.dirname(__file__))
    assert settings["featurization"]["textual"]["window_feature"]["size"] == 8
    assert settings["learning"]["LSTM"]["host_device"] == "GPU"

    # Check that defaults are used for unspecified settings
    assert (
        settings["featurization"]["tabular"]["unary_features"]["get_head_ngrams"]["max"]
        == 2
    )
