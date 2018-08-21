#! /usr/bin/env python
import logging
import os

from fonduer.utils.config import get_config


def test_load_config(caplog):
    """Simple sanity check for loading feature config."""
    caplog.set_level(logging.INFO)

    # Check that default is loaded
    defaults = get_config()
    assert defaults["featurization"]["content"]["window_feature"]["size"] == 3

    # Check that file is loaded if present
    settings = get_config(os.path.dirname(__file__))
    assert settings["featurization"]["content"]["window_feature"]["size"] == 8

    # Check that defaults are used for unspecified settings
    assert (
        settings["featurization"]["table"]["unary_features"]["get_head_ngrams"]["max"]
        == 2
    )
