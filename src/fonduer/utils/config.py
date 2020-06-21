"""Fonduer basic config and config utils."""
import logging
import os
from typing import Dict

import yaml

MAX_CONFIG_SEARCH_DEPTH = 25  # Max num of parent directories to look for config
logger = logging.getLogger(__name__)

default = {
    "featurization": {
        "textual": {
            "window_feature": {"size": 3, "combinations": True, "isolated": True},
            "word_feature": {"window": 7},
        },
        "tabular": {
            "unary_features": {
                "attrib": ["words"],
                "get_cell_ngrams": {"max": 2},
                "get_head_ngrams": {"max": 2},
                "get_row_ngrams": {"max": 2},
                "get_col_ngrams": {"max": 2},
            },
            "multinary_features": {
                "min_row_diff": {"absolute": False},
                "min_col_diff": {"absolute": False},
            },
        },
    },
    "learning": {
        "LSTM": {
            "emb_dim": 100,
            "hidden_dim": 100,
            "attention": True,
            "dropout": 0.1,
            "bidirectional": True,
            "bias": False,
        },
        "LogisticRegression": {"hidden_dim": 100, "bias": False},
    },
}


def _merge(x: Dict, y: Dict) -> Dict:
    """Merge two nested dictionaries. Overwrite values in x with values in y."""
    merged = {**x, **y}

    xkeys = x.keys()

    for key in xkeys:
        if isinstance(x[key], dict) and key in y:
            merged[key] = _merge(x[key], y[key])

    return merged


def get_config(path: str = os.getcwd()) -> Dict:
    """Search for settings file in root of project and its parents."""
    config = default
    tries = 0
    current_dir = path
    while current_dir and tries < MAX_CONFIG_SEARCH_DEPTH:
        potential_path = os.path.join(current_dir, ".fonduer-config.yaml")
        if os.path.exists(potential_path):
            with open(potential_path, "r") as f:
                config = _merge(config, yaml.safe_load(f))
            logger.debug(f"Loading Fonduer config from {potential_path}.")
            break

        new_dir = os.path.split(current_dir)[0]
        if current_dir == new_dir:
            logger.debug("Unable to find config file. Using defaults.")
            break
        current_dir = new_dir
        tries += 1

    return config
