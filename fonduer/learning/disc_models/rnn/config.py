import logging
import os

import yaml

MAX_CONFIG_SEARCH_DEPTH = 25  # Max num of parent directories to look for config
logger = logging.getLogger(__name__)

default = {
    "learning": {
        "LSTM": {
            "emb_dim": 100,
            "hidden_dim": 100,
            "attention": True,
            "dropout": 0.1,
            "bidirectional": True,
            "host_device": "CPU",
            "max_sentence_length": 100,
        }
    }
}


def get_config(path=os.getcwd()):
    """Search for settings file in root of project and its parents."""
    config = default
    tries = 0
    current_dir = path
    while current_dir and tries < MAX_CONFIG_SEARCH_DEPTH:
        potential_path = os.path.join(current_dir, ".fonduer-config.yaml")
        if os.path.exists(potential_path):
            with open(potential_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info("Loading Fonduer config from {}.".format(potential_path))
            break

        new_dir = os.path.split(current_dir)[0]
        if current_dir == new_dir:
            logger.debug("Unable to find config file. Using defaults.")
            break
        current_dir = new_dir
        tries += 1

    return config
