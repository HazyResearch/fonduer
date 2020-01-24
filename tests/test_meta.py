import logging

import pytest

from fonduer import Meta
from fonduer.candidates.models import mention_subclass

logger = logging.getLogger(__name__)
DB = "meta_test"


def test_meta_connection_strings():
    """Simple sanity checks for validating postgres connection strings."""

    with pytest.raises(ValueError):
        Meta.init("postgresql" + DB).Session()

    with pytest.raises(ValueError):
        Meta.init("sqlite://somethingsilly" + DB).Session()

    with pytest.raises(ValueError):
        Meta.init("postgresql://somethingsilly:5432/").Session()

    Meta.init("postgresql://localhost:5432/" + DB).Session()
    assert Meta.DBNAME == DB


def test_subclass_before_meta_init():
    """Test if it is possible to create a mention (candidate) subclass even before Meta
    is initialized.
    """
    Part = mention_subclass("Part")
    logger.info(f"Create a mention subclass '{Part.__tablename__}'")
    Meta.init("postgresql://localhost:5432/" + DB).Session()
    Temp = mention_subclass("Temp")
    logger.info(f"Create a mention subclass '{Temp.__tablename__}'")
