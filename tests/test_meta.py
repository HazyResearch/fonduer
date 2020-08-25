"""Test Fonduer meta."""
import logging
import os

import psycopg2
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy.exc import OperationalError

from fonduer import Meta
from fonduer.candidates.models import mention_subclass
from tests.conftest import CONN_STRING, DB

logger = logging.getLogger(__name__)


def test_meta_connection_strings(database_session):
    """Simple sanity checks for validating postgres connection strings."""
    with pytest.raises(ValueError):
        Meta.init("postgresql" + DB).Session()

    with pytest.raises(ValueError):
        Meta.init("sqlite://somethingsilly" + DB).Session()

    with pytest.raises(OperationalError):
        Meta.init("postgresql://somethingsilly:5432/").Session()

    session = Meta.init("postgresql://localhost:5432/" + DB).Session()
    engine = session.get_bind()
    session.close()
    engine.dispose()
    assert Meta.DBNAME == DB


def test_subclass_before_meta_init():
    """Test if mention (candidate) subclass can be created before Meta init."""
    # Test if mention (candidate) subclass can be created
    Part = mention_subclass("Part")
    logger.info(f"Create a mention subclass '{Part.__tablename__}'")

    # Setup a database
    con = psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = con.cursor()
    cursor.execute(f'create database "{DB}";')
    session = Meta.init(CONN_STRING).Session()

    # Test if another mention subclass can be created
    Temp = mention_subclass("Temp")
    logger.info(f"Create a mention subclass '{Temp.__tablename__}'")

    # Teardown the database
    session.close()
    Meta.engine.dispose()
    Meta.engine = None

    cursor.execute(f'drop database "{DB}";')
    cursor.close()
    con.close()
