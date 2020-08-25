import os

import psycopg2
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from fonduer.meta import Meta

DB = "fonduer_test"
if "CI" in os.environ:
    CONN_STRING = (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        + f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{DB}"
    )
else:
    CONN_STRING = f"postgresql://127.0.0.1:5432/{DB}"


@pytest.fixture
def database_session():
    if "CI" in os.environ:
        con = psycopg2.connect(
            host=os.environ["POSTGRES_HOST"],
            port=os.environ["POSTGRES_PORT"],
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
        )
    else:
        con = psycopg2.connect(host="127.0.0.1", port="5432")
    # Setup
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = con.cursor()
    cursor.execute(f'create database "{DB}";')
    session = Meta.init(CONN_STRING).Session()
    yield session

    # Teardown
    engine = session.get_bind()
    session.close()
    engine.dispose()
    Meta.engine = None

    cursor.execute(f'drop database "{DB}";')
    cursor.close()
    con.close()
