import pytest

from fonduer import Meta

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
    Meta.init("postgresql://localhost:5432/" + "cand_test").Session()
    assert Meta.DBNAME == "cand_test"
