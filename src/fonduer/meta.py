import logging
from builtins import object
from urllib.parse import urlparse

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


# Defines procedure for setting up a sessionmaker
def new_sessionmaker():
    # Turning on autocommit for Postgres, see
    # http://oddbird.net/2014/06/14/sqlalchemy-postgres-autocommit/
    # Otherwise any e.g. query starts a transaction, locking tables... very
    # bad for e.g. multiple notebooks open, multiple processes, etc.
    try:
        engine = create_engine(
            Meta.conn_string,
            client_encoding="utf8",
            use_batch_mode=True,
            isolation_level="AUTOCOMMIT",
        )
        engine.connect()
    except sqlalchemy.exc.OperationalError as e:
        raise ValueError(
            "{}\n{}".format(
                e,
                "To resolve this error, check our FAQs at: "
                + "https://fonduer.readthedocs.io/en/latest/user/faqs.html",
            )
        )
    except Exception as e:
        raise ValueError(
            "{}\n{}".format(
                e,
                "Meta variables have not been initialized with "
                + "a valid postgres connection string.\n"
                + "Use the form: "
                + "postgresql://<user>:<pw>@<host>:<port>/<database_name>",
            )
        )
    # New sessionmaker
    return sessionmaker(bind=engine)


def _update_meta(conn_string):
    """Update Meta class."""
    url = urlparse(conn_string)
    Meta.conn_string = conn_string
    Meta.DBNAME = url.path[1:]
    Meta.DBUSER = url.username
    Meta.DBPWD = url.password
    Meta.DBHOST = url.hostname
    Meta.DBPORT = url.port
    Meta.postgres = url.scheme.startswith("postgresql")


class Meta(object):
    """Singleton-like metadata class for all global variables.

    Adapted from the Unique Design Pattern:
    https://stackoverflow.com/questions/1318406/why-is-the-borg-pattern-better-than-the-singleton-pattern-in-python
    """

    # Static class variables
    conn_string = None
    DBNAME = None
    DBUSER = None
    DBHOST = None
    DBPORT = None
    DBPWD = None
    Session = None
    engine = None
    Base = declarative_base(name="Base", cls=object)
    postgres = False

    @classmethod
    def init(cls, conn_string=None):
        """Return the unique Meta class."""
        if conn_string:
            _update_meta(conn_string)
            # We initialize the engine within the models module because models'
            # schema can depend on which data types are supported by the engine
            Meta.Session = new_sessionmaker()
            Meta.engine = Meta.Session.kw["bind"]
            logger.info(
                "Connecting user:{} to {}:{}/{}".format(
                    Meta.DBUSER, Meta.DBHOST, Meta.DBPORT, Meta.DBNAME
                )
            )
            Meta._init_db()

        return cls

    @classmethod
    def _init_db(cls):
        """ Initialize the storage schema.

        This call must be performed after all classes that extend
        Base are declared to ensure the storage schema is initialized.
        """
        # This list of import defines which SQLAlchemy classes will be
        # initialized when Meta.init() is called. If a sqlalchemy class is not
        # imported before the call to create_all(), it will not be created.
        import fonduer.candidates.models  # noqa
        import fonduer.features.models  # noqa
        import fonduer.learning.models  # noqa
        import fonduer.parser.models  # noqa
        import fonduer.supervision.models  # noqa
        import fonduer.utils.models  # noqa

        logger.info("Initializing the storage schema")
        Meta.Base.metadata.create_all(Meta.engine)
