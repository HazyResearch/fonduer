import logging
import os
import tempfile
from builtins import object
from datetime import datetime
from urllib.parse import urlparse

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


def init_logging(
    log_dir=tempfile.gettempdir(),
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.INFO,
):
    """Configures logging to output to the provided log_dir.

    Will use a nested directory whose name is the current timestamp.

    :param log_dir: The directory to store logs in.
    :type log_dir: str
    :param format: The logging format string to use.
    :type format: str
    :param level: The logging level to use, e.g., logging.INFO.
    """

    # Generate a new directory using the log_dir, if it doesn't exist
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, dt)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Configure the logger using the provided path
    logging.basicConfig(
        format=format,
        level=level,
        handlers=[
            logging.FileHandler(os.path.join(log_path, "fonduer.log")),
            logging.StreamHandler(),
        ],
    )

    # Notify user of log location
    if not Meta.log_path:
        logger.info(f"Setting logging directory to: {log_path}")

    Meta.log_path = log_path


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
            f"{e}\n"
            f"To resolve this error, check our FAQs at: "
            f"https://fonduer.readthedocs.io/en/latest/user/faqs.html"
        )
    except Exception as e:
        raise ValueError(
            f"{e}\n"
            f"Meta variables have not been initialized with "
            f"a valid postgres connection string.\n"
            f"Use the form: "
            f"postgresql://<user>:<pw>@<host>:<port>/<database_name>"
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
    log_path = None

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
                f"Connecting user:{Meta.DBUSER} "
                f"to {Meta.DBHOST}:{Meta.DBPORT}/{Meta.DBNAME}"
            )
            Meta._init_db()

        if not Meta.log_path:
            init_logging()

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
