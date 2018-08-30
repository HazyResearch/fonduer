import logging

from fonduer import Meta

logger = logging.getLogger(__name__)


def add_keys(session, key_table, keys):
    """Bulk add annotation keys to the specified table.

    :param table: The sqlalchemy class to insert into.
    :param keys: A list of strings to insert into the table.
    """
    # Do nothing if empty
    if not keys:
        logger.warning(
            "Attempted to insert empty list of keys to {}.".format(
                key_table.__tablename__
            )
        )
        return

    # NOTE: If you pprint these values, it may look funny because of the
    # newlines and tabs as whitespace characters in these names. Use normal
    # print.
    existing_keys = [k.name for k in session.query(key_table).all()]
    new_keys = [k for k in keys if k not in existing_keys]

    # Bulk insert all new feature keys
    if new_keys:
        Meta.engine.execute(
            key_table.__table__.insert(), [{"name": key} for key in new_keys]
        )
