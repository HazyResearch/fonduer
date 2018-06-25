"""
Subpackage for all built-in Fonduer data models.

After creating additional models that extend fonduer.models.meta.Meta.Base,
they must be added to the database schema. For example, the correct way to
define a new type of Context is:

.. code:: python

    from fonduer.models.context import Context
    from sqlalchemy import Column, String, ForeignKey

    class NewType(Context):
        # Declares name for storage table
        __tablename__ = 'newtype'
        # Connects NewType records to generic Context records
        id = Column(String, ForeignKey('context.id', ondelete='CASCADE'), primary_key=True)

        # Polymorphism information for SQLAlchemy
        __mapper_args__ = {
            'polymorphic_identity': 'newtype',
        }

        # Rest of class definition here

    # Adds the corresponding table to the underlying database's schema
    from fonduer import Meta
    Meta.init_db()
"""
from fonduer.models.meta import Meta

__all__ = ["Meta"]
