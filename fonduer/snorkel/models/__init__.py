"""
Subpackage for all built-in Snorkel data models.

After creating additional models that extend snorkel.models.meta.Meta.Base,
they must be added to the database schema. For example, the correct way to
define a new type of Context is:

.. code:: python

    from fonduer.snorkel.models.context import Context
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
    from fonduer.snorkel.models.meta import Meta
    Meta.init_db()
"""
from __future__ import absolute_import

from .meta import Meta
from .context import Context, Document, Sentence, TemporarySpan, Span
from .context import construct_stable_id, split_stable_id
from .candidate import Candidate, candidate_subclass, Marginal
from .annotation import (Feature, FeatureKey, Label, LabelKey, GoldLabel,
                         GoldLabelKey, StableLabel, Prediction, PredictionKey)

__all__ = [
    'Meta', 'Context', 'Document', 'Sentence', 'TemporarySpan', 'Span',
    'construct_stable_id', 'split_stable_id', 'Candidate',
    'candidate_subclass', 'Marginal', 'Feature', 'FeatureKey', 'Label',
    'LabelKey', 'GoldLabel', 'GoldLabelKey', 'StableLabel', 'Prediction',
    'PredictionKey'
]
