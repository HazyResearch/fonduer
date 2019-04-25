import re
from builtins import range

from fonduer.candidates.models.entity import Entity


def camel_to_under(name):
    """
    Converts camel-case string to lowercase string separated by underscores.

    Written by epost (http://stackoverflow.com/questions/1175208).

    :param name: String to be converted
    :return: new String with camel-case converted to lowercase, underscored
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_as_dict(x):
    """Return an object as a dictionary of its attributes."""
    if isinstance(x, dict):
        return x
    else:
        try:
            return x._asdict()
        except AttributeError:
            return x.__dict__


def tokens_to_ngrams(tokens, n_min=1, n_max=3, delim=" ", lower=False):
    f = (lambda x: x.lower()) if lower else (lambda x: x)
    N = len(tokens)
    for root in range(N):
        for n in range(max(n_min - 1, 0), min(n_max, N - root)):
            yield f(delim.join(tokens[root : root + n + 1]))


def get_entity(session, entity_id):
    """Return an entity. If not exists yet, create one and return it.
    :param session:
    :param entity_id: id for the entity to be returned
    :return: the entity
    """
    entity = session.query(Entity).filter(Entity.id == entity_id).one_or_none()
    if not entity:
        entity = Entity(id=entity_id)
        session.add(entity)
        session.commit()
    return entity
