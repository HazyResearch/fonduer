"""Fonduer parser utils."""
from typing import List, Optional, Tuple


def build_node(type: str, name: str, content: str) -> str:
    """
    Wrap up content in to a html node.

    :param type: content type (e.g., doc, section, text, figure)
    :param name: content name (e.g., the name of the section)
    :param name: actual content
    :return: new String with content in html format
    """
    if type == "doc":
        return f"<html>{content}</html>"
    if type == "section":
        return f"<section name='{name}'>{content}</section>"
    if type == "text":
        return f"<p name='{name}'>{content}</p>"
    if type == "figure":
        return f"<img name='{name}' src='{content}'/>"
    raise RuntimeError("unknown type")


def column_constructor(
    text: str,
    name: Optional[str] = None,
    type: str = "text",
    delim: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Column constructor.

    Convert raw content to a list of strutured tuple where each tuple contains
        (type, name, content).

    :param text: content to be converted ()
    :param type: content name (default: None)
    :param type: content type (default: text)
    :param delim: delimiter to split the content
    :return: A list of tuple where each tuple contains
        (content type, content name, content)
    """
    if delim is None:
        return [(type, name, text)]
    return [(type, name, content) for content in text.split(delim)]
