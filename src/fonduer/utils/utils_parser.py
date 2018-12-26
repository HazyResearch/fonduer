def build_node(type, name, content):
    """
    Wrap up content in to a html node.

    :param type: content type (e.g., doc, section, text, figure)
    :type path: str
    :param name: content name (e.g., the name of the section)
    :type path: str
    :param name: actual content
    :type path: str
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


def column_constructor(text, name=None, type="text", delim=None):
    """
    Converts raw content to a list of strutured tuple where each tuple contains
        (type, name, content).

    :param text: content to be converted ()
    :type path: str
    :param type: content name (default: None)
    :type path: str
    :param type: content type (default: text)
    :type path: str
    :param delim: delimiter to split the content
    :type path: str
    :return: A list of tuple where each tuple contains
        (content type, content name, content)
    """
    if delim is None:
        return [(type, name, text)]
    return [(type, name, content) for content in text.split(delim)]
