import lxml.etree as et

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class XMLMultiDocPreprocessor(DocPreprocessor):
    """
    Parse an XML file *which contains multiple documents* into a set of
    Document objects.

    Use XPath queries to specify a *document* object, and then for each
    document, a set of *text* sections and an *id*.

    **Note: Include the full document XML etree in the attribs dict with
    keep_xml_tree=True**
    """

    def __init__(
        self,
        path,
        doc=".//document",
        text="./text/text()",
        id="./id/text()",
        keep_xml_tree=False,
        *args,
        **kwargs
    ):
        super(XMLMultiDocPreprocessor, self).__init__(path, *args, **kwargs)
        self.doc = doc
        self.text = text
        self.id = id
        self.keep_xml_tree = keep_xml_tree

    def parse_file(self, f, file_name):
        for i, doc in enumerate(et.parse(f).xpath(self.doc)):
            doc_id = str(doc.xpath(self.id)[0])
            text = "\n".join([t for t in doc.xpath(self.text) if t is not None])
            meta = {"file_name": str(file_name)}
            if self.keep_xml_tree:
                meta["root"] = et.tostring(doc)
            stable_id = self.get_stable_id(doc_id)
            yield Document(name=doc_id, stable_id=stable_id, meta=meta), text

    def _can_read(self, fpath):
        return fpath.endswith(".xml")
