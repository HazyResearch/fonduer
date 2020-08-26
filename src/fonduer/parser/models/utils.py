"""Utilities for constructing and splitting stable ids."""
from typing import List, Tuple

from fonduer.parser.models import Context


def construct_stable_id(
    parent_context: Context,
    polymorphic_type: str,
    relative_char_offset_start: int,
    relative_char_offset_end: int,
) -> str:
    """Construct Context's stable ID.

    Construct a stable ID for a Context given its parent and its character
    offsets relative to the parent.
    """
    doc_id, type, idx = split_stable_id(parent_context.stable_id)

    if polymorphic_type in [
        "section_mention",
        "figure_mention",
        "table_mention",
        "paragraph_mention",
        "caption_mention",
    ]:
        parent_doc_start = idx[0]
        return f"{doc_id}::{polymorphic_type}:{parent_doc_start}"
    elif polymorphic_type in ["cell_mention"]:
        cell_pos = idx[0]
        cell_row_start = idx[1]
        cell_col_start = idx[2]
        return (
            f"{doc_id}::{polymorphic_type}:{cell_pos}:{cell_row_start}:{cell_col_start}"
        )
    elif polymorphic_type in ["sentence", "document_mention", "span_mention"]:
        parent_doc_char_start = idx[0]
        start = parent_doc_char_start + relative_char_offset_start
        end = parent_doc_char_start + relative_char_offset_end
        return f"{doc_id}::{polymorphic_type}:{start}:{end}"

    raise ValueError(f"Unrecognized context type:\t{polymorphic_type}")


def split_stable_id(
    stable_id: str,
) -> Tuple[str, str, List[int]]:
    """Split stable ID.

    Analyzing stable ID and return the following information:

        * Document (root) stable ID
        * Context polymorphic type
        * Character offset start, end *relative to document start*

    Returns tuple of four values.
    """
    split1 = stable_id.split("::")
    if len(split1) == 2:
        split2 = split1[1].split(":")
        type = split2[0]
        idx = [int(_) for _ in split2[1:]]
        return split1[0], type, idx

    raise ValueError(f"Malformed stable_id:\t{stable_id}")
