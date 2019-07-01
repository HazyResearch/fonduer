import sys

import pytest

from fonduer.utils.visualizer import get_pdf_dim


# TODO test on MacOSX too
@pytest.mark.skipif(
    sys.platform == "darwin", reason="Only run visualizer test on Linux"
)
def test_get_pdf_dim(caplog):
    """Test get_pdf_dim on different pages"""
    assert get_pdf_dim("tests/data/pdf/BC546A_Series_B14-521026.pdf") == (729, 1032)
    assert get_pdf_dim("tests/data/pdf/BC546A_Series_B14-521026.pdf", page=1) == (
        729,
        1032,
    )
    assert get_pdf_dim("tests/data/pdf/BC546A_Series_B14-521026.pdf", page=6) == (
        612,
        792,
    )
