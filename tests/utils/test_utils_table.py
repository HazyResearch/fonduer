import logging

from fonduer.utils.utils_table import _min_range_diff


def test_min_range_diff(caplog):
    """Test the minimum range calculation for table utils."""
    caplog.set_level(logging.INFO)

    assert _min_range_diff(0, 5, 0, 5) == 0
    assert _min_range_diff(1, 5, 3, 6) == 0
    assert _min_range_diff(1, 2, 2, 3) == 0
    assert _min_range_diff(3, 6, 1, 4) == 0
    assert _min_range_diff(1, 2, 3, 4) == 1
    assert _min_range_diff(3, 4, 1, 2) == 1
    assert _min_range_diff(3, 4, 1, 2, absolute=False) == 1
    assert _min_range_diff(1, 2, 3, 4, absolute=False) == -1
