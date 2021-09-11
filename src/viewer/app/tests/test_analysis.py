import pytest
import pandas as pd
from collections import OrderedDict
from app.analysis import (
    as_css_class,
    calculate_trends,
)


def test_as_css_class():
    assert as_css_class(10, -10) == "recent-upward-trend"
    assert as_css_class(-10, 10) == "recent-downward-trend"
    assert as_css_class(0, 0) == "none"


def test_calculate_trends():
    df = pd.DataFrame.from_records(
        [
            {
                "asx_code": "ANZ",
                "2021-01-02": 1.0,
                "2021-01-03": 2.0,
                "2021-01-04": 3.0,
                "2021-01-05": 4.0,
            },
            {
                "asx_code": "BHP",
                "2021-01-02": 1.0,
                "2021-01-03": 1.0,
                "2021-01-04": 1.0,
                "2021-01-05": 1.0,
            },
        ],
        index="asx_code",
    )
    result = calculate_trends(df)
    assert result is not None
    assert isinstance(result, OrderedDict)
    assert len(result) == 1
    for stock, val in result.items():
        assert stock == "ANZ"
        assert isinstance(val, tuple)
        assert val[0] - 1.0 < 1e-6
        assert val[1] < 1e-6
        assert val[2] == 0.0  # since 30 days data is not available, these must be zero
        assert val[3] == 0.0
        assert val[4] == "none"
