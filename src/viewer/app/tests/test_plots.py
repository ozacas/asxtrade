import pytest
from app.plots import price_change_bins

def test_price_change_bins():
    bins, labels = price_change_bins()
    assert all([bins is not None, labels is not None])
    assert len(bins) == len(labels) + 1
    assert labels == ['-100.0', '-10.0', '-5.0', '-3.0', '-2.0', '-1.0', '-1e-06',
                      '0.0', '1e-06', '1.0', '2.0', '3.0', '5.0', '10.0', '25.0', '100.0', '1000.0']
