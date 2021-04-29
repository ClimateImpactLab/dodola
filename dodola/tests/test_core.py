import numpy as np
import pytest
import xarray as xr
from dodola.core import qdm_rollingyearwindow


@pytest.mark.parametrize(
    "in_dts, goalyears",
    [
        pytest.param(
            ("2015-01-01", "2100-01-01"), (2026, 2088), id="early start, early end"
        ),
        pytest.param(
            ("2015-12-25", "2100-01-01"), (2027, 2088), id="late start, early end"
        ),
        pytest.param(
            ("2015-01-01", "2100-02-01"), (2026, 2089), id="early start, late end"
        ),
    ],
)
def test_qdm_rollingyearwindow(in_dts, goalyears):
    """Test qdm_rollingyearwindow ensures account for have 15 day buffer in edge years"""
    # Create test data
    t = xr.cftime_range(start=in_dts[0], end=in_dts[1], freq="D", calendar="noleap")
    x = np.ones(len(t))
    in_ds = xr.Dataset({"fakevariable": (["time"], x)}, coords={"time": t})

    actual_first, actual_last = qdm_rollingyearwindow(in_ds)

    assert actual_first == goalyears[0]
    assert actual_last == goalyears[1]
