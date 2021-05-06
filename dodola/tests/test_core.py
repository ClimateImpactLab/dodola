import numpy as np
import pytest
import xarray as xr
import cftime
from dodola.core import (
    qdm_rollingyearwindow,
    train_quantiledeltamapping,
    adjust_quantiledeltamapping_year,
)


def _timeseriesfactory(x, start_dt="1995-01-01", variable_name="fakevariable"):
    """Populate xr.Dataset with synthetic data for testing, only has time coords"""
    start_time = str(start_dt)
    if x.ndim != 1:
        raise ValueError("'x' needs dim of one")

    time = xr.cftime_range(
        start=start_time, freq="D", periods=len(x), calendar="noleap"
    )

    out = xr.Dataset({variable_name: (["time"], x)}, coords={"time": time})
    return out


def _train_simple_qdm(
    kind, target_variable="fakevariable", additive_bias=0.0, multiplicative_bias=1.0
):
    """Train a QDM on a single, simple time series with select biases"""
    variable_kind = str(kind)
    n_histdays = 20 * 365  # 20 years of daily historical.

    ts_ref = np.ones(n_histdays, dtype=np.float64)
    hist = _timeseriesfactory(
        (ts_ref + additive_bias) * multiplicative_bias, variable_name=target_variable
    )
    ref = _timeseriesfactory(ts_ref, variable_name=target_variable)

    qdm = train_quantiledeltamapping(
        historical=hist,
        reference=ref,
        variable=target_variable,
        kind=variable_kind,
    )
    return qdm


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
    """Test qdm_rollingyearwindow accounts for ± 15 day buffer at time edges"""
    # Create test data
    t = xr.cftime_range(start=in_dts[0], end=in_dts[1], freq="D", calendar="noleap")
    x = np.ones(len(t))
    in_ds = xr.Dataset({"fakevariable": (["time"], x)}, coords={"time": t})

    actual_first, actual_last = qdm_rollingyearwindow(in_ds)

    assert actual_first == goalyears[0]
    assert actual_last == goalyears[1]


@pytest.mark.parametrize(
    "variable_kind, expected",
    [
        pytest.param("+", -1.0, id="additive kind"),
        pytest.param("*", 0.5, id="multiplicative kind"),
    ],
)
def test_adjust_quantiledeltamapping_year_kind(variable_kind, expected):
    """Test that QDM 'kind' is handled"""
    # Setup input data.
    target_variable = "fakevariable"
    n_simdays = 100 * 365  # 100 years of daily simulation.

    model_bias = 2.0
    ts_sim = np.ones(n_simdays, dtype=np.float64)
    sim = _timeseriesfactory(
        ts_sim * model_bias, start_dt="2015-01-01", variable_name=target_variable
    )

    # Target the earliest year we can:
    target_year, _ = qdm_rollingyearwindow(sim)

    # Yes, I'm intentionally training the QDM to a different bias. This is to
    # spurn a difference between "kind" adjustments...
    qdm = _train_simple_qdm(
        target_variable="fakevariable", kind=variable_kind, additive_bias=model_bias + 1
    )
    adjusted_ds = adjust_quantiledeltamapping_year(
        simulation=sim,
        qdm=qdm,
        year=target_year,
        variable=target_variable,
    )
    assert all(adjusted_ds[target_variable] == expected)


def test_adjust_quantiledeltamapping_year_output_time():
    """Check 'time' year and edges of QDM adjusted output

    This tests integration between `dodola.core.adjust_quantiledeltamapping_year`,
    `dodola.core.qdm_rollingyearwindow`, and
    `dodola.core.train_quantiledeltamapping`.
    """
    # Setup input data.
    target_variable = "fakevariable"
    variable_kind = "+"
    n_simdays = 85 * 365

    model_bias = 2.0
    ts_sim = np.ones(n_simdays, dtype=np.float64)
    sim = _timeseriesfactory(
        ts_sim + model_bias, start_dt="2015-01-01", variable_name=target_variable
    )

    # Target the earliest year we can:
    _, target_year = qdm_rollingyearwindow(sim)

    qdm = _train_simple_qdm(
        target_variable="fakevariable", kind=variable_kind, additive_bias=model_bias
    )
    adjusted_ds = adjust_quantiledeltamapping_year(
        simulation=sim,
        qdm=qdm,
        year=target_year,
        variable=target_variable,
    )
    assert np.unique(adjusted_ds["time"].dt.year).item() == 2088
    assert min(adjusted_ds["time"].values) == cftime.DatetimeNoLeap(
        2088, 1, 1, 0, 0, 0, 0
    )
    assert max(adjusted_ds["time"].values) == cftime.DatetimeNoLeap(
        2088, 12, 31, 0, 0, 0, 0
    )
