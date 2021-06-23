import numpy as np
import pytest
import xarray as xr
import cftime
from dodola.core import (
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


def test_train_quantiledeltamapping_quantiles_excludeendpoints():
    """Test that "endpoints" are not included in trained QDM models"""
    n_quantiles = 100
    n_days = 365
    n_years = 20

    # Make up some data to training on...
    variable_kind = "+"
    target_variable = "fakevariable"
    ts_ref = np.arange(
        n_years * n_days, dtype=np.float64  # 20 years of daily historical.
    )
    hist = _timeseriesfactory(ts_ref, variable_name=target_variable)
    ref = _timeseriesfactory(ts_ref[::-1], variable_name=target_variable)

    qdm = train_quantiledeltamapping(
        historical=hist,
        reference=ref,
        variable=target_variable,
        kind=variable_kind,
        quantiles_n=n_quantiles,
    )

    # qdm.ds.hist_q.shape[1] should NOT be n_quantiles+2.
    assert qdm.ds.hist_q.shape[1] == n_quantiles
    # Check that 0, 1 are literally excluded:
    assert 1.0 not in qdm.ds.hist_q
    assert 0.0 not in qdm.ds.hist_q


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

    target_year = 2026

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

    target_year = 2088

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
