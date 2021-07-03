import numpy as np
from numpy.testing import assert_approx_equal
import pytest
import xarray as xr
import cftime
from xclim.core.calendar import convert_calendar
from xclim.sdba.adjustment import QuantileDeltaMapping
from xclim.sdba.utils import equally_spaced_nodes
from xclim import sdba, set_options
from dodola.core import (
    train_quantiledeltamapping,
    adjust_quantiledeltamapping_year,
    train_analogdownscaling,
    adjust_analogdownscaling_year,
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
    "variable_kind",
    [
        pytest.param("+", id="additive kind"),
        pytest.param("*", id="multiplicative kind"),
    ],
)
def test_train_quantiledeltamapping_quantiles_excludeendpoints(variable_kind):
    """Test that "endpoints" are not included in trained QDM models"""
    n_quantiles = 100
    n_days = 365
    n_years = 20

    # Make up some data to training on...
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


def test_adjust_quantiledeltamapping_include_quantiles():
    """Test that include-quantiles flag results in bias corrected quantiles
    included in output"""
    target_variable = "fakevariable"
    n_simdays = 5 * 365  # 100 years of daily simulation.

    model_bias = 2.0
    ts_sim = np.ones(n_simdays, dtype=np.float64)
    sim = _timeseriesfactory(
        ts_sim * model_bias, start_dt="2015-01-01", variable_name=target_variable
    )

    target_year = 2017

    # Yes, I'm intentionally training the QDM to a different bias. This is to
    # spurn a difference between "kind" adjustments...
    qdm = _train_simple_qdm(
        target_variable="fakevariable", kind="+", additive_bias=model_bias + 1
    )
    adjusted_ds = adjust_quantiledeltamapping_year(
        simulation=sim,
        qdm=qdm,
        year=target_year,
        variable=target_variable,
        include_quantiles=True,
    )
    # check that quantiles are contained in output
    assert "sim_q" in adjusted_ds[target_variable].coords


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


def test_analoginspired_quantilepreserving_downscaling():
    """Tests that analog-inspired quantile-preserving downscaling
    method produces downscaled values for a fine-res grid such
    that the average of the downscaled values equals the bias
    corrected value for that timestep"""
    # load test data - using xarray's air temperature tutorial dataset
    # resample to daily
    ds = xr.tutorial.load_dataset("air_temperature").resample(time="D").mean()
    # remove leap days and only use four gridcells
    temp_slice = convert_calendar(ds["air"][:, :2, :2], target="noleap")

    # take the mean across space to represent coarse reference data for AFs
    temp_slice_mean = temp_slice.mean(["lat", "lon"])
    # then tile it to be on the same grid as the fine reference data
    temp_slice_mean_resampled = temp_slice_mean.broadcast_like(temp_slice)

    # need to create some fake bias corrected data so that we can use it to downscale
    with set_options(sdba_extra_output=True):
        quantiles = equally_spaced_nodes(620, eps=None)
        QDM = QuantileDeltaMapping(
            kind="+",
            nquantiles=quantiles,
            group=sdba.Grouper("time.dayofyear", window=31),
        )
        QDM.train(temp_slice_mean + 2, temp_slice_mean)
        fake_biascorrected = QDM.adjust(temp_slice_mean + 4)
        # this is necessary to make sim_q a coordinate on 'scen'
        fake_biascorrected = (
            fake_biascorrected["scen"]
            .assign_coords(sim_q=fake_biascorrected.sim_q)
            .to_dataset()
        )

    # now downscale
    aiqpd = train_analogdownscaling(
        temp_slice_mean_resampled.to_dataset(name="scen"),
        temp_slice.to_dataset(name="scen"),
        variable="scen",
        kind="+",
        quantiles_n=62,
    )

    # make bias corrected data on the fine resolution grid
    biascorrected = fake_biascorrected["scen"].broadcast_like(temp_slice)
    # downscale the bias corrected data
    aiqpd_downscaled = aiqpd.adjust(biascorrected)

    # check that bias corrected value at a given timestep equals the average
    # of the downscaled values that correspond to the bias corrected value
    bias_corrected_value = biascorrected.isel(time=100).values[0][0]
    downscaled_average = aiqpd_downscaled.isel(time=100).mean().values

    assert_approx_equal(
        bias_corrected_value, downscaled_average, significant=5, verbose=True
    )
