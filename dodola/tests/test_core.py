import numpy as np
import pytest
import xarray as xr
import cftime
from dodola.core import (
    train_quantiledeltamapping,
    adjust_quantiledeltamapping,
    adjust_quantiledeltamapping_year,
    train_analogdownscaling,
    adjust_analogdownscaling,
    _add_cyclic,
    xclim_units_any2pint,
    xclim_units_pint2cf,
)


def _timeseriesfactory(x, start_dt="1995-01-01", variable_name="fakevariable", units="K"):
    """Populate xr.Dataset with synthetic data for testing, only has time coords"""
    start_time = str(start_dt)
    if x.ndim != 1:
        raise ValueError("'x' needs dim of one")

    time = xr.cftime_range(
        start=start_time, freq="D", periods=len(x), calendar="noleap"
    )

    out = xr.Dataset({variable_name: (["time"], x)}, coords={"time": time})
    # need to set variable units to pass xclim 0.29 check on units
    out[variable_name].attrs["units"] = units
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


def test_add_cyclic():
    """Test _add_cyclic adds wraparound values"""
    in_da = xr.DataArray(
        np.ones([5, 6]) * np.arange(6),
        coords=[np.arange(5) + 1, np.arange(6) + 1],
        dims=["lat", "lon"],
    )
    out_ds = _add_cyclic(ds=in_da.to_dataset(name="fakevariable"), dim="lon")
    assert all(
        out_ds["fakevariable"].isel(lon=0) == out_ds["fakevariable"].isel(lon=-1)
    )


def test_qplad_integration_af_quantiles():
    """
    Test QPLAD correctly matches adjustmentfactor and quantiles for lat, dayofyear and for a specific quantile

    The strategy is to bias-correct a Dataset of ones, and then try to
    downscale it to two gridpoints with QPLAD. In one case we take the
    adjustment factors for a single dayofyear and manually change it to
    0.0. Then check for the corresponding change in the output dataset. In
    the other case we take the adjustment factors for one of the two
    latitudes we're downscaling to and manually change it to 0.0. We then
    check for the corresponding change in the output dataset for that latitude.
    To check for a specific quantile, we choose a particular day of year with
    associated quantile from the bias corrected data, manually change the
    adjustment factor for that quantile and day of year, and check that
    the changed adjustment factor has been applied to the bias corrected day value.
    """
    kind = "*"
    lat = [1.0, 1.5]
    time = xr.cftime_range(start="1994-12-17", end="2015-01-15", calendar="noleap")
    variable = "scen"

    data_ref = xr.DataArray(
        np.ones((len(time), len(lat)), dtype="float64"),
        coords={"time": time, "lat": lat},
        attrs={"units": "K"},
        dims=["time", "lat"],
        name=variable,
    ).chunk({"time": -1, "lat": -1})
    data_train = data_ref + 2
    data_train.attrs["units"] = "K"

    ref_fine = data_ref.to_dataset()
    ds_train = data_train.to_dataset()

    # take the mean across space to represent coarse reference data for AFs
    ds_ref_coarse = ref_fine.mean(["lat"], keep_attrs=True)
    ds_train = ds_train.mean(["lat"], keep_attrs=True)

    # tile the fine resolution grid with the coarse resolution ref data
    ref_coarse = ds_ref_coarse.broadcast_like(ref_fine)
    ds_bc = ds_train
    ds_bc[variable].attrs["units"] = "K"

    # this is an integration test between QDM and QPLAD, so use QDM services
    # for bias correction
    target_year = 2005
    qdm_model = train_quantiledeltamapping(
        reference=ds_ref_coarse, historical=ds_train, variable=variable, kind=kind
    )
    biascorrected_coarse = adjust_quantiledeltamapping(
        simulation=ds_bc,
        variable=variable,
        qdm=qdm_model.ds,
        years=[target_year],
        include_quantiles=True,
    )

    # make bias corrected data on the fine resolution grid
    biascorrected_fine = biascorrected_coarse.broadcast_like(
        ref_fine.sel(
            time=slice("{}-01-01".format(target_year), "{}-12-31".format(target_year))
        )
    )

    qplad_model = train_analogdownscaling(
        coarse_reference=ref_coarse,
        fine_reference=ref_fine,
        variable=variable,
        kind=kind,
    )

    # TODO: These prob should be two separate tests with setup fixtures...
    spoiled_time = qplad_model.ds.copy(deep=True)
    spoiled_latitude = qplad_model.ds.copy(deep=True)
    spoiled_quantile = qplad_model.ds.copy(deep=True)

    # Spoil one dayoftheyear value in adjustment factors (force it to be 0.0)
    # and test that the spoiled value correctly propigates through to output.
    time_idx_to_spoil = 25
    spoiled_time["af"][:, time_idx_to_spoil, :] = 0.0
    qplad_model.ds = spoiled_time
    downscaled = adjust_analogdownscaling(
        simulation=biascorrected_fine.set_coords(
            ["sim_q"]
        ),  # func assumes sim_q is coordinate...
        qplad=qplad_model,
        variable=variable,
    )

    # All but two values should be 1.0...
    assert (downscaled[variable].values == 1.0).sum() == 728
    # We should have 2 `0.0` entires. One in each lat...
    assert (downscaled[variable].values == 0.0).sum() == 2
    # All our 0.0s should be in this dayofyear/time slice in output dataset.
    np.testing.assert_array_equal(
        downscaled[variable].values[time_idx_to_spoil, :], np.array([0.0, 0.0])
    )

    # Similar to above, spoil one lat value in adjustment factors
    # (force it to be 0.0) and test that the spoiled value correctly
    # propagates through to output.
    latitude_idx_to_spoil = 0
    spoiled_latitude["af"][latitude_idx_to_spoil, ...] = 0.0
    qplad_model.ds = spoiled_latitude
    downscaled = adjust_analogdownscaling(
        simulation=biascorrected_fine.set_coords(
            ["sim_q"]
        ),  # func assumes sim_q is coordinate...
        qplad=qplad_model,
        variable=variable,
    )
    # Half of values in output should be 1.0...
    assert (downscaled[variable].values == 1.0).sum() == 365
    # The other half should be `0.0` due to the spoiled data...
    assert (downscaled[variable].values == 0.0).sum() == 365
    # All our 0.0s should be in this single lat in output dataset.
    assert all(downscaled[variable].values[:, latitude_idx_to_spoil] == 0.0)

    # spoil one quantile in adjustment factors for one day of year
    # force it to be 200 and ensure that a bias corrected day with that
    # quantile gets the spoiled value after downscaling
    # pick a day of year
    doy = 100
    # only do this for one lat pt
    lat_pt = 0
    # get the quantile from the bias corrected data for this doy and latitude
    q_100 = biascorrected_fine.sim_q[doy, lat_pt].values
    # extract quantiles from afs to get the corresponding quantile index
    bc_quantiles = qplad_model.ds.af[0, 100, :].quantiles.values
    # get index of the af for that day
    q_idx = np.argmin(np.abs(q_100 - bc_quantiles))

    # now spoil that doy quantile adjustment factor
    spoiled_quantile["af"][0, 100, q_idx] = 200
    qplad_model.ds = spoiled_quantile

    downscaled = adjust_analogdownscaling(
        simulation=biascorrected_fine.set_coords(
            ["sim_q"]
        ),  # func assumes sim_q is coordinate...
        qplad=qplad_model,
        variable=variable,
    )

    # the 100th doy and corresponding quantile should be equal to the spoiled value
    assert np.max(downscaled[variable].values[:, lat_pt]) == 200
    assert np.argmax(downscaled[variable].values[:, lat_pt]) == 100
    # check that the adjustment factor did not get applied to any other days of the year
    assert (downscaled[variable].values[:, lat_pt]).sum() == 564

def test_xclim_units_conversion():

    initial_unit = "mm d-1"
    cf_style = _timeseriesfactory(
        np.ones(1), start_dt="2015-01-01", variable_name="fake_variable", units=initial_unit
    )
    xclim_pint_style = xclim_units_any2pint(cf_style, "fake_variable")
    assert xclim_pint_style["fake_variable"].attrs["units"] == "millimeter / day"
    back_to_cf_style = xclim_units_pint2cf(xclim_pint_style, "fake_variable")
    assert back_to_cf_style["fake_variable"].attrs["units"] == initial_unit