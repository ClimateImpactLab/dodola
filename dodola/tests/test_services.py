"""Test application services
"""

import numpy as np
import pytest
import xarray as xr
from xesmf.data import wave_smooth
from xesmf.util import grid_global
from xclim.sdba.adjustment import QuantileDeltaMapping
from dodola.services import (
    prime_qplad_output_zarrstore,
    prime_qdm_output_zarrstore,
    rechunk,
    regrid,
    remove_leapdays,
    clean_cmip6,
    correct_wet_day_frequency,
    adjust_maximum_precipitation,
    train_qdm,
    apply_qdm,
    train_qplad,
    apply_qplad,
    validate,
    get_attrs,
    correct_small_dtr,
)
import dodola.repository as repository


def _datafactory(x, start_time="1950-01-01", variable_name="fakevariable"):
    """Populate xr.Dataset with synthetic data for testing"""
    start_time = str(start_time)
    if x.ndim != 1:
        raise ValueError("'x' needs dim of one")

    time = xr.cftime_range(
        start=start_time, freq="D", periods=len(x), calendar="standard"
    )

    out = xr.Dataset(
        {variable_name: (["time", "lon", "lat"], x[:, np.newaxis, np.newaxis])},
        coords={
            "index": time,
            "time": time,
            "lon": (["lon"], [1.0]),
            "lat": (["lat"], [1.0]),
        },
    )
    # need to set variable units to pass xclim 0.29 check on units
    out[variable_name].attrs["units"] = "K"
    return out


def _modeloutputfactory(
    start_time="1950-01-01", end_time="2014-12-31", variable_name="fakevariable"
):
    """Populate xr.Dataset with synthetic output data for testing"""
    start_time = str(start_time)
    end_time = str(end_time)

    np.random.seed(0)
    time = xr.cftime_range(start=start_time, end=end_time, calendar="noleap")
    # make sure that test data range is reasonable for the variable being tested

    low_val = None
    high_val = None
    if variable_name == "tasmax" or variable_name == "tasmin":
        low_val = 160
        high_val = 340
    elif variable_name == "dtr":
        low_val = 1
        high_val = 40
    elif variable_name == "pr":
        low_val = 0.01
        high_val = 1900
    data = np.random.randint(low_val, high_val, len(time)).astype(np.float64)

    out = xr.Dataset(
        {variable_name: (["time", "lon", "lat"], data[:, np.newaxis, np.newaxis])},
        coords={
            "index": time,
            "time": time,
            "lon": (["lon"], [1.0]),
            "lat": (["lat"], [1.0]),
        },
    )
    # need to set variable units to pass xclim 0.29 check on units
    out[variable_name].attrs["units"] = "K"
    return out


def _gcmfactory(x, gcm_variable="fakevariable", start_time="1950-01-01"):
    """Populate xr.Dataset with synthetic GCM data for testing
    that includes extra dimensions and leap days to be removed.
    """
    start_time = str(start_time)
    if x.ndim != 1:
        raise ValueError("'x' needs dim of one")

    time = xr.cftime_range(
        start=start_time, freq="D", periods=len(x), calendar="standard"
    )

    out = xr.Dataset(
        {
            gcm_variable: (
                ["time", "lon", "lat", "member_id"],
                x[:, np.newaxis, np.newaxis, np.newaxis],
            )
        },
        coords={
            "index": time,
            "time": time,
            "bnds": [0, 1],
            "lon": (["lon"], [1.0]),
            "lat": (["lat"], [1.0]),
            "member_id": (["member_id"], [1.0]),
            "height": (["height"], [1.0]),
            "time_bnds": (["time", "bnds"], np.ones((len(x), 2))),
        },
    )

    return out


@pytest.fixture
def domain_file(request):
    """Creates a fake domain Dataset for testing"""
    lon_name = "lon"
    lat_name = "lat"
    domain = grid_global(request.param, request.param)
    domain[lat_name] = np.unique(domain[lat_name].values)
    domain[lon_name] = np.unique(domain[lon_name].values)

    return domain


def test_prime_qplad_output_zarrstore():
    """
    Test that prime_qplad_output_zarrstore creates a Zarr with good variables, shapes, attrs
    """
    # Make fake simulation data for test.
    target_variable = "fakevariable"
    sim = _datafactory(np.ones(365, dtype=np.float32), variable_name=target_variable)
    sim.attrs["foo"] = "bar"
    goal_shape = sim[target_variable].shape
    sim_key = "memory://test_prime_qplad_output_zarrstore/sim.zarr"
    repository.write(sim_key, sim)

    primed_url = "memory://test_prime_qplad_output_zarrstore/primed.zarr"

    prime_qplad_output_zarrstore(
        simulation=sim_key,
        variable=target_variable,
        out=primed_url,
        zarr_region_dims=["lat"],
    )

    primed_ds = repository.read(primed_url)

    assert target_variable in primed_ds.variables
    assert primed_ds[target_variable].shape == goal_shape
    assert primed_ds.attrs["foo"] == "bar"


def test_prime_qdm_output_zarrstore():
    """
    Test that prime_qdm_output_zarrstore creates a Zarr with variables, shapes, attrs.

    We're testing this by running QDM (train + apply) in it's usualy mode and
    then using comparable parameters to prime a Zarr Store. We then compare
    the two.
    """
    # Setup input data.
    quantile_variable = "sim_q"
    target_variable = "fakevariable"
    variable_kind = "additive"
    n_histdays = 10 * 365  # 10 years of daily historical.
    n_simdays = 50 * 365  # 50 years of daily simulation.

    model_bias = 2
    ts_ref = np.ones(n_histdays, dtype=np.float64)
    ts_sim = np.ones(n_simdays, dtype=np.float64)
    hist = _datafactory(ts_ref + model_bias, variable_name=target_variable)
    ref = _datafactory(ts_ref, variable_name=target_variable)
    sim = _datafactory(ts_sim + model_bias, variable_name=target_variable)

    # Load up a fake repo with our input data in the place of big data and cloud
    # storage.
    qdm_key = "memory://test_prime_qdm_output_zarrstore/qdm.zarr"
    hist_key = "memory://test_prime_qdm_output_zarrstore/hist.zarr"
    ref_key = "memory://test_prime_qdm_output_zarrstore/ref.zarr"
    sim_key = "memory://test_prime_qdm_output_zarrstore/sim.zarr"
    sim_adj_key = "memory://test_prime_qdm_output_zarrstore/sim_adjusted.zarr"
    primed_url = "memory://test_prime_qdm_output_zarrstore/primed.zarr"

    repository.write(sim_key, sim)
    repository.write(hist_key, hist)
    repository.write(ref_key, ref)

    target_year = 1995

    # Lets prime a QDM output.
    prime_qdm_output_zarrstore(
        simulation=sim_key,
        variable=target_variable,
        years=[target_year],
        out=primed_url,
        zarr_region_dims=["lat"],
    )

    primed_ds = repository.read(primed_url)

    # Now train, apply actual QDM and compare outputs with primed Zarr Store
    train_qdm(
        historical=hist_key,
        reference=ref_key,
        out=qdm_key,
        variable=target_variable,
        kind=variable_kind,
    )
    apply_qdm(
        simulation=sim_key,
        qdm=qdm_key,
        years=[target_year],
        variable=target_variable,
        out=sim_adj_key,
    )
    adjusted_ds = repository.read(sim_adj_key)

    # Desired variables present?
    assert (
        quantile_variable in primed_ds.variables
        and target_variable in primed_ds.variables
    )
    # Desired shapes with dims in correct order?
    assert primed_ds[quantile_variable].shape == primed_ds[target_variable].shape
    assert primed_ds[target_variable].shape == adjusted_ds[target_variable].shape
    # Output attrs matching for root and variables?
    assert primed_ds.attrs == adjusted_ds.attrs
    assert primed_ds[target_variable].attrs == adjusted_ds[target_variable].attrs
    assert primed_ds[quantile_variable].attrs == adjusted_ds[quantile_variable].attrs


def test_prime_qdm_regional_apply():
    """
    Integration test checking that prime_qdm_output_zarrstore and apply_qdm can write regionally.

    The strategy is to create input data for two latitudes and run two QDMs
    (train + apply). One doing a "vanilla", global QDM. The other using a
    "regional" strategy: training and applying QDM on each latitude and then
    writing each to the same, primed zarr store. We then compare output from
    the vanilla and regional approaches.
    """
    # Setup input data.
    target_variable = "fakevariable"
    variable_kind = "additive"
    n_histdays = 10 * 365  # 10 years of daily historical.
    n_simdays = 50 * 365  # 50 years of daily simulation.

    model_bias = 2
    ts_ref = np.ones(n_histdays, dtype=np.float64)
    ts_sim = np.ones(n_simdays, dtype=np.float64)
    hist = _datafactory(ts_ref + model_bias, variable_name=target_variable)
    ref = _datafactory(ts_ref, variable_name=target_variable)
    sim = _datafactory(ts_sim + model_bias, variable_name=target_variable)

    # Append a copy of the data onto a new latitude of "2.0". I'm too lazy to
    # modify the data factories to get this. Gives us a way to test regional
    # writes.
    sim = xr.concat([sim, sim.assign({"lat": np.array([2.0])})], dim="lat")
    sim[target_variable][
        :, :, -1
    ] += 1  # Introducing a slight difference for different lat.
    ref = xr.concat([ref, ref.assign({"lat": np.array([2.0])})], dim="lat")
    hist = xr.concat([hist, hist.assign({"lat": np.array([2.0])})], dim="lat")

    # Datafactory appends cruft "index" coordinate. We're removing it because we
    # dont need it and I'm too lazy to tinker with input data fixtures.
    hist = hist.drop_vars("index")
    ref = ref.drop_vars("index")
    sim = sim.drop_vars("index")

    # Load up a fake repo with our input data in the place of big data and cloud
    # storage.
    qdm_key = "memory://test_apply_qdm/qdm_global.zarr"
    qdm_region1_key = "memory://test_apply_qdm/qdm_region1.zarr"
    qdm_region2_key = "memory://test_apply_qdm/qdm_region2.zarr"
    hist_key = "memory://test_apply_qdm/hist.zarr"
    ref_key = "memory://test_apply_qdm/ref.zarr"
    sim_key = "memory://test_apply_qdm/sim.zarr"
    primed_url = "memory://test_prime_qdm_output_zarrstore/primed.zarr"
    sim_adj_key = "memory://test_apply_qdm/sim_adjusted.zarr"

    repository.write(sim_key, sim)
    repository.write(hist_key, hist)
    repository.write(ref_key, ref)

    target_years = [1994, 1995]

    # Lets prime a QDM output.
    prime_qdm_output_zarrstore(
        simulation=sim_key,
        variable=target_variable,
        years=target_years,
        out=primed_url,
        zarr_region_dims=["lat"],
    )

    # Now train, apply QDM for two cases. One with region write to primed
    # zarr and one without.

    # Writing to regions
    region_1 = {"lat": slice(0, 1)}
    train_qdm(
        historical=hist_key,
        reference=ref_key,
        out=qdm_region1_key,
        variable=target_variable,
        kind=variable_kind,
        isel_slice=region_1,
    )
    apply_qdm(
        simulation=sim_key,
        qdm=qdm_region1_key,
        years=target_years,
        variable=target_variable,
        out=primed_url,
        isel_slice=region_1,
        out_zarr_region=region_1,
    )
    region_2 = {"lat": slice(1, 2)}
    train_qdm(
        historical=hist_key,
        reference=ref_key,
        out=qdm_region2_key,
        variable=target_variable,
        kind=variable_kind,
        isel_slice=region_2,
    )
    apply_qdm(
        simulation=sim_key,
        qdm=qdm_region2_key,
        years=target_years,
        variable=target_variable,
        out=primed_url,
        isel_slice=region_2,
        out_zarr_region=region_2,
    )
    primed_adjusted_ds = repository.read(primed_url)

    # Doing it globally, all "regions" at once.
    train_qdm(
        historical=hist_key,
        reference=ref_key,
        out=qdm_key,
        variable=target_variable,
        kind=variable_kind,
    )
    apply_qdm(
        simulation=sim_key,
        qdm=qdm_key,
        years=target_years,
        variable=target_variable,
        out=sim_adj_key,
    )
    adjusted_ds = repository.read(sim_adj_key)

    # Desired variables present?
    xr.testing.assert_allclose(primed_adjusted_ds, adjusted_ds)


def test_apply_qdm():
    """Test to apply a trained QDM to input data and read the output.

    This is an integration test between train_qdm, apply_qdm.
    """
    # Setup input data.
    target_variable = "fakevariable"
    variable_kind = "additive"
    n_histdays = 10 * 365  # 10 years of daily historical.
    n_simdays = 50 * 365  # 50 years of daily simulation.

    model_bias = 2
    ts_ref = np.ones(n_histdays, dtype=np.float64)
    ts_sim = np.ones(n_simdays, dtype=np.float64)
    hist = _datafactory(ts_ref + model_bias, variable_name=target_variable)
    ref = _datafactory(ts_ref, variable_name=target_variable)
    sim = _datafactory(ts_sim + model_bias, variable_name=target_variable)

    # Load up a fake repo with our input data in the place of big data and cloud
    # storage.
    qdm_key = "memory://test_apply_qdm/qdm.zarr"
    hist_key = "memory://test_apply_qdm/hist.zarr"
    ref_key = "memory://test_apply_qdm/ref.zarr"
    sim_key = "memory://test_apply_qdm/sim.zarr"
    sim_adj_key = "memory://test_apply_qdm/sim_adjusted.zarr"
    repository.write(sim_key, sim)
    repository.write(hist_key, hist)
    repository.write(ref_key, ref)

    target_years = [1994, 1995]

    train_qdm(
        historical=hist_key,
        reference=ref_key,
        out=qdm_key,
        variable=target_variable,
        kind=variable_kind,
    )
    apply_qdm(
        simulation=sim_key,
        qdm=qdm_key,
        years=target_years,
        variable=target_variable,
        out=sim_adj_key,
    )
    adjusted_ds = repository.read(sim_adj_key)
    assert target_variable in adjusted_ds.variables


@pytest.mark.parametrize("kind", ["multiplicative", "additive"])
def test_train_qdm(kind):
    """Test that train_qdm outputs store giving sdba.adjustment.QuantileDeltaMapping

    Checks that output is consistent if we do "additive" or "multiplicative"
    QDM kinds.
    """
    # Setup input data.
    n_years = 10
    n = n_years * 365

    model_bias = 2
    ts = np.sin(np.linspace(-10 * 3.14, 10 * 3.14, n)) * 0.5
    hist = _datafactory(ts + model_bias)
    ref = _datafactory(ts)

    output_key = "memory://test_train_qdm/test_output.zarr"
    hist_key = "memory://test_train_qdm/hist.zarr"
    ref_key = "memory://test_train_qdm/ref.zarr"

    # Load up a fake repo with our input data in the place of big data and cloud
    # storage.
    repository.write(hist_key, hist)
    repository.write(ref_key, ref)

    train_qdm(
        historical=hist_key,
        reference=ref_key,
        out=output_key,
        variable="fakevariable",
        kind=kind,
    )

    assert QuantileDeltaMapping.from_dataset(repository.read(output_key))


def test_train_qdm_isel_slice():
    """Test that train_qdm outputs subset data when passed isel_slice"""
    # Setup input data.
    n_years = 10
    n = n_years * 365

    # Lazy way to make fake data for 2 latitudes...
    model_bias = 2
    ts = np.sin(np.linspace(-10 * 3.14, 10 * 3.14, n)) * 0.5
    hist1 = _datafactory(ts + model_bias)
    hist2 = _datafactory(ts + model_bias).assign_coords(
        {"lat": hist1["lat"].data + 1.0}
    )
    hist = xr.concat([hist1, hist2], dim="lat")
    ref1 = _datafactory(ts)
    ref2 = _datafactory(ts + model_bias).assign_coords({"lat": ref1["lat"].data + 1.0})
    ref = xr.concat([ref1, ref2], dim="lat")

    output_key = "memory://test_train_qdm_isel_slice/test_output.zarr"
    hist_key = "memory://test_train_qdm_isel_slice/hist.zarr"
    ref_key = "memory://test_train_qdm_isel_slice/ref.zarr"

    repository.write(hist_key, hist)
    repository.write(ref_key, ref)

    train_qdm(
        historical=hist_key,
        reference=ref_key,
        out=output_key,
        variable="fakevariable",
        kind="additive",
        isel_slice={"lat": slice(0, 1)},  # select only 1 of 2 lats by idx...
    )

    # Check we can read output and it's the selected value, only.
    ds_result = repository.read(output_key)
    np.testing.assert_equal(ds_result["lat"].data, ref["lat"].data[0])
    assert QuantileDeltaMapping.from_dataset(ds_result)


def test_rechunk():
    """Test that rechunk service rechunks"""
    chunks_goal = {"time": 4, "lon": 1, "lat": 1}
    test_ds = xr.Dataset(
        {"fakevariable": (["time", "lon", "lat"], np.ones((4, 4, 4)))},
        coords={
            "time": [1, 2, 3, 4],
            "lon": (["lon"], [1.0, 2.0, 3.0, 4.0]),
            "lat": (["lat"], [1.5, 2.5, 3.5, 4.5]),
        },
    )

    in_url = "memory://test_rechunk/input_ds.zarr"
    out_url = "memory://test_rechunk/output_ds.zarr"
    repository.write(in_url, test_ds)

    rechunk(
        in_url,
        target_chunks=chunks_goal,
        out=out_url,
    )
    actual_chunks = repository.read(out_url)["fakevariable"].data.chunksize

    assert actual_chunks == tuple(chunks_goal.values())


@pytest.mark.parametrize(
    "domain_file, regrid_method, expected_shape",
    [
        pytest.param(
            1.0,
            "bilinear",
            (180, 360),
            id="Bilinear regrid",
        ),
        pytest.param(
            1.0,
            "conservative",
            (180, 360),
            id="Conservative regrid",
        ),
    ],
    indirect=["domain_file"],
)
def test_regrid_methods(domain_file, regrid_method, expected_shape):
    """Smoke test that services.regrid outputs with different regrid methods

    The expected shape is the same, but change in methods should not error.
    """
    # Make fake input data.
    ds_in = grid_global(30, 20)
    ds_in["fakevariable"] = wave_smooth(ds_in["lon"], ds_in["lat"])

    in_url = "memory://test_regrid_methods/an/input/path.zarr"
    domain_file_url = "memory://test_regrid_methods/a/domainfile/path.zarr"
    out_url = "memory://test_regrid_methods/an/output/path.zarr"
    repository.write(in_url, ds_in)
    repository.write(domain_file_url, domain_file)

    regrid(in_url, out=out_url, method=regrid_method, domain_file=domain_file_url)
    actual_shape = repository.read(out_url)["fakevariable"].shape
    assert actual_shape == expected_shape


@pytest.mark.parametrize(
    "domain_file, expected_dtype",
    [
        pytest.param(
            2.0,
            "float64",
            id="Cast output to float64",
        ),
        pytest.param(
            2.0,
            "float32",
            id="Cast output to float32",
        ),
    ],
    indirect=["domain_file"],
)
def test_regrid_dtype(domain_file, expected_dtype):
    """Tests that services.regrid casts output to different dtypes"""
    # Make fake input data.
    ds_in = grid_global(5, 10)
    ds_in["fakevariable"] = wave_smooth(ds_in["lon"], ds_in["lat"])

    in_url = "memory://test_regrid_dtype/an/input/path.zarr"
    domain_file_url = "memory://test_regrid_dtype/a/domainfile/path.zarr"
    out_url = "memory://test_regrid_dtype/an/output/path.zarr"
    repository.write(in_url, ds_in)
    repository.write(domain_file_url, domain_file)

    regrid(
        in_url,
        out=out_url,
        method="bilinear",
        domain_file=domain_file_url,
        astype=expected_dtype,
    )
    actual_dtype = repository.read(out_url)["fakevariable"].dtype
    assert actual_dtype == expected_dtype


@pytest.mark.parametrize(
    "domain_file, regrid_method",
    [pytest.param(5.0, "bilinear"), pytest.param(5.0, "conservative")],
    indirect=["domain_file"],
)
def test_regrid_attrs(domain_file, regrid_method):
    """Tests that services.regrid copies attrs metadata to output"""
    # Make fake input data.
    ds_in = grid_global(5, 10)
    ds_in["fakevariable"] = wave_smooth(ds_in["lon"], ds_in["lat"])

    ds_in.attrs["foo"] = "bar"
    ds_in["fakevariable"].attrs["bar"] = "foo"

    in_url = "memory://test_regrid_attrs/an/input/path.zarr"
    domain_file_url = "memory://test_regrid_attrs/a/domainfile/path.zarr"
    out_url = "memory://test_regrid_attrs/an/output/path.zarr"
    repository.write(in_url, ds_in)
    repository.write(domain_file_url, domain_file)

    regrid(
        in_url,
        out=out_url,
        method="bilinear",
        domain_file=domain_file_url,
        astype="float32",
    )
    assert repository.read(out_url).attrs["foo"] == "bar"
    assert repository.read(out_url)["fakevariable"].attrs["bar"] == "foo"


@pytest.mark.parametrize(
    "domain_file, expected_shape",
    [
        pytest.param(
            1.0,
            (180, 360),
            id="Regrid to domain file grid",
        ),
        pytest.param(
            2.0,
            (90, 180),
            id="Regrid to global 2.0° x 2.0° grid",
        ),
    ],
    indirect=["domain_file"],
)
def test_regrid_resolution(domain_file, expected_shape):
    """Smoke test that services.regrid outputs with different grid resolutions

    The expected shape is the same, but change in methods should not error.
    """
    # Make fake input data.
    ds_in = grid_global(30, 20)
    ds_in["fakevariable"] = wave_smooth(ds_in["lon"], ds_in["lat"])

    in_url = "memory://test_regrid_resolution/an/input/path.zarr"
    domain_file_url = "memory://test_regrid_resolution/a/domainfile/path.zarr"
    out_url = "memory://test_regrid_resolution/an/output/path.zarr"
    repository.write(in_url, ds_in)
    repository.write(domain_file_url, domain_file)

    regrid(in_url, out=out_url, method="bilinear", domain_file=domain_file_url)
    actual_shape = repository.read(out_url)["fakevariable"].shape
    assert actual_shape == expected_shape


def test_clean_cmip6():
    """Tests that cmip6 cleanup removes extra dimensions on dataset"""
    # Setup input data
    n = 1500  # need over four years of daily data
    ts = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 0.5
    ds_gcm = _gcmfactory(ts, start_time="1950-01-01")

    in_url = "memory://test_clean_cmip6/an/input/path.zarr"
    out_url = "memory://test_clean_cmip6/an/output/path.zarr"
    repository.write(in_url, ds_gcm)

    clean_cmip6(in_url, out_url, leapday_removal=True)
    ds_cleaned = repository.read(out_url)

    assert "height" not in ds_cleaned.coords
    assert "member_id" not in ds_cleaned.coords
    assert "time_bnds" not in ds_cleaned.coords


@pytest.mark.parametrize(
    "gcm_variable", [pytest.param("tasmax"), pytest.param("tasmin"), pytest.param("pr")]
)
def test_cmip6_precip_unitconversion(gcm_variable):
    """Tests that precip units are converted in CMIP6 cleanup if variable is precip"""
    # Setup input data
    n = 1500  # need over four years of daily data
    ts = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 0.5
    ds_gcm = _gcmfactory(ts, gcm_variable=gcm_variable, start_time="1950-01-01")

    if gcm_variable == "pr":
        # assign units to typical GCM pr units so they can be cleaned
        ds_gcm["pr"].attrs["units"] = "kg m-2 s-1"

    in_url = "memory://test_clean_cmip6/an/input/path.zarr"
    out_url = "memory://test_clean_cmip6/an/output/path.zarr"
    repository.write(in_url, ds_gcm)

    clean_cmip6(in_url, out_url, leapday_removal=True)
    ds_cleaned = repository.read(out_url)

    assert "height" not in ds_cleaned.coords
    assert "member_id" not in ds_cleaned.coords
    assert "time_bnds" not in ds_cleaned.coords

    if "pr" in ds_cleaned.variables:
        assert ds_cleaned["pr"].units == "mm day-1"


def test_remove_leapdays():
    """Test that leapday removal service removes leap days"""
    # Setup input data
    n = 1500  # need over four years of daily data
    ts = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 0.5
    ds_leap = _gcmfactory(ts, start_time="1950-01-01")

    in_url = "memory://test_remove_leapdays/an/input/path.zarr"
    out_url = "memory://test_remove_leapdays/an/output/path.zarr"
    repository.write(in_url, ds_leap)

    remove_leapdays(in_url, out_url)
    ds_noleap = repository.read(out_url)
    ds_leapyear = ds_noleap.loc[dict(time=slice("1952-01-01", "1952-12-31"))]

    # check to be sure that leap days have been removed
    assert len(ds_leapyear.time) == 365


@pytest.mark.parametrize("process", [pytest.param("pre"), pytest.param("post")])
def test_correct_wet_day_frequency(process):
    """Test that wet day frequency correction corrects the frequency of wet days"""
    # Make some fake precip data
    n = 700
    threshold = 1.0  # mm/day
    ts = np.linspace(0.0, 10, num=n)
    ds_precip = _datafactory(ts, start_time="1950-01-01")
    in_url = "memory://test_correct_wet_day_frequency/an/input/path.zarr"
    out_url = "memory://test_correct_wet_day_frequency/an/output/path.zarr"
    repository.write(in_url, ds_precip)

    correct_wet_day_frequency(in_url, out=out_url, process=process)
    ds_precip_corrected = repository.read(out_url)

    low = threshold / 2.0
    if process == "pre":
        # all 0s and very small negative values should have been set to a random uniform value below threshold and above threshold / 2.0
        corrected_values = ds_precip_corrected["fakevariable"].where(
            ds_precip["fakevariable"] < threshold, drop=True
        )
        assert corrected_values.all() >= low
        assert corrected_values.all() <= threshold
    elif process == "post":
        # all values below threshold should be reset to 0
        assert (
            ds_precip_corrected["fakevariable"]
            .where(ds_precip["fakevariable"] < threshold, drop=True)
            .all()
            == 0.0
        )


def test_correct_small_dtr():
    """Test that diurnal temperature range (DTR) correction corrects small values of DTR"""
    # Make some fake dtr data
    n = 700
    threshold = 1.0
    ts = np.linspace(0.0, 10, num=n)
    ds_dtr = _datafactory(ts, start_time="1950-01-01")
    in_url = "memory://test_correct_small_dtr/an/input/path.zarr"
    out_url = "memory://test_correct_small_dtr/an/output/path.zarr"
    repository.write(in_url, ds_dtr)

    correct_small_dtr(in_url, out=out_url)
    ds_dtr_corrected = repository.read(out_url)

    # all values below threshold should have been set to the threshold value
    assert (
        ds_dtr_corrected["fakevariable"]
        .where(ds_dtr["fakevariable"] < threshold, drop=True)
        .all()
        >= threshold
    )


def test_adjust_maximum_precipitation():
    """Test that maximum precipitation adjustment corrects precipitation values above a set threshold"""
    # make some fake precip data
    n = 700
    ts = np.linspace(0.0, 4000, num=n)
    threshold = 3000  # mm/day
    ds_precip = _datafactory(ts, start_time="1950-01-01")
    in_url = "memory://test_adjust_maximum_precipitation/an/input/path.zarr"
    out_url = "memory://test_adjust_maximum_precipitation/an/output/path.zarr"
    repository.write(in_url, ds_dtr)

    adjust_maximum_precipitation(in_url, out=out_url, threshold=threshold)
    ds_precip_corrected = repository.read(out_url)

    # all values above threshold should have been set to the threshold value
    assert (
        ds_precip_corrected["fakevariable"]
        .where(ds_precip["fakevariable"] > threshold, drop=True)
        .all()
        <= threshold
    )


@pytest.mark.parametrize("kind", ["multiplicative", "additive"])
def test_qplad_train(tmpdir, monkeypatch, kind):
    """Tests that the shape of adjustment factors matches the expected shape"""
    monkeypatch.setenv(
        "HDF5_USE_FILE_LOCKING", "FALSE"
    )  # Avoid thread lock conflicts with dask scheduler
    # make test data
    np.random.seed(0)
    lon = [-99.83, -99.32, -99.79, -99.23]
    lat = [42.25, 42.21, 42.63, 42.59]
    time = xr.cftime_range(start="1994-12-17", end="2015-01-15", calendar="noleap")
    data_ref = 15 + 8 * np.random.randn(len(time), 4, 4)

    ref_fine = xr.Dataset(
        data_vars=dict(
            scen=(["time", "lat", "lon"], data_ref),
        ),
        coords=dict(
            time=time,
            lon=(["lon"], lon),
            lat=(["lat"], lat),
        ),
        attrs=dict(description="Weather related data."),
    )
    # need to set variable units to pass xclim 0.29 check on units
    ref_fine["scen"].attrs["units"] = "K"

    # take the mean across space to represent coarse reference data for AFs
    ds_ref_coarse = ref_fine.mean(["lat", "lon"])
    # tile the fine resolution grid with the coarse resolution ref data
    ref_coarse = ds_ref_coarse.broadcast_like(ref_fine)
    ref_coarse["scen"].attrs["units"] = "K"

    # write test data
    ref_coarse_url = "memory://test_qplad_downscaling/a/ref_coarse/path.zarr"
    ref_fine_url = "memory://test_qplad_downscaling/a/ref_fine/path.zarr"
    train_out_url = "memory://test_qplad_downscaling/a/train_output/path.zarr"

    repository.write(
        ref_coarse_url,
        ref_coarse.chunk({"time": -1}),
    )
    repository.write(ref_fine_url, ref_fine.chunk({"time": -1}))

    # now train QPLAD model
    train_qplad(ref_coarse_url, ref_fine_url, train_out_url, "scen", kind)

    # load adjustment factors
    qplad_model = repository.read(train_out_url)

    af_expected_shape = (len(lon), len(lat), 365, 620)

    assert qplad_model.af.shape == af_expected_shape


def test_train_qplad_isel_slice():
    """Tests that services.train_qplad subsets with isel_slice"""
    lon = [-99.83, -99.32, -99.79, -99.23]
    lat = [42.25, 42.21, 42.63, 42.59]
    time = xr.cftime_range(start="1994-12-17", end="2015-01-15", calendar="noleap")
    data_ref = 15 + 8 * np.ones((len(time), 4, 4))
    ref_fine = xr.Dataset(
        data_vars=dict(
            scen=(["time", "lat", "lon"], data_ref),
        ),
        coords=dict(
            time=time,
            lon=(["lon"], lon),
            lat=(["lat"], lat),
        ),
        attrs=dict(description="Weather related data."),
    )
    # need to set variable units to pass xclim 0.29 check on units
    ref_fine["scen"].attrs["units"] = "K"
    # take the mean across space to represent coarse reference data for AFs
    ds_ref_coarse = ref_fine.mean(["lat", "lon"])
    # tile the fine resolution grid with the coarse resolution ref data
    ref_coarse = ds_ref_coarse.broadcast_like(ref_fine)
    ref_coarse["scen"].attrs["units"] = "K"

    # write test data
    ref_coarse_url = "memory://train_qplad_isel_slice/a/ref_coarse/path.zarr"
    ref_fine_url = "memory://train_qplad_isel_slice/a/ref_fine/path.zarr"
    train_out_url = "memory://train_qplad_isel_slice/a/train_output/path.zarr"

    repository.write(ref_coarse_url, ref_coarse.chunk({"time": -1}))
    repository.write(ref_fine_url, ref_fine.chunk({"time": -1}))

    # now train QPLAD model
    train_qplad(
        coarse_reference=ref_coarse_url,
        fine_reference=ref_fine_url,
        out=train_out_url,
        variable="scen",
        kind="additive",
        isel_slice={"lat": slice(0, 3)},
    )

    qplad_model = repository.read(train_out_url)
    assert qplad_model["lat"].shape == (3,)


def test_train_qplad_sel_slice():
    """Tests that services.train_qplad subsets with sel_slice"""
    # This should prob go to a test fixture for input data setup.
    lon = [-99.83, -99.32, -99.79, -99.23]
    lat = [42.25, 42.21, 42.63, 42.59]
    time = xr.cftime_range(start="1994-12-17", end="2015-01-15", calendar="noleap")
    data_ref = 15 + 8 * np.ones((len(time), 4, 4))
    ref_fine = xr.Dataset(
        data_vars=dict(
            scen=(["time", "lat", "lon"], data_ref),
        ),
        coords=dict(
            time=time,
            lon=(["lon"], lon),
            lat=(["lat"], lat),
        ),
        attrs=dict(description="Weather related data."),
    )
    # need to set variable units to pass xclim 0.29 check on units
    ref_fine["scen"].attrs["units"] = "K"
    # take the mean across space to represent coarse reference data for AFs
    ds_ref_coarse = ref_fine.mean(["lat", "lon"])
    # tile the fine resolution grid with the coarse resolution ref data
    ref_coarse = ds_ref_coarse.broadcast_like(ref_fine)
    ref_coarse["scen"].attrs["units"] = "K"

    ref_coarse_url = "memory://test_train_qplad_sel_slice/a/ref_coarse/path.zarr"
    ref_fine_url = "memory://test_train_qplad_sel_slice/a/ref_fine/path.zarr"
    train_out_url = "memory://test_train_qplad_sel_slice/a/train_output/path.zarr"

    repository.write(ref_coarse_url, ref_coarse.chunk({"time": -1}))
    repository.write(ref_fine_url, ref_fine.chunk({"time": -1}))

    train_qplad(
        coarse_reference=ref_coarse_url,
        fine_reference=ref_fine_url,
        out=train_out_url,
        variable="scen",
        kind="additive",
        sel_slice={"lat": slice(lat[0], lat[2])},
    )

    qplad_model = repository.read(train_out_url)
    assert qplad_model["lat"].shape == (3,)


@pytest.mark.parametrize("kind", ["multiplicative", "additive"])
def test_qplad_integration(kind):
    """Integration test of the QDM and QPLAD services"""
    lon = [-99.83, -99.32, -99.79, -99.23]
    lat = [42.25, 42.21, 42.63, 42.59]
    time = xr.cftime_range(start="1994-12-17", end="2015-01-15", calendar="noleap")
    data_ref = 15 + 8 * np.random.randn(len(time), 4, 4)
    data_train = 15 + 8 * np.random.randn(len(time), 4, 4)
    variable = "scen"

    ref_fine = xr.Dataset(
        data_vars=dict(
            scen=(["time", "lat", "lon"], data_ref),
        ),
        coords=dict(
            time=time,
            lon=(["lon"], lon),
            lat=(["lat"], lat),
        ),
        attrs=dict(description="Weather related data."),
    )

    ds_train = xr.Dataset(
        data_vars=dict(
            scen=(["time", "lat", "lon"], data_train),
        ),
        coords=dict(
            time=time,
            lon=(["lon"], lon),
            lat=(["lat"], lat),
        ),
        attrs=dict(description="Weather related data."),
    )

    # take the mean across space to represent coarse reference data for AFs
    ds_ref_coarse = ref_fine.mean(["lat", "lon"])
    ds_train = ds_train.mean(["lat", "lon"])

    # tile the fine resolution grid with the coarse resolution ref data
    ref_coarse = ds_ref_coarse.broadcast_like(ref_fine)
    ds_bc = ds_train + 3

    # need to set variable units to pass xclim 0.29 check on units
    ds_train["scen"].attrs["units"] = "K"
    ds_bc["scen"].attrs["units"] = "K"
    ref_coarse["scen"].attrs["units"] = "K"
    ref_fine["scen"].attrs["units"] = "K"
    ds_ref_coarse["scen"].attrs["units"] = "K"

    # write test data
    ref_coarse_coarse_url = (
        "memory://test_qplad_downscaling/a/ref_coarse_coarse/path.zarr"
    )
    ref_coarse_url = "memory://test_qplad_downscaling/a/ref_coarse/path.zarr"
    ref_fine_url = "memory://test_qplad_downscaling/a/ref_fine/path.zarr"
    qdm_train_url = "memory://test_qplad_downscaling/a/qdm_train/path.zarr"
    sim_url = "memory://test_qplad_downscaling/a/sim/path.zarr"
    qdm_train_out_url = "memory://test_qplad_downscaling/a/qdm_train_out/path.zarr"
    biascorrected_url = "memory://test_qplad_downscaling/a/biascorrected/path.zarr"
    sim_biascorrected_key = (
        "memory://test_qplad_downscaling/a/biascorrected/sim_biascorrected.zarr"
    )

    repository.write(ref_coarse_coarse_url, ds_ref_coarse)
    repository.write(
        ref_coarse_url,
        ref_coarse.chunk({"time": -1, "lat": -1, "lon": -1}),
    )
    repository.write(
        ref_fine_url,
        ref_fine.chunk({"time": -1, "lat": -1, "lon": -1}),
    )
    repository.write(qdm_train_url, ds_train)
    repository.write(sim_url, ds_bc)

    # this is an integration test between QDM and QPLAD, so use QDM services
    # for bias correction
    target_year = 2005

    train_qdm(
        historical=qdm_train_url,
        reference=ref_coarse_coarse_url,
        out=qdm_train_out_url,
        variable=variable,
        kind=kind,
    )
    apply_qdm(
        simulation=sim_url,
        qdm=qdm_train_out_url,
        years=[target_year],
        variable=variable,
        out=sim_biascorrected_key,
    )
    biascorrected_coarse = repository.read(sim_biascorrected_key)
    # make bias corrected data on the fine resolution grid
    biascorrected_fine = biascorrected_coarse.broadcast_like(
        ref_fine.sel(
            time=slice("{}-01-01".format(target_year), "{}-12-31".format(target_year))
        )
    )
    repository.write(
        biascorrected_url,
        biascorrected_fine.chunk({"time": -1, "lat": -1, "lon": -1}),
    )

    # write test data
    qplad_afs_url = "memory://test_qplad_downscaling/a/qplad_afs/path.zarr"

    # Writes NC to local disk, so diff format here:
    sim_downscaled_url = "memory://test_qplad_downscaling/a/qplad_afs/downscaled.zarr"

    # now train QPLAD model
    train_qplad(ref_coarse_url, ref_fine_url, qplad_afs_url, variable, kind)

    # downscale
    apply_qplad(biascorrected_url, qplad_afs_url, variable, sim_downscaled_url)

    # check output
    downscaled_ds = repository.read(sim_downscaled_url)

    # check that downscaled average equals bias corrected value
    bc_timestep = biascorrected_fine[variable].isel(time=100).values[0][0]
    qplad_downscaled_mean = downscaled_ds[variable].isel(time=100).mean().values
    np.testing.assert_almost_equal(bc_timestep, qplad_downscaled_mean)


@pytest.mark.parametrize("variable", ["tasmax", "tasmin", "dtr", "pr"])
@pytest.mark.parametrize("data_type", ["cmip6", "bias_corrected", "downscaled"])
@pytest.mark.parametrize("time_period", ["historical", "future"])
def test_validation(variable, data_type, time_period):
    """Tests that validate passes for fake output data"""
    # Setup input data
    start_time = None
    end_time = None
    if data_type == "bias_corrected" or data_type == "downscaled":
        if time_period == "historical":
            start_time = "1950-01-01"
            end_time = "2014-12-31"
        else:
            start_time = "2015-01-01"
            end_time = "2100-12-31"
    elif data_type == "cmip6":
        if time_period == "historical":
            start_time = "1950-01-01"
            end_time = "2025-12-31"
        else:
            start_time = "2004-01-01"
            end_time = "2100-12-31"

    # create test data
    ds = _modeloutputfactory(
        start_time=start_time, end_time=end_time, variable_name=variable
    )

    # write test data
    in_url = "memory://test_validate/an/input/path.zarr"
    repository.write(in_url, ds)

    validate(in_url, variable, data_type, time_period)


def test_get_attrs_global():
    """Test that services.get_attrs returns json of global attrs"""
    url = "memory://test_get_attrs_global/x.zarr"
    repository.write(url, xr.Dataset({"bar": "SPAM"}, attrs={"fish": "chips"}))
    out = get_attrs(url)
    assert out == '{"fish": "chips"}'


def test_get_attrs_variable():
    """Test that services.get_attrs returns json of variable attrs"""
    url = "memory://test_get_attrs/x.zarr"
    variable_name = "bar"
    ds_in = xr.Dataset({variable_name: "SPAM"}, attrs={"fish": "chips"})
    ds_in[variable_name].attrs["carrot"] = "sticks"

    repository.write(url, ds_in)
    out = get_attrs(url, variable=variable_name)

    assert out == '{"carrot": "sticks"}'
