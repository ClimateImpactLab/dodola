"""Test application services
"""

import numpy as np
import pytest
import pandas as pd
import xarray as xr
from xesmf.data import wave_smooth
from xesmf.util import grid_global
from dodola.services import (
    bias_correct,
    build_weights,
    rechunk,
    regrid,
    remove_leapdays,
    clean_cmip6,
    downscale,
)
import dodola.repository as repository


def _datafactory(x, start_time="1950-01-01"):
    """Populate xr.Dataset with synthetic data for testing"""
    start_time = str(start_time)
    if x.ndim != 1:
        raise ValueError("'x' needs dim of one")

    time = xr.cftime_range(
        start=start_time, freq="D", periods=len(x), calendar="standard"
    )

    out = xr.Dataset(
        {"fakevariable": (["time", "lon", "lat"], x[:, np.newaxis, np.newaxis])},
        coords={
            "index": time,
            "time": time,
            "lon": (["lon"], [1.0]),
            "lat": (["lat"], [1.0]),
        },
    )
    return out


def _gcmfactory(x, start_time="1950-01-01"):
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
            "fakevariable": (
                ["time", "lon", "lat", "member_id"],
                x[:, np.newaxis, np.newaxis, np.newaxis],
            )
        },
        coords={
            "index": time,
            "time": time,
            "lon": (["lon"], [1.0]),
            "lat": (["lat"], [1.0]),
            "member_id": (["member_id"], [1.0]),
            "height": (["height"], [1.0]),
            "time_bnds": (["time_bnds"], [1.0]),
        },
    )
    return out


@pytest.fixture
def domain_file(request):
    """ Creates a fake domain Dataset for testing"""
    lon_name = "lon"
    lat_name = "lat"
    domain = grid_global(request.param, request.param)
    domain[lat_name] = np.unique(domain[lat_name].values)
    domain[lon_name] = np.unique(domain[lon_name].values)

    return domain


@pytest.mark.parametrize(
    "method, expected_head, expected_tail",
    [
        pytest.param(
            "QDM",
            np.array(
                [6.6613381e-16, 8.6090365e-3, 1.7215521e-2, 2.58169e-2, 3.4410626e-2]
            ),
            np.array(
                [
                    -3.4410626e-2,
                    -2.58169e-2,
                    -1.7215521e-2,
                    -8.6090365e-3,
                    -1.110223e-15,
                ]
            ),
            id="QDM head/tail",
        ),
    ],
)
def test_bias_correct_basic_call(method, expected_head, expected_tail):
    """Simple integration test of bias_correct service"""
    # Setup input data.
    n_years = 10
    n = n_years * 365  # need daily data...

    # Our "biased model".
    model_bias = 2
    ts = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 0.5
    x_train = _datafactory((ts + model_bias))
    # True "observations".
    y_train = _datafactory(ts)
    # Yes, we're testing and training on the same data...
    x_test = x_train.copy(deep=True)

    output_key = "memory://test_bias_correct_basic_call/test_output.zarr"
    training_model_key = "memory://test_bias_correct_basic_call/x_train.zarr"
    training_obs_key = "memory://test_bias_correct_basic_call/y_train.zarr"
    forecast_model_key = "memory://test_bias_correct_basic_call/x_test.zarr"

    # Load up a fake repo with our input data in the place of big data and cloud
    # storage.
    repository.write(training_model_key, x_train)
    repository.write(training_obs_key, y_train)
    repository.write(forecast_model_key, x_test)

    bias_correct(
        forecast_model_key,
        x_train=training_model_key,
        train_variable="fakevariable",
        y_train=training_obs_key,
        out=output_key,
        out_variable="fakevariable",
        method=method,
    )

    # We can't just test for removal of bias here since quantile mapping
    # and adding in trend are both components of bias correction,
    # so testing head and tail values instead
    np.testing.assert_almost_equal(
        repository.read(output_key)["fakevariable"].squeeze(drop=True).values[:5],
        expected_head,
    )
    np.testing.assert_almost_equal(
        repository.read(output_key)["fakevariable"].squeeze(drop=True).values[-5:],
        expected_tail,
    )


@pytest.mark.parametrize("regrid_method", ["bilinear", "conservative"])
def test_build_weights(regrid_method, tmpdir):
    """Test that services.build_weights produces a weights file"""
    # Output to tmp dir so we cleanup & don't clobber existing files...
    weightsfile = tmpdir.join("a_file_path_weights.nc")

    # Make fake input data.
    ds_in = grid_global(30, 20)
    ds_in["fakevariable"] = wave_smooth(ds_in["lon"], ds_in["lat"])

    url = "memory://test_build_weights/a_file_path.zarr"
    repository.write(url, ds_in)

    build_weights(url, method=regrid_method, outpath=weightsfile)
    # Test that weights file is actually created where we asked.
    assert weightsfile.exists()


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


@pytest.mark.parametrize(
    "domain_file", [pytest.param(1.0, id="Regrid to domain file grid")], indirect=True
)
def test_regrid_weights_integration(domain_file, tmpdir):
    """Test basic integration between service.regrid and service.build_weights"""
    expected_shape = (180, 360)
    # Output to tmp dir so we cleanup & don't clobber existing files...
    weightsfile = tmpdir.join("a_file_path_weights.nc")

    # Make fake input data.
    ds_in = grid_global(30, 20)
    ds_in["fakevariable"] = wave_smooth(ds_in["lon"], ds_in["lat"])

    in_url = "memory://test_regrid_weights_integration/an/input/path.zarr"
    domain_file_url = "memory://test_regrid_weights_integration/a/domainfile/path.zarr"
    out_url = "memory://test_regrid_weights_integration/an/output/path.zarr"
    repository.write(in_url, ds_in)
    repository.write(domain_file_url, domain_file)

    # First, use service to pre-build regridding weights files, then read-in to regrid.
    build_weights(in_url, method="bilinear", outpath=weightsfile)
    regrid(
        in_url,
        out=out_url,
        method="bilinear",
        weights_path=weightsfile,
        domain_file=domain_file_url,
    )
    actual_shape = repository.read(out_url)["fakevariable"].shape
    assert actual_shape == expected_shape


def test_clean_cmip6():
    """ Tests that cmip6 cleanup removes extra dimensions on dataset """
    # Setup input data
    n = 1500  # need over four years of daily data
    ts = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 0.5
    ds_gcm = _gcmfactory(ts, start_time="1950-01-01")

    in_url = "memory://test_clean_cmip6/an/input/path.zarr"
    out_url = "memory://test_clean_cmip6/an/output/path.zarr"
    repository.write(in_url, ds_gcm)

    clean_cmip6(in_url, out_url, leapday_removal=True)
    ds_cleaned = repository.read(out_url)

    assert "height" not in ds_cleaned.dims
    assert "member_id" not in ds_cleaned.dims
    assert "time_bnds" not in ds_cleaned.dims


def test_remove_leapdays():
    """ Test that leapday removal service removes leap days """
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


@pytest.mark.parametrize(
    "domain_file, method, var",
    [
        pytest.param(30, "BCSD", "temperature"),
        pytest.param(30, "BCSD", "precipitation"),
    ],
    indirect=["domain_file"],
)
def test_downscale(domain_file, method, var):
    """Simple test of downscaling service"""
    # set up test data
    start_time = "1950-01-01"
    end_time = "1951-12-31"

    # make fake bias corrected data
    res = 36
    lat_name = "lat"
    lon_name = "lon"
    ds_bc = grid_global(res, res)
    ds_bc = ds_bc.rename({"x": lon_name, "y": lat_name})
    ds_bc[lat_name] = np.unique(ds_bc[lat_name].values)
    ds_bc[lon_name] = np.unique(ds_bc[lon_name].values)
    time = pd.date_range(start_time, end_time, freq="D")
    ds_bc[var] = xr.DataArray(
        np.random.randn(len(time), len(ds_bc["lat"]), len(ds_bc["lon"])),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": ds_bc["lat"], "lon": ds_bc["lon"]},
    )
    ds_bc = ds_bc.drop(["lon_b", "lat_b"])

    # make fake climatology at coarse res
    ds_for_climo = grid_global(res, res)
    ds_for_climo = ds_for_climo.rename({"x": lon_name, "y": lat_name})
    ds_for_climo[lat_name] = np.unique(ds_for_climo[lat_name].values)
    ds_for_climo[lon_name] = np.unique(ds_for_climo[lon_name].values)
    ds_for_climo[var] = xr.DataArray(
        np.random.randn(len(time), len(ds_for_climo["lat"]), len(ds_for_climo["lon"])),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": ds_for_climo["lat"], "lon": ds_for_climo["lon"]},
    )
    climo_coarse = (
        ds_for_climo.groupby("time.dayofyear").mean().drop(["lon_b", "lat_b"])
    )

    # compute adjustment factor
    if var == "temperature":
        af_coarse = ds_bc.groupby("time.dayofyear") - climo_coarse
    elif var == "precipitation":
        af_coarse = ds_bc.groupby("time.dayofyear") / climo_coarse

    ds_bc_url = "memory://test_downscale/a/biascorrected/path.zarr"
    repository.write(ds_bc_url, ds_bc)

    domain_file_url = "memory://test_downscale/a/domainfile/path.zarr"
    repository.write(domain_file_url, domain_file)

    climo_coarse_url = "memory://test_downscale/a/coarseclimo/path.zarr"
    repository.write(climo_coarse_url, climo_coarse)

    af_coarse_url = "memory://test_downscale/a/coarseaf/path.zarr"
    repository.write(af_coarse_url, af_coarse)

    climo_fine_url = "memory://test_downscale/a/fineclimo/path.zarr"

    af_fine_url = "memory://test_downscale/a/fineaf/path.zarr"

    downscaled_url = "memory://test_downscale/a/downscaled/path.zarr"

    af_saved_url = "memory://test_downscale/a/afsaved/path.zarr"

    # regrid climatology to fine resolution
    regrid(
        climo_coarse_url,
        out=climo_fine_url,
        method="bilinear",
        domain_file=domain_file_url,
    )
    climo_fine = repository.read(climo_fine_url)[var]

    # regrid adjustment factor
    regrid(
        af_coarse_url,
        af_fine_url,
        method="bilinear",
        domain_file=domain_file_url,
    )
    af_fine = repository.read(af_fine_url)[var]

    # compute test downscaled values
    if var == "temperature":
        downscaled_test = af_fine.groupby("time.dayofyear") + climo_fine
    elif var == "precipitation":
        downscaled_test = af_fine.groupby("time.dayofyear") * climo_fine

    downscale(
        ds_bc_url,
        climo_coarse_url,
        climo_fine_url,
        downscaled_url,
        af_saved_url,
        train_variable=var,
        out_variable=var,
        method=method,
        domain_file=domain_file_url,
        weights_path=None,
    )
    downscaled_ds = repository.read(downscaled_url)[var]
    np.testing.assert_almost_equal(downscaled_ds.values, downscaled_test.values)
