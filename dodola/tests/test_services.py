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
from dodola.repository import memory_repository


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

    output_key = "test_output"
    training_model_key = "x_train"
    training_obs_key = "y_train"
    forecast_model_key = "x_test"

    # Load up a fake repo with our input data in the place of big data and cloud
    # storage.
    fakestorage = memory_repository(
        {
            training_model_key: x_train,
            training_obs_key: y_train,
            forecast_model_key: x_test,
        }
    )

    bias_correct(
        forecast_model_key,
        x_train=training_model_key,
        train_variable="fakevariable",
        y_train=training_obs_key,
        out=output_key,
        out_variable="fakevariable",
        method=method,
        storage=fakestorage,
    )

    # We can't just test for removal of bias here since quantile mapping
    # and adding in trend are both components of bias correction,
    # so testing head and tail values instead
    np.testing.assert_almost_equal(
        fakestorage.read(output_key)["fakevariable"].squeeze(drop=True).values[:5],
        expected_head,
    )
    np.testing.assert_almost_equal(
        fakestorage.read(output_key)["fakevariable"].squeeze(drop=True).values[-5:],
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

    fakestorage = memory_repository(
        {
            "a_file_path": ds_in,
        }
    )

    build_weights(
        "a_file_path", method=regrid_method, storage=fakestorage, outpath=weightsfile
    )
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

    fakestorage = memory_repository({"input_ds": test_ds})

    rechunk(
        "input_ds",
        target_chunks=chunks_goal,
        out="output_ds",
        storage=fakestorage,
    )
    actual_chunks = fakestorage.read("output_ds")["fakevariable"].data.chunksize

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

    fakestorage = memory_repository(
        {
            "an/input/path.zarr": ds_in,
            "a/domainfile/path.zarr": domain_file,
        }
    )

    regrid(
        "an/input/path.zarr",
        out="an/output/path.zarr",
        method=regrid_method,
        storage=fakestorage,
        domain_file="a/domainfile/path.zarr",
    )
    actual_shape = fakestorage.read("an/output/path.zarr")["fakevariable"].shape
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

    fakestorage = memory_repository(
        {
            "an/input/path.zarr": ds_in,
            "a/domainfile/path.zarr": domain_file,
        }
    )

    regrid(
        "an/input/path.zarr",
        out="an/output/path.zarr",
        method="bilinear",
        storage=fakestorage,
        domain_file="a/domainfile/path.zarr",
    )
    actual_shape = fakestorage.read("an/output/path.zarr")["fakevariable"].shape
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

    fakestorage = memory_repository(
        {
            "an/input/path.zarr": ds_in,
            "a/domainfile/path.zarr": domain_file,
        }
    )

    # First, use service to pre-build regridding weights files, then read-in to regrid.
    build_weights(
        "an/input/path.zarr",
        method="bilinear",
        storage=fakestorage,
        outpath=weightsfile,
    )
    regrid(
        "an/input/path.zarr",
        out="an/output/path.zarr",
        method="bilinear",
        weights_path=weightsfile,
        storage=fakestorage,
        domain_file="a/domainfile/path.zarr",
    )
    actual_shape = fakestorage.read("an/output/path.zarr")["fakevariable"].shape
    assert actual_shape == expected_shape


def test_clean_cmip6():
    """ Tests that cmip6 cleanup removes extra dimensions on dataset """
    # Setup input data
    n = 1500  # need over four years of daily data
    ts = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 0.5
    ds_gcm = _gcmfactory(ts, start_time="1950-01-01")

    fakestorage = memory_repository(
        {
            "an/input/path.zarr": ds_gcm,
        }
    )

    clean_cmip6(
        "an/input/path.zarr",
        "an/output/path.zarr",
        storage=fakestorage,
        leapday_removal=True,
    )
    ds_cleaned = fakestorage.read("an/output/path.zarr")

    assert "height" not in ds_cleaned.dims
    assert "member_id" not in ds_cleaned.dims
    assert "time_bnds" not in ds_cleaned.dims


def test_remove_leapdays():
    """ Test that leapday removal service removes leap days """
    # Setup input data
    n = 1500  # need over four years of daily data
    ts = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 0.5
    ds_leap = _gcmfactory(ts, start_time="1950-01-01")

    fakestorage = memory_repository(
        {
            "an/input/path.zarr": ds_leap,
        }
    )

    remove_leapdays("an/input/path.zarr", "an/output/path.zarr", storage=fakestorage)
    ds_noleap = fakestorage.read("an/output/path.zarr")
    ds_leapyear = ds_noleap.loc[dict(time=slice("1952-01-01", "1952-12-31"))]

    # check to be sure that leap days have been removed
    assert len(ds_leapyear.time) == 365


@pytest.mark.parametrize(
    "domain_file, method, var",
    [
        pytest.param(0.25, "BCSD", "temperature"),
        pytest.param(0.25, "BCSD", "precipitation"),
    ],
    indirect=["domain_file"],
)
def test_downscale(domain_file, method, var):
    """Simple test of downscaling service"""
    # set up test data
    start_time = "1950-01-01"
    end_time = "1951-12-31"

    # make fake bias corrected data
    ds_bc = grid_global(1, 1)
    time = pd.date_range(start_time, end_time, freq="D")
    ds_bc["fakevariable"] = xr.DataArray(
        np.random.randn(len(time), len(ds_bc["y"]), len(ds_bc["x"])),
        dims=("time", "y", "x"),
        coords={"time": time, "y": ds_bc["y"], "x": ds_bc["x"]},
    )

    # make fake climatology at coarse res
    ds_for_climo = grid_global(1, 1)
    ds_for_climo["fakevariable"] = xr.DataArray(
        np.random.randn(len(time), len(ds_for_climo["y"]), len(ds_for_climo["x"])),
        dims=("time", "y", "x"),
        coords={"time": time, "y": ds_for_climo["y"], "x": ds_for_climo["x"]},
    )
    climo_coarse = ds_for_climo.groupby("time.dayofyear").mean()

    # compute adjustment factor
    if var == "temperature":
        af_coarse = ds_bc["fakevariable"].groupby("time.dayofyear") - climo_coarse
    elif var == "precipitation":
        af_coarse = ds_bc["fakevariable"].groupby("time.dayofyear") / climo_coarse

    fakestorage = memory_repository(
        {
            "a/biascorrected/path.zarr": ds_bc,
            "a/domainfile/path.zarr": domain_file,
            "a/coarseclimo/path.zarr": climo_coarse,
            "a/coarseaf/path.zarr": af_coarse,
        }
    )

    # regrid climatology to fine resolution
    regrid(
        "a/coarseclimo/path.zarr",
        out="a/fineclimo/path.zarr",
        method="bilinear",
        storage=fakestorage,
        domain_file="a/domainfile/path.zarr",
    )
    climo_fine = fakestorage.read("a/fineclimo/path.zarr")["fakevariable"]

    # regrid adjustment factor
    regrid(
        "a/coarseaf/path.zarr",
        out="a/fineaf/path.zarr",
        method="bilinear",
        storage=fakestorage,
        domain_file="a/domainfile/path.zarr",
    )
    af_fine = fakestorage.read("a/fineaf/path.zarr")

    # compute test downscaled values
    if var == "temperature":
        downscaled_test = af_fine + climo_fine
    elif var == "precipitation":
        downscaled_test = af_fine * climo_fine

    afs, downscaled_ds = downscale(
        ds_bc,
        climo_coarse,
        climo_fine,
        "a/downscaled/path.zarr",
        storage=fakestorage,
        train_variable="fakevariable",
        out_variable="fakevariable",
        method=method,
        domain_file=domain_file,
    )

    np.testing.assert_almost_equal(downscaled_ds.values, downscaled_test.values)
