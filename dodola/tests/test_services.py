"""Test application services
"""

import json
import fsspec
import numpy as np
import pytest
import xarray as xr
from xesmf.data import wave_smooth
from xesmf.util import grid_global
from xclim.sdba.adjustment import QuantileDeltaMapping
from dodola.services import (
    bias_correct,
    build_weights,
    rechunk,
    regrid,
    remove_leapdays,
    clean_cmip6,
    find_qdm_rollingyearwindow,
    train_qdm,
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
    # out['time'] = out['time'].assign_attrs({'calendar': 'standard'})
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


@pytest.mark.parametrize("kind", ["precipitation", "temperature"])
def test_train_qdm(kind):
    """Test that train_qdm outputs store giving sdba.adjustment.QuantileDeltaMapping

    Checks that output is consistent if we do "temperature" or "precipitation"
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


def test_find_qdm_rollingyearwindow():
    """Basic test that find_qdm_rollingyearwindow output correct time range in JSON"""
    # Create test data
    expected = {"firstyear": 2026, "lastyear": 2088}
    t = xr.cftime_range(
        start="2015-01-01", freq="D", end="2100-01-01", calendar="noleap"
    )
    x = np.ones(len(t))
    in_ds = xr.Dataset({"fakevariable": (["time"], x)}, coords={"time": t})
    in_key = "memory://test_find_qdm_rollingyearwindow/test_in.zarr"
    out_key = "memory://test_find_qdm_rollingyearwindow/test_out.json"
    repository.write(in_key, in_ds)

    find_qdm_rollingyearwindow(in_key, out_key)

    # Can't use repository read because output is JSON...
    with fsspec.open(out_key) as fl:
        actual = json.load(fl)
    assert actual == expected


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
        pytest.param(
            "BCSD",
            np.array([-0.08129293, -0.07613746, -0.0709855, -0.0658377, -0.0606947]),
            np.array([0.0520793, 0.06581804, 0.07096781, 0.07612168, 0.08127902]),
            id="BCSD head/tail",
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

    assert "height" not in ds_cleaned.dims
    assert "member_id" not in ds_cleaned.dims
    assert "time_bnds" not in ds_cleaned.dims


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
