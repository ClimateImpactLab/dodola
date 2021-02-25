"""Test application services
"""

import numpy as np
import pytest
import xarray as xr
from xesmf.data import wave_smooth
from xesmf.util import grid_global
from dodola.services import bias_correct, build_weights, rechunk
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


@pytest.mark.parametrize(
    "method, expected_head, expected_tail",
    [
        pytest.param(
            "QDM",
            np.array([4.0, 4.008609, 4.0172155, 4.0258169, 4.0344106]),
            np.array([3.9655894, 3.9741831, 3.9827845, 3.991391, 4.0]),
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
        target_chunks={"fakevariable": chunks_goal},
        out="output_ds",
        max_mem=256000,
        storage=fakestorage,
    )

    assert fakestorage.read("output_ds")["fakevariable"].chunks == chunks_goal
