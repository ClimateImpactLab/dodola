"""Test application services
"""

import numpy as np
import xarray as xr
from dodola.services import bias_correct, build_weights
from dodola.repository import FakeRepository


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


def test_bias_correct_basic_call():
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
    fakestorage = FakeRepository(
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
        storage=fakestorage,
    )

    # We can't just test for removal of bias here since quantile mapping
    # and adding in trend are both components of bias correction,
    # so testing head and tail values instead
    head_vals = np.array([-0.08129293, -0.07613746, -0.0709855, -0.0658377, -0.0606947])
    tail_vals = np.array([0.0520793, 0.06581804, 0.07096781, 0.07612168, 0.08127902])
    np.testing.assert_almost_equal(
        fakestorage.storage[output_key]["fakevariable"].squeeze(drop=True).values[:5],
        head_vals,
    )
    np.testing.assert_almost_equal(
        fakestorage.storage[output_key]["fakevariable"].squeeze(drop=True).values[-5:],
        tail_vals,
    )


def test_build_weights(tmpdir):
    """Test that services.build_weights produces a weights file"""
    # Output to tmp dir so we cleanup & don't clobber existing files...
    weightsfile = tmpdir.join("a_file_path_weights.nc")

    # Make fake input data.
    lat_n = 30
    lon_n = 60
    ds_in = xr.Dataset(
        {"fakevariable": (["lon", "lat"], np.ones(shape=(lon_n, lat_n)))},
        coords={
            "lon": (["lon"], np.linspace(-180, 179, lon_n)),
            "lat": (["lat"], np.linspace(-90, 90, lat_n)),
        },
    )

    fakestorage = FakeRepository(
        {
            "a_file_path": ds_in,
        }
    )

    build_weights(
        "a_file_path", method="bilinear", storage=fakestorage, outpath=weightsfile
    )
    # Test that weights file is actually created where we asked.
    assert weightsfile.exists()
