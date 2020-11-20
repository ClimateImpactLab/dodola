"""Test application services
"""

import numpy as np
import xarray as xr
from dodola.services import bias_correct
from dodola.repository import FakeRepository


def _datafactory(x, start_time="1950-01-01"):
    """Populate xr.Dataset with synthetic data for testing"""
    start_time = str(start_time)
    if x.ndim != 1:
        raise ValueError("'x' needs dim of one")

    out = xr.Dataset(
        {"fakevariable": (["lon", "lat", "time"], x[np.newaxis, np.newaxis, :])},
        coords={
            "lon": (["lon"], [1.0]),
            "lat": (["lat"], [1.0]),
            "time": xr.cftime_range(
                start=start_time, freq="D", periods=len(x), calendar="noleap"
            ),
        },
    )
    return out


# def test_bias_correct_basic_call():
#     """Simple integration test of bias_correct service"""
#     # Setup input data.
#     n_years = 5
#     n = n_years * 365  # need daily data...

#     # Our "biased model".
#     model_bias = 10
#     x_train = _datafactory(np.arange(0, n) + model_bias)
#     # True "observations".
#     y_train = _datafactory(np.arange(0, n))
#     # Yes, we're testing and training on the same data...
#     x_test = x_train.copy(deep=True)

#     output_key = "test_output"
#     training_model_key = "x_train"
#     training_obs_key = "y_train"
#     forecast_model_key = "x_test"

#     # Load up a fake repo with our input data in the place of big data and cloud
#     # storage.
#     fakestorage = FakeRepository(
#         {
#             training_model_key: x_train,
#             training_obs_key: y_train,
#             forecast_model_key: x_test,
#         }
#     )

#     bias_correct(
#         forecast_model_key,
#         x_train=training_model_key,
#         y_train=training_obs_key,
#         out=output_key,
#         storage=fakestorage,
#     )

#     # Our testing model forecast is identical to our training model data so model
#     # forecast should equal obsvations we tuned to.
#     xr.testing.assert_equal(fakestorage.storage[output_key], y_train)
