"""Used by the CLI or any UI to deliver services to our lovely users
"""
import logging
from dodola.core import bias_correct_bcsd, build_xesmf_weights_file


logger = logging.getLogger(__name__)


def bias_correct(x, x_train, train_variable, y_train, out, out_variable, storage):
    """Bias correct input model data with IO to storage

    Parameters
    ----------
    x : str
        Storage URL to input data to bias correct.
    x_train : str
        Storage URL to input biased data to use for training bias-correction
        model.
    train_variable : str
        Variable name used in training and obs data.
    y_train : str
        Storage URL to input 'true' data or observations to use for training
        bias-correction model.
    out : str
        Storage URL to write bias-corrected output to.
    out_variable : str
        Variable name used as output variable name.
    storage : RepositoryABC-like
        Storage abstraction for data IO.
    """
    logger.info("Correcting bias")
    gcm_training_ds = storage.read(x_train)
    obs_training_ds = storage.read(y_train)
    gcm_predict_ds = storage.read(x)

    # This is all made up demo. Just get the output dataset the user expects.
    bias_corrected_ds = bias_correct_bcsd(
        gcm_training_ds, obs_training_ds, gcm_predict_ds, train_variable, out_variable
    )

    storage.write(out, bias_corrected_ds)
    logger.info("Bias corrected")


def build_weights(x, method, storage, target_resolution=1.0, outpath=None):
    """Generate local NetCDF weights file for regridding climate data

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be regridded.
    method : str
        Method of regridding. Passed to ``xesmf.Regridder``.
    target_resolution : float, optional
        Decimal-degree resolution of global grid to regrid to.
    storage : RepositoryABC-like
        Storage abstraction for data IO.
    outpath : optional
        Local file path name to write regridding weights file to.
    """
    logger.info("Building weights")
    ds = storage.read(x)
    build_xesmf_weights_file(
        ds, method=method, target_resolution=target_resolution, filename=outpath
    )
    logger.info("Weights built")


def disaggregate(x, weights, out, repo):
    """This is just an example. Please replace or delete."""
    raise NotImplementedError
