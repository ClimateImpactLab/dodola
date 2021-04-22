"""Used by the CLI or any UI to deliver services to our lovely users
"""
from functools import wraps
import logging
from dodola.core import (
    apply_bias_correction,
    build_xesmf_weights_file,
    xesmf_regrid,
    standardize_gcm,
    xclim_remove_leapdays,
)

logger = logging.getLogger(__name__)


def log_service(func):
    """Decorator for dodola.services to log service start and stop"""

    @wraps(func)
    def service_logger(*args, **kwargs):
        servicename = func.__name__
        logger.info(f"Starting {servicename} dodola service")
        func(*args, **kwargs)
        logger.info(f"dodola service {servicename} done")

    return service_logger


@log_service
def bias_correct(
    x, x_train, train_variable, y_train, out, out_variable, method, storage
):
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
    af : str
        Storage URL to write fine-resolution adjustment factors to. 
    out_variable : str
        Variable name used as output variable name.
    method : str
        Bias correction method to be used.
    storage : dodola.repository._ZarrRepo
        Storage abstraction for data IO.
    """
    gcm_training_ds = storage.read(x_train)
    obs_training_ds = storage.read(y_train)
    gcm_predict_ds = storage.read(x)

    # This is all made up demo. Just get the output dataset the user expects.
    bias_corrected_ds = apply_bias_correction(
        gcm_training_ds,
        obs_training_ds,
        gcm_predict_ds,
        train_variable,
        out_variable,
        method,
    )

    storage.write(out, bias_corrected_ds)


@log_service
def downscale(
    x, y_train, out, af, storage, train_variable, out_variable, method, 
):
    """Downscale bias corrected model data with IO to storage

    Parameters
    ----------
    x : str
        Storage URL to bias corrected input data to downscale.
    y_train : str
        Storage URL to input 'true' data or observations to use for downscaling.
    y_climo : str
        Storage URL to input obs climatology to use for computing adjustment factors. 
    out : str
        Storage URL to write downscaled output to.
    af : str, optional
        Storage URL to write fine-resolution adjustment factors to.
    storage : dodola.repository._ZarrRepo
        Storage abstraction for data IO.
    train_variable : str
        Variable name used in training and obs data.
    out_variable : str
        Variable name used as output variable name.
    method : str
        Downscaling method to be used.
    """
    bc_ds = storage.read(x)
    obs_ds = storage.read(y_train)
    obs_climo = storage.read(y_climo)

    adjustment_factors, downscaled_ds = apply_downscaling(
        bc_ds,
        obs_ds,
        obs_climo,
        train_variable,
        out_variable,
        method,
    )

    storage.write(out, downscaled_ds)
    storage.write(af, adjustment_factors)

@log_service
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
    storage : dodola.repository._ZarrRepo
        Storage abstraction for data IO.
    outpath : optional
        Local file path name to write regridding weights file to.
    """
    ds = storage.read(x)
    build_xesmf_weights_file(
        ds, method=method, target_resolution=target_resolution, filename=outpath
    )


@log_service
def rechunk(x, target_chunks, out, storage):
    """Rechunk data to specification

    Parameters
    ----------
    x : str
        Storage URL to input data.
    target_chunks : dict
        Mapping {coordinate_name: chunk_size} showing how data is
        to be rechunked.
    out : str
        Storage URL to write rechunked output to.
    storage : dodola.repository._ZarrRepo
        Storage abstraction for data IO.
    """
    ds = storage.read(x)

    # Simple, stable, but not for more specialized rechunking needs.
    # In that case use "rechunker" package, or similar.
    ds = ds.chunk(target_chunks)

    # Hack to get around issue with writing chunks to zarr in xarray ~v0.17.0
    # https://github.com/pydata/xarray/issues/2300
    for v in ds.data_vars.keys():
        del ds[v].encoding["chunks"]

    storage.write(out, ds)


@log_service
def regrid(x, out, method, storage, domain_file, weights_path=None):
    """Regrid climate data

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be regridded.
    out : str
        Storage URL to write regridded output to.
    method : str
        Method of regridding. Passed to ``xesmf.Regridder``.
    storage : dodola.repository._ZarrRepo
        Storage abstraction for data IO.
    domain_file : str
        Storage URL to input xr.Dataset domain file to regrid to.
    weights_path : optional
        Local file path name to write regridding weights file to.
    """
    ds = storage.read(x)

    ds_domain = storage.read(domain_file)

    regridded_ds = xesmf_regrid(
        ds,
        ds_domain,
        method=method,
        weights_path=weights_path,
    )

    storage.write(out, regridded_ds)


@log_service
def clean_cmip6(x, out, leapday_removal, storage):
    """Cleans and standardizes CMIP6 GCM

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be cleaned.
    out : str
        Storage URL to write cleaned GCM output to.
    leapday_removal : bool
        Whether or not to remove leap days.
    storage : dodola.repository._ZarrRepo
        Storage abstraction for data IO.
    """
    ds = storage.read(x)
    cleaned_ds = standardize_gcm(ds, leapday_removal)
    storage.write(out, cleaned_ds)


@log_service
def remove_leapdays(x, out, storage):
    """Removes leap days and updates calendar attribute

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be regridded.
    out : str
        Storage URL to write regridded output to.
    storage : dodola.repository._ZarrRepo
        Storage abstraction for data IO.
    """
    ds = storage.read(x)
    noleap_ds = xclim_remove_leapdays(ds)
    storage.write(out, noleap_ds)


@log_service
def disaggregate(x, weights, out, repo):
    """This is just an example. Please replace or delete."""
    raise NotImplementedError
