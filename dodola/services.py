"""Used by the CLI or any UI to deliver services to our lovely users
"""
from functools import wraps
import json
import logging
import fsspec
from dodola.core import (
    apply_bias_correction,
    build_xesmf_weights_file,
    xesmf_regrid,
    standardize_gcm,
    xclim_remove_leapdays,
    apply_downscaling,
    qdm_rollingyearwindow,
    train_quantiledeltamapping,
    adjust_quantiledeltamapping_year,
)
import dodola.repository as storage

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
def train_qdm(historical, reference, out, variable, kind):
    """Train quantile delta mapping and dump to `out`

    Parameters
    ----------
    historical : str
        fsspec-compatible URL to historical simulation store.
    reference : str
        fsspec-compatible URL to store to use as model reference.
    out : str
        fsspec-compatible URL to store trained model.
    variable : str
        Name of target variable in input and output stores.
    kind : {"additive", "multiplicative"}
        Kind of QDM scaling.
    """
    hist = storage.read(historical)
    ref = storage.read(reference)

    kind_map = {"additive": "+", "multiplicative": "*"}
    if kind not in kind_map.keys():
        # So we get a helpful exception message showing accepted kwargs...
        ValueError(f"kind must be {set(kind_map.keys())}, got {kind}")

    qdm = train_quantiledeltamapping(
        historical=hist, reference=ref, variable=variable, kind=kind_map[kind]
    )

    storage.write(out, qdm.ds)


@log_service
def apply_qdm(simulation, qdm, year, variable, out):
    """Apply trained QDM to adjust a year within a simulation, dump to NetCDF.

    Dumping to NetCDF is a feature likely to change in the near future.

    Parameters
    ----------
    simulation : str
        fsspec-compatible URL containing simulation data to be adjusted.
    qdm : str
        fsspec-compatible URL pointing to Zarr Store containing canned
        ``xclim.sdba.adjustment.QuantileDeltaMapping`` Dataset.
    year : int
        Target year to adjust, with rolling years and day grouping.
    variable : str
        Target variable in `sim` to adjust. Adjusted output will share the
        same name.
    out : str
        fsspec-compatible path or URL pointing to NetCDF4 file where the
        QDM-adjusted simulation data will be written.
    """
    sim_df = storage.read(simulation)
    qdm_df = storage.read(qdm)

    year = int(year)
    variable = str(variable)

    adjusted_ds = adjust_quantiledeltamapping_year(
        simulation=sim_df, qdm=qdm_df, year=year, variable=variable
    )

    # Write to NetCDF, usually on local disk, pooling and "fanning-in" NetCDFs is
    # currently faster and more reliable than Zarr Stores. This logic is handled
    # in workflow and cloud artifact repository.
    logger.debug(f"Writing to {out}")
    adjusted_ds.to_netcdf(out, compute=True)
    logger.info(f"Written {out}")


@log_service
def find_qdm_rollingyearwindow(x, out):
    """Write JSON of first and last years for QDM of x with rolling yearly window

    Parameters
    ----------
    x : str
        fsspec-compliant URL to climate data. Must have a CF-compliant time
        dimension.
    out : str
        fsspec-compliant URL to write JSON information to. The output file is a
        mapping of str to ints: {'firstyear': x, 'lastyear': y}.
    """
    ds = storage.read(x)
    firstyear, lastyear = qdm_rollingyearwindow(ds)

    logger.debug(f"Writing to {out}")
    with fsspec.open(out, mode="w") as fl:
        json.dump({"firstyear": firstyear, "lastyear": lastyear}, fl)
    logger.info(f"Written {out}")


@log_service
def bias_correct(x, x_train, train_variable, y_train, out, out_variable, method):
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
    method : str
        Bias correction method to be used.
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
    x,
    y_climo_coarse,
    y_climo_fine,
    out,
    train_variable,
    out_variable,
    method,
    domain_file,
    adjustmentfactors=None,
    weights_path=None,
):
    """Downscale bias corrected model data with IO to storage

    Parameters
    ----------
    x : str
        Storage URL to bias corrected input data to downscale.
    y_climo_coarse : str
        Storage URL to input coarse-res obs climatology to use for computing adjustment factors.
    y_climo_fine : str
        Storage URL to input fine-res obs climatology to use for computing adjustment factors.
    out : str
        Storage URL to write downscaled output to.
    adjustmentfactors : str or None, optional
        Storage URL to write fine-resolution adjustment factors to.
    train_variable : str
        Variable name used in training and obs data.
    out_variable : str
        Variable name used as output variable name.
    method : {"BCSD"}
        Downscaling method to be used.
    domain_file : str
        Storage URL to input grid for regridding adjustment factors
    adjustmentfactors : str, optional
        Storage URL to write fine-resolution adjustment factors to.
    weights_path : str or None, optional
        Storage URL for input weights for regridding
    """
    bc_ds = storage.read(x)
    obs_climo_coarse = storage.read(y_climo_coarse)
    obs_climo_fine = storage.read(y_climo_fine)
    domain_fine = storage.read(domain_file)

    adjustment_factors, downscaled_ds = apply_downscaling(
        bc_ds,
        obs_climo_coarse=obs_climo_coarse,
        obs_climo_fine=obs_climo_fine,
        train_variable=train_variable,
        out_variable=out_variable,
        method=method,
        domain_fine=domain_fine,
        weights_path=weights_path,
    )

    storage.write(out, downscaled_ds)
    if adjustmentfactors is not None:
        storage.write(adjustmentfactors, adjustment_factors)


@log_service
def build_weights(x, method, domain_file, outpath=None):
    """Generate local NetCDF weights file for regridding climate data

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be regridded.
    method : str
        Method of regridding. Passed to ``xesmf.Regridder``.
    domain_file : str
        Storage URL to input xr.Dataset domain file to regrid to.
    outpath : optional
        Local file path name to write regridding weights file to.
    """
    ds = storage.read(x)

    ds_domain = storage.read(domain_file)

    build_xesmf_weights_file(ds, ds_domain, method=method, filename=outpath)


@log_service
def rechunk(x, target_chunks, out):
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
def regrid(x, out, method, domain_file, weights_path=None):
    """Regrid climate data

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be regridded.
    out : str
        Storage URL to write regridded output to.
    method : str
        Method of regridding. Passed to ``xesmf.Regridder``.
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
def clean_cmip6(x, out, leapday_removal):
    """Cleans and standardizes CMIP6 GCM

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be cleaned.
    out : str
        Storage URL to write cleaned GCM output to.
    leapday_removal : bool
        Whether or not to remove leap days.
    """
    ds = storage.read(x)
    cleaned_ds = standardize_gcm(ds, leapday_removal)
    storage.write(out, cleaned_ds)


@log_service
def remove_leapdays(x, out):
    """Removes leap days and updates calendar attribute

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be regridded.
    out : str
        Storage URL to write regridded output to.
    """
    ds = storage.read(x)
    noleap_ds = xclim_remove_leapdays(ds)
    storage.write(out, noleap_ds)


@log_service
def disaggregate(x, weights, out, repo):
    """This is just an example. Please replace or delete."""
    raise NotImplementedError
