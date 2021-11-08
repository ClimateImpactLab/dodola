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
    apply_downscaling,
    apply_wet_day_frequency_correction,
    train_quantiledeltamapping,
    adjust_quantiledeltamapping_year,
    train_analogdownscaling,
    adjust_analogdownscaling,
    validate_dataset,
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
def train_qdm(
    historical, reference, out, variable, kind, sel_slice=None, isel_slice=None
):
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
    sel_slice: dict or None, optional
        Label-index slice hist and ref to subset before training.
        A mapping of {variable_name: slice(...)} passed to
        `xarray.Dataset.sel()`.
    isel_slice: dict or None, optional
        Integer-index slice hist and ref to subset before training. A mapping
        of {variable_name: slice(...)} passed to `xarray.Dataset.isel()`.
    """
    hist = storage.read(historical)
    ref = storage.read(reference)

    kind_map = {"additive": "+", "multiplicative": "*"}
    try:
        k = kind_map[kind]
    except KeyError:
        # So we get a helpful exception message showing accepted kwargs...
        raise ValueError(f"kind must be {set(kind_map.keys())}, got {kind}")

    if sel_slice:
        logger.debug(f"Slicing by {sel_slice=}")
        hist = hist.sel(sel_slice)
        ref = ref.sel(sel_slice)

    if isel_slice:
        logger.debug(f"Slicing by {isel_slice=}")
        hist = hist.isel(isel_slice)
        ref = ref.isel(isel_slice)

    qdm = train_quantiledeltamapping(
        reference=ref, historical=hist, variable=variable, kind=k
    )

    storage.write(out, qdm.ds)


@log_service
def apply_qdm(simulation, qdm, year, variable, out, include_quantiles=False):
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
        Target variable in `simulation` to adjust. Adjusted output will share the
        same name.
    out : str
        fsspec-compatible path or URL pointing to NetCDF4 file where the
        QDM-adjusted simulation data will be written.
    include_quantiles : bool
        Flag to indicate whether bias-corrected quantiles should be
        included in the QDM-adjusted output.
    """
    sim_ds = storage.read(simulation)
    qdm_ds = storage.read(qdm)

    year = int(year)
    variable = str(variable)

    adjusted_ds = adjust_quantiledeltamapping_year(
        simulation=sim_ds,
        qdm=qdm_ds,
        year=year,
        variable=variable,
        include_quantiles=include_quantiles,
    )

    # Write to NetCDF, usually on local disk, pooling and "fanning-in" NetCDFs is
    # currently faster and more reliable than Zarr Stores. This logic is handled
    # in workflow and cloud artifact repository.
    logger.debug(f"Writing to {out}")
    adjusted_ds.to_netcdf(out, compute=True)
    logger.info(f"Written {out}")


@log_service
def train_aiqpd(
    coarse_reference,
    fine_reference,
    out,
    variable,
    kind,
    sel_slice=None,
    isel_slice=None,
):
    """Train analog-inspired quantile preserving downscaling and dump to `out`

    Parameters
    ----------
    coarse_reference : str
        fsspec-compatible URL to resampled coarse reference store.
    fine_reference : str
        fsspec-compatible URL to fine-resolution reference store.
    out : str
        fsspec-compatible URL to store adjustment factors.
    variable : str
        Name of target variable in input and output stores.
    kind : {"additive", "multiplicative"}
        Kind of AIQPD downscaling.
    sel_slice: dict or None, optional
        Label-index slice hist and ref to subset before training.
        A mapping of {variable_name: slice(...)} passed to
        `xarray.Dataset.sel()`.
    isel_slice: dict or None, optional
        Integer-index slice hist and ref to subset before training. A mapping
        of {variable_name: slice(...)} passed to `xarray.Dataset.isel()`.
    """
    ref_coarse = storage.read(coarse_reference)
    ref_fine = storage.read(fine_reference)

    kind_map = {"additive": "+", "multiplicative": "*"}
    try:
        k = kind_map[kind]
    except KeyError:
        # So we get a helpful exception message showing accepted kwargs...
        raise ValueError(f"kind must be {set(kind_map.keys())}, got {kind}")

    if sel_slice:
        logger.debug(f"Slicing by {sel_slice=}")
        ref_coarse = ref_coarse.sel(sel_slice)
        ref_fine = ref_fine.sel(sel_slice)

    if isel_slice:
        logger.debug(f"Slicing by {isel_slice=}")
        ref_coarse = ref_coarse.isel(isel_slice)
        ref_fine = ref_fine.isel(isel_slice)

    # needs to not be chunked
    ref_coarse.load()
    ref_fine.load()

    aiqpd = train_analogdownscaling(
        coarse_reference=ref_coarse,
        fine_reference=ref_fine,
        variable=variable,
        kind=k,
    )

    storage.write(out, aiqpd.ds)


@log_service
def apply_aiqpd(simulation, aiqpd, variable, out):
    """Apply AIQPD adjustment factors to downscale a simulation, dump to NetCDF.

    Dumping to NetCDF is a feature likely to change in the near future.

    Parameters
    ----------
    simulation : str
        fsspec-compatible URL containing simulation data to be adjusted.
    aiqpd : str
        fsspec-compatible URL pointing to Zarr Store containing canned
        ``xclim.sdba.adjustment.AnalogQuantilePreservingDownscaling`` Dataset.
    variable : str
        Target variable in `simulation` to downscale. Downscaled output will share the
        same name.
    out : str
        fsspec-compatible path or URL pointing to NetCDF4 file where the
        AIQPD-downscaled simulation data will be written.
    """
    sim_ds = storage.read(simulation)
    aiqpd_ds = storage.read(aiqpd)

    # needs to not be chunked
    sim_ds = sim_ds.load()
    aiqpd_ds = aiqpd_ds.load()

    variable = str(variable)

    downscaled_ds = adjust_analogdownscaling(
        simulation=sim_ds, aiqpd=aiqpd_ds, variable=variable
    )

    # Write to NetCDF, usually on local disk, pooling and "fanning-in" NetCDFs is
    # currently faster and more reliable than Zarr Stores. This logic is handled
    # in workflow and cloud artifact repository.
    logger.debug(f"Writing to {out}")
    downscaled_ds.to_netcdf(out, compute=True, engine="netcdf4")
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
def regrid(
    x, out, method, domain_file, weights_path=None, astype=None, add_cyclic=None
):
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
    astype : str, numpy.dtype, or None, optional
        Typecode or data-type to which the regridded output is cast.
    add_cyclic : str, or None, optional
        Add cyclic (aka wrap-around values) to dimension before regridding.
         Useful for avoiding dateline artifacts along longitude in global
         datasets.
    """
    ds = storage.read(x)
    ds_domain = storage.read(domain_file)

    regridded_ds = xesmf_regrid(
        ds,
        ds_domain,
        method=method,
        weights_path=weights_path,
        astype=astype,
        add_cyclic=add_cyclic,
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
def correct_wet_day_frequency(x, out, process):
    """Corrects wet day frequency in a dataset

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be regridded.
    out : str
        Storage URL to write regridded output to.
    process : {"pre", "post"}
        Step in pipeline, used in determining how to correct.
        "Pre" replaces all zero values with a uniform random value below a threshold (before bias correction).
        "Post" replaces all values below a threshold with zeroes (after bias correction).
    """
    ds = storage.read(x)
    ds_corrected = apply_wet_day_frequency_correction(ds, process=process)
    storage.write(out, ds_corrected)


@log_service
def validate(x, var, data_type, time_period):
    """Performs validation on an input dataset

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be validated.
    var : {"tasmax", "tasmin", "dtr", "pr"}
        Variable in xr.Dataset that should be validated.
        Some validation functions are specific to each variable.
    data_type : {"cmip6", "bias_corrected", "downscaled"}
        Step in pipeline, used in determining how to validate.
    time_period: {"historical", "future"}
        Time period that input data should cover, used in validating the number of timesteps
        in conjunction with the data type.
    """

    ds = storage.read(x)
    validate_dataset(ds, var, data_type, time_period)
