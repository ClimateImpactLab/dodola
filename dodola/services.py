"""Used by the CLI or any UI to deliver services to our lovely users
"""
from functools import wraps
import json
import logging
import dask
from dodola.core import (
    xesmf_regrid,
    standardize_gcm,
    xclim_remove_leapdays,
    apply_wet_day_frequency_correction,
    train_quantiledeltamapping,
    adjust_quantiledeltamapping,
    train_analogdownscaling,
    adjust_analogdownscaling,
    dtr_floor,
    non_polar_dtr_ceiling,
    apply_precip_ceiling,
    xclim_units_any2pint,
    xclim_units_pint2cf,
    test_for_nans,
    test_variable_names,
    test_timesteps,
    test_temp_range,
    test_dtr_range,
    test_negative_values,
    test_maximum_precip,
)
import dodola.repository as storage

logger = logging.getLogger(__name__)


def log_service(func):
    """Decorator for dodola.services to log service start and stop"""

    @wraps(func)
    def service_logger(*args, **kwargs):
        servicename = func.__name__
        logger.info(f"Starting dodola service {servicename} with {args=}, {kwargs=})")
        func(*args, **kwargs)
        logger.info(f"dodola service {servicename} done")

    return service_logger


@log_service
def prime_qdm_output_zarrstore(
    simulation,
    variable,
    years,
    out,
    zarr_region_dims,
    root_attrs_json_file=None,
    new_attrs=None,
):
    """Init a Zarr Store for writing QDM output regionally in independent processes.

    Parameters
    ----------
    simulation : str
        fsspec-compatible URL containing simulation data to be adjusted.
    variable : str
        Target variable in `simulation` to adjust. Adjusted output will share the
        same name.
    years : sequence of ints
        Years of simulation to adjust, with rolling years and day grouping.
    out : str
        fsspec-compatible path or URL pointing to Zarr Store file where the
        QDM-adjusted simulation data will be written.
    zarr_region_dims: sequence of str
        Sequence giving the name of dimensions that will be used to later write
        to regions of the Zarr Store. Variables with dimensions that do not use
        these regional variables will be appended to the primed Zarr Store as
        part of this call.
    root_attrs_json_file : str or None, optional
        fsspec-compatible URL pointing to a JSON file to use as root ``attrs``
        for the output data. ``new_attrs`` will be appended to this.
    new_attrs : dict or None, optional
        dict to merge with output Dataset's root ``attrs`` before output.
    """
    # TODO: Options to change primed output zarr store chunking?
    import xarray as xr  # TODO: Clean up this import or move the import-depending code to doodla.core

    quantile_variable_name = "sim_q"
    sim_df = storage.read(simulation)

    if root_attrs_json_file:
        logger.info(f"Using root attrs from {root_attrs_json_file}")
        sim_df.attrs = storage.read_attrs(root_attrs_json_file)

    # Yes, the time slice needs to use strs, not ints. It's already going to be inclusive so don't need to +1.
    primer = sim_df.sel(time=slice(str(min(years)), str(max(years))))

    ## This is where chunking happens... not sure about whether this is needed or how to effectively handle this.
    # primed_out = dodola.repository.read(simulation_zarr).sel(time=timeslice).chunk({"time": 73, "lat": 10, "lon":180})

    primer[quantile_variable_name] = xr.zeros_like(primer[variable])
    # Analysts said sim_q needed no attrs.
    primer[quantile_variable_name].attrs = {}

    if new_attrs:
        primer.attrs |= new_attrs

    # Logic below might be better off in dodola.repository.
    logger.debug(f"Priming Zarr Store with {primer=}")
    primer.to_zarr(
        out,
        mode="w",
        compute=False,
        consolidated=True,
        safe_chunks=False,
    )
    logger.info(f"Written primer to {out}")

    # Append variables that do not depend on dims we're using to define the
    # region we'll later write to in the Zarr Store.
    variables_to_append = []
    for variable_name, variable in primer.variables.items():
        if any(
            region_variable not in variable.dims for region_variable in zarr_region_dims
        ):
            variables_to_append.append(variable_name)

    if variables_to_append:
        logger.info(f"Appending {variables_to_append} to primed Zarr Store")
        primer[variables_to_append].to_zarr(
            out, mode="a", compute=True, consolidated=True, safe_chunks=False
        )
        logger.info(f"Appended non-regional variables to {out}")
    else:
        logger.info("No non-regional variables to append to Zarr Store")


@log_service
def prime_qplad_output_zarrstore(
    simulation,
    variable,
    out,
    zarr_region_dims,
    root_attrs_json_file=None,
    new_attrs=None,
):
    """Init a Zarr Store for writing QPLAD output regionally in independent processes.

    Parameters
    ----------
    simulation : str
        fsspec-compatible URL containing simulation data to be adjusted.
    variable : str
        Target variable in `simulation` to adjust. Adjusted output will share the
        same name.
    out : str
        fsspec-compatible path or URL pointing to Zarr Store file where the
        QPLAD-adjusted simulation data will be written.
    zarr_region_dims: sequence of str
        Sequence giving the name of dimensions that will be used to later write
        to regions of the Zarr Store. Variables with dimensions that do not use
        these regional variables will be appended to the primed Zarr Store as
        part of this call.
    root_attrs_json_file : str or None, optional
        fsspec-compatible URL pointing to a JSON file to use as root ``attrs``
        for the output data. ``new_attrs`` will be appended to this.
    new_attrs : dict or None, optional
        dict to merge with output Dataset's root ``attrs`` before output.
    """
    sim_df = storage.read(simulation)

    if root_attrs_json_file:
        logger.info(f"Using root attrs from {root_attrs_json_file}")
        sim_df.attrs = storage.read_attrs(root_attrs_json_file)

    primer = sim_df[[variable]]
    # Ensure we get root attrs. Not sure explicit copy is still required.
    primer.attrs = sim_df.attrs.copy()

    if new_attrs:
        primer.attrs |= new_attrs

    # Logic below might be better off in dodola.repository.
    logger.debug(f"Priming Zarr Store with {primer=}")
    primer.to_zarr(out, mode="w", compute=False, consolidated=True)
    logger.info(f"Written primer to {out}")

    # Append variables that do not depend on dims we're using to define the
    # region we'll later write to in the Zarr Store.
    variables_to_append = []
    for variable_name, variable in primer.variables.items():
        if any(
            region_variable not in variable.dims for region_variable in zarr_region_dims
        ):
            variables_to_append.append(variable_name)

    if variables_to_append:
        logger.info(f"Appending {variables_to_append} to primed Zarr Store")
        primer[variables_to_append].to_zarr(
            out, mode="a", compute=True, consolidated=True, safe_chunks=False
        )
        logger.info(f"Appended non-regional variables to {out}")
    else:
        logger.info("No non-regional variables to append to Zarr Store")


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
        logger.info(f"Slicing by {sel_slice=}")
        hist = hist.sel(sel_slice)
        ref = ref.sel(sel_slice)

    if isel_slice:
        logger.info(f"Slicing by {isel_slice=}")
        hist = hist.isel(isel_slice)
        ref = ref.isel(isel_slice)

    ref = xclim_units_any2pint(ref, variable)
    hist = xclim_units_any2pint(hist, variable)

    qdm = train_quantiledeltamapping(
        reference=ref, historical=hist, variable=variable, kind=k
    )

    storage.write(out, qdm.ds)


@log_service
def apply_qdm(
    simulation,
    qdm,
    years,
    variable,
    out,
    sel_slice=None,
    isel_slice=None,
    out_zarr_region=None,
    root_attrs_json_file=None,
    new_attrs=None,
):
    """Apply trained QDM to adjust a years in a simulation, write to Zarr Store.

    Output includes bias-corrected variable `variable` as well as a variable giving quantiles
    from the QDM, "sim_q".

    Parameters
    ----------
    simulation : str
        fsspec-compatible URL containing simulation data to be adjusted.
    qdm : str
        fsspec-compatible URL pointing to Zarr Store containing canned
        ``xclim.sdba.adjustment.QuantileDeltaMapping`` Dataset.
    years : sequence of ints
        Years of simulation to adjust, with rolling years and day grouping.
    variable : str
        Target variable in `simulation` to adjust. Adjusted output will share the
        same name.
    out : str
        fsspec-compatible path or URL pointing to Zarr Store file where the
        QDM-adjusted simulation data will be written.
    sel_slice: dict or None, optional
        Label-index slice input slimulation dataset before adjusting.
        A mapping of {variable_name: slice(...)} passed to
        `xarray.Dataset.sel()`.
    isel_slice: dict or None, optional
        Integer-index slice input slimulation dataset before adjusting. A mapping
        of {variable_name: slice(...)} passed to `xarray.Dataset.isel()`.
    out_zarr_region: dict or None, optional
        A mapping of {variable_name: slice(...)} giving the region to write
        to if outputting to existing Zarr Store.
    root_attrs_json_file : str or None, optional
        fsspec-compatible URL pointing to a JSON file to use as root ``attrs``
        for the output data. ``new_attrs`` will be appended to this.
    new_attrs : dict or None, optional
        dict to merge with output Dataset's root ``attrs`` before output.
    """
    sim_ds = storage.read(simulation)
    qdm_ds = storage.read(qdm)

    if root_attrs_json_file:
        logger.info(f"Using root attrs from {root_attrs_json_file}")
        sim_ds.attrs = storage.read_attrs(root_attrs_json_file)

    if sel_slice:
        logger.info(f"Slicing by {sel_slice=}")
        sim_ds = sim_ds.sel(sel_slice)

    if isel_slice:
        logger.info(f"Slicing by {isel_slice=}")
        sim_ds = sim_ds.isel(isel_slice)

    variable = str(variable)

    qdm_ds.load()
    sim_ds.load()

    sim_ds = xclim_units_any2pint(sim_ds, variable)

    adjusted_ds = adjust_quantiledeltamapping(
        simulation=sim_ds,
        variable=variable,
        qdm=qdm_ds,
        years=years,
        astype=sim_ds[variable].dtype,
        include_quantiles=True,
    )

    adjusted_ds = xclim_units_pint2cf(adjusted_ds, variable)

    if new_attrs:
        adjusted_ds.attrs |= new_attrs

    storage.write(out, adjusted_ds, region=out_zarr_region)


@log_service
def train_qplad(
    coarse_reference,
    fine_reference,
    out,
    variable,
    kind,
    sel_slice=None,
    isel_slice=None,
):
    """Train Quantile-Preserving, Localized Analogs Downscaling and dump to `out`

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
        Kind of QPLAD downscaling.
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
        logger.info(f"Slicing by {sel_slice=}")
        ref_coarse = ref_coarse.sel(sel_slice)
        ref_fine = ref_fine.sel(sel_slice)

    if isel_slice:
        logger.info(f"Slicing by {isel_slice=}")
        ref_coarse = ref_coarse.isel(isel_slice)
        ref_fine = ref_fine.isel(isel_slice)

    # needs to not be chunked
    ref_coarse.load()
    ref_fine.load()

    ref_coarse = xclim_units_any2pint(ref_coarse, variable)
    ref_fine = xclim_units_any2pint(ref_fine, variable)

    qplad = train_analogdownscaling(
        coarse_reference=ref_coarse,
        fine_reference=ref_fine,
        variable=variable,
        kind=k,
    )

    storage.write(out, qplad.ds)


@log_service
def apply_qplad(
    simulation,
    qplad,
    variable,
    out,
    sel_slice=None,
    isel_slice=None,
    out_zarr_region=None,
    root_attrs_json_file=None,
    new_attrs=None,
    wet_day_post_correction=False,
):
    """Apply QPLAD adjustment factors to downscale a simulation, dump to NetCDF.

    Dumping to NetCDF is a feature likely to change in the near future.

    Parameters
    ----------
    simulation : str
        fsspec-compatible URL containing simulation data to be adjusted.
        Dataset must have `variable` as well as a variable, "sim_q", giving
        the quantiles from QDM bias-correction.
    qplad : str
        fsspec-compatible URL pointing to Zarr Store containing canned
        ``xclim.sdba.adjustment.AnalogQuantilePreservingDownscaling`` Dataset.
    variable : str
        Target variable in `simulation` to downscale. Downscaled output will share the
        same name.
    out : str
        fsspec-compatible path or URL pointing to Zarr Store where the
        QPLAD-downscaled simulation data will be written.
    sel_slice: dict or None, optional
        Label-index slice input slimulation dataset before adjusting.
        A mapping of {variable_name: slice(...)} passed to
        `xarray.Dataset.sel()`.
    isel_slice: dict or None, optional
        Integer-index slice input slimulation dataset before adjusting. A mapping
        of {variable_name: slice(...)} passed to `xarray.Dataset.isel()`.
    out_zarr_region: dict or None, optional
        A mapping of {variable_name: slice(...)} giving the region to write
        to if outputting to existing Zarr Store.
    root_attrs_json_file : str or None, optional
        fsspec-compatible URL pointing to a JSON file to use as root ``attrs``
        for the output data. ``new_attrs`` will be appended to this.
    new_attrs : dict or None, optional
        dict to merge with output Dataset's root ``attrs`` before output.
    wet_day_post_correction : bool
        Whether to apply wet day frequency correction on downscaled data
    """
    sim_ds = storage.read(simulation)
    qplad_ds = storage.read(qplad)

    if root_attrs_json_file:
        logger.info(f"Using root attrs from {root_attrs_json_file}")
        sim_ds.attrs = storage.read_attrs(root_attrs_json_file)

    if sel_slice:
        logger.info(f"Slicing by {sel_slice=}")
        sim_ds = sim_ds.sel(sel_slice)

    if isel_slice:
        logger.info(f"Slicing by {isel_slice=}")
        sim_ds = sim_ds.isel(isel_slice)

    sim_ds = sim_ds.set_coords(["sim_q"])

    # needs to not be chunked
    sim_ds = sim_ds.load()
    qplad_ds = qplad_ds.load()

    variable = str(variable)

    sim_ds = xclim_units_any2pint(sim_ds, variable)
    for var in qplad_ds:
        qplad_ds = xclim_units_any2pint(qplad_ds, var)

    adjusted_ds = adjust_analogdownscaling(
        simulation=sim_ds, qplad=qplad_ds, variable=variable
    )

    # no need to revert units change. adjust_analogdownscaling does it

    if wet_day_post_correction:
        adjusted_ds = apply_wet_day_frequency_correction(adjusted_ds, "post")

    if new_attrs:
        adjusted_ds.attrs |= new_attrs

    storage.write(out, adjusted_ds, region=out_zarr_region)


def get_attrs(x, variable=None):
    """Get JSON str of `x` attrs metadata."""
    d = storage.read(x)
    if variable:
        d = d[variable]
    return json.dumps(d.attrs)


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

    This loads the entire `x` Dataset into memory for speed
    and to avoid chunking errors.

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
def correct_wet_day_frequency(x, out, process, variable="pr"):
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
    variable: str
    """
    ds = storage.read(x)
    ds_corrected = apply_wet_day_frequency_correction(
        ds, process=process, variable=variable
    )
    storage.write(out, ds_corrected)


@log_service
def apply_dtr_floor(x, out, floor=1.0):
    """Applies a floor to diurnal temperature range (DTR) values

    This constrains the values in a DTR dataset by applying a floor. The floor is assigned to the value of the
    data points which have their value strictly below the floor.

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be corrected.
    out : str
        Storage URL to write corrected output to.
    floor : int or float, optional
        All DTR values lower than this value are corrected to that value.
    """
    ds = storage.read(x)
    ds = dtr_floor(ds, floor)
    storage.write(out, ds)


@log_service
def apply_non_polar_dtr_ceiling(x, out, ceiling=70.0):
    """Applies a ceiling to diurnal temperature range (DTR) values

    This constrains the values in a DTR dataset by applying a ceiling. The ceiling is assigned to the value of the
    data points which have their value strictly above the ceiling.

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be corrected.
    out : str
        Storage URL to write corrected output to.
    ceiling : int or float, optional
        All DTR values above this value are corrected to that value.
    """
    ds = storage.read(x)
    ds = non_polar_dtr_ceiling(ds, ceiling)
    storage.write(out, ds)


@log_service
def adjust_maximum_precipitation(x, out, threshold=3000.0):
    """Adjusts maximum precipitation in a dataset

    Parameters
    ----------
    x : str
        Storage URL to input xr.Dataset that will be corrected.
    out : str
        Storage URL to write corrected output to.
    threshold : int or float, optional
        All precipitation values lower than this value are corrected to the threshold value.
    """

    ds = storage.read(x)
    ds_corrected = apply_precip_ceiling(ds, threshold)
    storage.write(out, ds_corrected)


@log_service
def validate(x, var, data_type, time_period):
    """Performs validation on an input dataset

    Valid for CMIP6, bias corrected and downscaled. Raises AssertionError when
    validation fails.

    This function performs more memory-intensive tests by reading input data
    and subsetting to each year in the "time" dimension.

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
    # This is pretty rough but works to communicate the idea.
    # Consider having failed tests raise something like ValidationError rather
    # than AssertionErrors.
    ds = storage.read(x)

    # These only read in Zarr Store metadata -- not memory intensive.
    test_variable_names(ds, var)
    test_timesteps(ds, data_type, time_period)

    # Other test are done on annual selections with dask.delayed to
    # avoid large memory errors.
    # Doing all this here because this involves storage and I/O logic.
    @dask.delayed
    def memory_intensive_tests(f, v, t):
        d = storage.read(f).sel(time=str(t))

        test_for_nans(d, v)

        if v == "tasmin":
            test_temp_range(d, v)
        elif v == "tasmax":
            test_temp_range(d, v)
        elif v == "dtr":
            test_dtr_range(d, v, data_type)
            test_negative_values(d, v)
        elif v == "pr":
            test_negative_values(d, v)
            test_maximum_precip(d, v)
        else:
            raise ValueError(f"Argument {v=} not recognized")

        # Assumes error thrown if had problem before this.
        return True

    tasks = []
    for t in set(ds["time"].dt.year.data):
        logger.debug(f"Validating year {t}")
        tasks.append(memory_intensive_tests(x, var, t))
    tasks = dask.compute(*tasks)
    assert all(tasks)  # Likely don't need this
