"""Core logic for bias-correction and downscaling

Math stuff and business logic goes here. This is the "business logic".
"""


import logging
import dask
import numpy as np
from skdownscale.spatial_models import SpatialDisaggregator
import xarray as xr
from xclim import sdba, set_options
from xclim.sdba.utils import equally_spaced_nodes
from xclim.core.calendar import convert_calendar
import xesmf as xe

logger = logging.getLogger(__name__)

# Break this down into a submodule(s) if needed.
# Assume data input here is generally clean and valid.


def train_quantiledeltamapping(
    reference, historical, variable, kind, quantiles_n=100, window_n=31
):
    """Train quantile delta mapping

    Parameters
    ----------
    reference : xr.Dataset
        Dataset to use as model reference. Target variable must have a units attribute.
    historical : xr.Dataset
        Dataset to use as historical simulation. Target variable must have a units attribute.
    variable : str
        Name of target variable to extract from `historical` and `reference`.
    kind : {"+", "*"}
        Kind of variable. Used for QDM scaling.
    quantiles_n : int, optional
        Number of quantiles for QDM.
    window_n : int, optional
        Centered window size for day-of-year grouping.

    Returns
    -------
    xclim.sdba.adjustment.QuantileDeltaMapping
    """
    qdm = sdba.adjustment.QuantileDeltaMapping.train(
        ref=reference[variable],
        hist=historical[variable],
        kind=str(kind),
        group=sdba.Grouper("time.dayofyear", window=int(window_n)),
        nquantiles=equally_spaced_nodes(int(quantiles_n), eps=None),
    )
    return qdm


def adjust_quantiledeltamapping_year(
    simulation, qdm, year, variable, halfyearwindow_n=10, include_quantiles=False
):
    """Apply QDM to adjust a year within a simulation.

    Parameters
    ----------
    simulation : xr.Dataset
        Daily simulation data to be adjusted. Must have sufficient observations
        around `year` to adjust. Target variable must have a units attribute.
    qdm : xr.Dataset or sdba.adjustment.QuantileDeltaMapping
        Trained ``xclim.sdba.adjustment.QuantileDeltaMapping``, or
        Dataset representation that will be instantiate
        ``xclim.sdba.adjustment.QuantileDeltaMapping``.
    year : int
        Target year to adjust, with rolling years and day grouping.
    variable : str
        Target variable in `simulation` to adjust. Adjusted output will share the
        same name.
    halfyearwindow_n : int, optional
        Half-length of the annual rolling window to extract along either
        side of `year`.
    include_quantiles : bool, optional
        Whether or not to output quantiles (sim_q) as a coordinate on
        the bias corrected data variable in output.

    Returns
    -------
    out : xr.Dataset
        QDM-adjusted values from `simulation`. May be a lazy-evaluated future, not
        yet computed.
    """
    year = int(year)
    variable = str(variable)
    halfyearwindow_n = int(halfyearwindow_n)

    if isinstance(qdm, xr.Dataset):
        qdm = sdba.adjustment.QuantileDeltaMapping.from_dataset(qdm)

    # Slice to get 15 days before and after our target year. This accounts
    # for the rolling 31 day rolling window.
    timeslice = slice(
        f"{year - halfyearwindow_n - 1}-12-17", f"{year + halfyearwindow_n + 1}-01-15"
    )
    simulation = simulation[variable].sel(
        time=timeslice
    )  # TODO: Need a check to ensure we have all the data in this slice!
    if include_quantiles:
        # include quantile information in output
        with set_options(sdba_extra_output=True):
            out = qdm.adjust(simulation, interp="nearest").sel(time=str(year))
            # make quantiles a coordinate of bias corrected output variable
            out = out["scen"].assign_coords(sim_q=out.sim_q)
    else:
        out = qdm.adjust(simulation, interp="nearest").sel(time=str(year))

    return out.to_dataset(name=variable)


def train_analogdownscaling(
    coarse_reference, fine_reference, variable, kind, quantiles_n=620, window_n=31
):
    """Train analog-inspired quantile-preserving downscaling

    Parameters
    ----------
    coarse_reference : xr.Dataset
        Dataset to use as resampled (to fine resolution) coarse reference.Target variable must have a units attribute.
    fine_reference : xr.Dataset
        Dataset to use as fine-resolution reference. Target variable must have a units attribute.
    variable : str
        Name of target variable to extract from `coarse_reference` and `fine_reference`.
    kind : {"+", "*"}
        Kind of variable. Used for creating AIQPD adjustment factors.
    quantiles_n : int, optional
        Number of quantiles for AIQPD.
    window_n : int, optional
        Centered window size for day-of-year grouping.

    Returns
    -------
    xclim.sdba.adjustment.AnalogQuantilePreservingDownscaling
    """

    # AIQPD method requires that the number of quantiles equals
    # the number of days in each day group
    # e.g. 20 years of data and a window of 31 = 620 quantiles

    # check that lengths of input data are the same, then only check years for one
    if len(coarse_reference.time) != len(fine_reference.time):
        raise ValueError("coarse and fine reference data inputs have different lengths")

    # check number of years in input data (subtract 2 for the +/- 15 days on each end)
    num_years = len(np.unique(fine_reference.time.dt.year)) - 2
    if (num_years * int(window_n)) != quantiles_n:
        raise ValueError(
            "number of quantiles {} must equal # of years {} * window length {}, day groups must {} days".format(
                quantiles_n, num_years, int(window_n), quantiles_n
            )
        )

    aiqpd = sdba.adjustment.AnalogQuantilePreservingDownscaling.train(
        ref=coarse_reference[variable],
        hist=fine_reference[variable],
        kind=str(kind),
        group=sdba.Grouper("time.dayofyear", window=int(window_n)),
        nquantiles=quantiles_n,
    )
    return aiqpd


def adjust_analogdownscaling(simulation, aiqpd, variable):
    """Apply AIQPD to downscale bias corrected output.

    Parameters
    ----------
    simulation : xr.Dataset
        Daily bias corrected data to be downscaled. Target variable must have a units attribute.
    aiqpd : xr.Dataset or sdba.adjustment.AnalogQuantilePreservingDownscaling
        Trained ``xclim.sdba.adjustment.AnalogQuantilePreservingDownscaling``, or
        Dataset representation that will instantiate
        ``xclim.sdba.adjustment.AnalogQuantilePreservingDownscaling``.
    variable : str
        Target variable in `simulation` to downscale. Downscaled output will share the
        same name.

    Returns
    -------
    out : xr.Dataset
        AIQPD-downscaled values from `simulation`. May be a lazy-evaluated future, not
        yet computed.
    """
    variable = str(variable)

    if isinstance(aiqpd, xr.Dataset):
        aiqpd = sdba.adjustment.AnalogQuantilePreservingDownscaling.from_dataset(aiqpd)

    out = aiqpd.adjust(simulation[variable])

    return out.to_dataset(name=variable)


def apply_bias_correction(
    gcm_training_ds,
    obs_training_ds,
    gcm_predict_ds,
    train_variable,
    out_variable,
    method,
):

    """Bias correct input model data using specified method,
       using either monthly or +/- 15 day time grouping. Currently
       the QDM method is supported.

    Parameters
    ----------
    gcm_training_ds : Dataset
        training model data for building quantile map
    obs_training_ds : Dataset
        observation data for building quantile map
    gcm_predict_ds : Dataset
        future model data to be bias corrected
    train_variable : str
        variable name used in training data
    out_variable : str
        variable name used in downscaled output
    method : {"QDM"}
        method to be used in the applied bias correction

    Returns
    -------
    ds_predicted : xr.Dataset
        Dataset that has been bias corrected.
    """

    if method == "QDM":
        # instantiates a grouper class that groups by day of the year
        # centered window: +/-15 day group
        group = sdba.Grouper("time.dayofyear", window=31)
        model = sdba.adjustment.QuantileDeltaMapping.train(
            ref=obs_training_ds[train_variable],
            hist=gcm_training_ds[train_variable],
            group=group,
            kind="+",
        )
        predicted = model.adjust(sim=gcm_predict_ds[train_variable])
    else:
        raise ValueError("this method is not supported")
    ds_predicted = predicted.to_dataset(name=out_variable)
    return ds_predicted


def apply_downscaling(
    bc_ds,
    obs_climo_coarse,
    obs_climo_fine,
    train_variable,
    out_variable,
    method,
    domain_fine,
    weights_path=None,
):

    """Downscale input bias corrected data using specified method.
       Currently only the BCSD method for spatial disaggregation is
       supported.

    Parameters
    ----------
    bc_ds : Dataset
        Model data that has already been bias corrected.
    obs_climo_coarse : Dataset
        Observation climatologies at coarse resolution.
    obs_climo_fine : Dataset
        Observation climatologies at fine resolution.
    train_variable : str
        Variable name used in obs data.
    out_variable : str
        Variable name used in downscaled output.
    method : {"BCSD"}
        Vethod to be used in the applied downscaling.
    domain_fine : Dataset
        Domain that specifies the fine resolution grid to downscale to.
    weights_path : str or None, optional
        Path to the weights file, used for downscaling to fine resolution.

    Returns
    -------
    af_fine : xr.Dataset
        A dataset of adjustment factors at fine resolution used in downscaling.
    ds_downscaled : xr.Dataset
        A model dataset that has been downscaled from the bias correction
        resolution to specified domain file resolution.
    """

    if method == "BCSD":
        model = SpatialDisaggregator(var=train_variable)
        af_coarse = model.fit(bc_ds, obs_climo_coarse, var_name=train_variable)

        # regrid adjustment factors
        # BCSD uses bilinear interpolation for both temperature and precip to
        # regrid adjustment factors
        af_fine = xesmf_regrid(af_coarse, domain_fine, "bilinear", weights_path)

        # apply adjustment factors
        predicted = model.predict(
            af_fine, obs_climo_fine[train_variable], var_name=train_variable
        )
    else:
        raise ValueError("this method is not supported")

    ds_downscaled = predicted.to_dataset(name=out_variable)
    return af_fine, ds_downscaled


def build_xesmf_weights_file(x, domain, method, filename=None):
    """Build ESMF weights file for regridding x to a global grid

    Parameters
    ----------
    x : xr.Dataset
    domain : xr.Dataset
        Domain to regrid to.
    method : str
        Method of regridding. Passed to ``xesmf.Regridder``.
    filename : optional
        Local path to output netCDF weights file.

    Returns
    -------
    outfilename : str
        Path to resulting weights file.
    """
    out = xe.Regridder(
        x,
        domain,
        method=method,
        filename=filename,
    )
    return str(out.filename)


def _add_cyclic(ds, dim):
    """
    Adds wrap-around, appending first value to end of data for named dimension.

    Basically an xarray version of ``cartopy.util.add_cyclic_point()``.
    """
    return ds.map(
        lambda x, d: xr.concat([x, x.isel({d: 0})], dim=d),
        keep_attrs=True,
        d=str(dim),
    )


def xesmf_regrid(
    x, domain, method, weights_path=None, astype=None, add_cyclic=None, keep_attrs=True
):
    """
    Regrid a Dataset.

    Parameters
    ----------
    x : xr.Dataset
    domain : xr.Dataset
        Domain to regrid to.
    method : str
        Method of regridding. Passed to ``xesmf.Regridder``.
    weights_path : str, optional
        Local path to netCDF file of pre-calculated XESMF regridding weights.
    astype : str, numpy.dtype, or None, optional
        Typecode or data-type to which the regridded output is cast.
    add_cyclic : str, or None, optional
        Add cyclic point (aka wrap-around pixel) to given dimension before
        regridding. Useful for avoiding dateline artifacts along longitude
        in global datasets.
    keep_attrs : bool, optional
        Whether to pass attrs from input to regridded output.

    Returns
    -------
    xr.Dataset
    """
    if add_cyclic:
        x = _add_cyclic(x, add_cyclic)

    regridder = xe.Regridder(
        x,
        domain,
        method=method,
        filename=weights_path,
    )
    if astype:
        return regridder(x, keep_attrs=keep_attrs).astype(astype)
    return regridder(x, keep_attrs=keep_attrs)


def standardize_gcm(ds, leapday_removal=True):
    """

    Parameters
    ----------
    ds : xr.Dataset
    leapday_removal : bool, optional

    Returns
    -------
    xr.Dataset
    """
    # Remove cruft coordinates, variables, dims.
    cruft_vars = ("height", "member_id", "time_bnds")

    dims_to_squeeze = []
    coords_to_drop = []
    for v in cruft_vars:
        if v in ds.dims:
            dims_to_squeeze.append(v)
        elif v in ds.coords:
            coords_to_drop.append(v)

    ds_cleaned = ds.squeeze(dims_to_squeeze, drop=True).reset_coords(
        coords_to_drop, drop=True
    )

    # Cleanup time.

    # if variable is precip, need to update units to mm day-1
    if "pr" in ds_cleaned.variables:
        # units should be kg/m2/s in CMIP6 output
        if ds_cleaned["pr"].units == "kg m-2 s-1":
            # convert to mm/day
            mmday_conversion = 24 * 60 * 60
            ds_cleaned["pr"] = ds_cleaned["pr"] * mmday_conversion
            # update units attribute
            ds_cleaned["pr"].attrs["units"] = "mm day-1"
        else:
            # we want this to fail, as pr units are something we don't expect
            raise ValueError("check units: pr units attribute is not kg m-2 s-1")

    if leapday_removal:
        # if calendar is just integers, xclim cannot understand it
        if ds.time.dtype == "int64":
            ds_cleaned["time"] = xr.decode_cf(ds_cleaned).time
        # remove leap days and update calendar
        ds_noleap = xclim_remove_leapdays(ds_cleaned)

        # rechunk, otherwise chunks are different sizes
        ds_out = ds_noleap.chunk({"time": 730, "lat": len(ds.lat), "lon": len(ds.lon)})
    else:
        ds_out = ds_cleaned

    return ds_out


def xclim_remove_leapdays(ds):
    """

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    xr.Dataset
    """
    ds_noleap = convert_calendar(ds, target="noleap")
    return ds_noleap


def apply_wet_day_frequency_correction(ds, process):
    """

    Parameters
    ----------
    ds : xr.Dataset
    process : {"pre", "post"}

    Returns
    -------
    xr.Dataset

    Notes
    -------
    [1] A.J. Cannon, S.R. Sobie, & T.Q. Murdock, "Bias correction of GCM
        precipitation by quantile mapping: How well do methods preserve
        changes in quantiles and extremes?", Journal of Climate, vol.
        28, Issue 7, pp. 6938-6959.
    """
    threshold = 0.05  # mm/day
    low = 1e-16
    if process == "pre":
        ds_corrected = ds.where(ds != 0.0, np.random.uniform(low=low, high=threshold))
    elif process == "post":
        ds_corrected = ds.where(ds >= threshold, 0.0)
    else:
        raise ValueError("this processing option is not implemented")
    return ds_corrected


def validate_dataset(ds, var, data_type, time_period="future"):
    """
    Validate a Dataset. Valid for CMIP6, bias corrected and downscaled.

    Raises AssertionError when validation fails.

    Parameters
    ----------
    ds : xr.Dataset
    var : {"tasmax", "tasmin", "dtr", "pr"}
        Variable in Dataset to validate.
    data_type : {"cmip6", "bias_corrected", "downscaled"}
        Type of data output to validate.
    time_period : {"historical", "future"}
        Time period of data that will be validated.
    """
    # This is pretty rough but works to communicate the idea.
    # Consider having failed tests raise something like ValidationError rather
    # than AssertionErrors.

    # These only read in Zarr Store metadata -- not memory intensive.
    _test_variable_names(ds, var)
    _test_timesteps(ds, data_type, time_period)

    # Other test are done on annual selections with dask.delayed to
    # avoid large memory errors. xr.map_blocks had trouble with this.
    @dask.delayed
    def memory_intensive_tests(ds, v, t):
        d = ds.sel(time=str(t))

        _test_for_nans(d, v)

        if v == "tasmin":
            _test_temp_range(d, v)
        elif v == "tasmax":
            _test_temp_range(d, v)
        elif v == "dtr":
            _test_dtr_range(d, v)
            _test_negative_values(d, v)
        elif v == "pr":
            _test_negative_values(d, v)
            _test_maximum_precip(d, v)
        else:
            raise ValueError(f"Argument {v=} not recognized")

        # Assumes error thrown if had problem before this.
        return True

    results = []
    for t in np.unique(ds["time"].dt.year.data):
        logger.debug(f"Validating year {t}")
        results.append(memory_intensive_tests(ds, var, t))
    results = dask.compute(*results)
    assert all(results)  # Likely don't need this
    return True


def _test_for_nans(ds, var):
    """
    Tests for presence of NaNs
    """
    assert ds[var].isnull().sum() == 0, "there are nans!"


def _test_timesteps(ds, data_type, time_period):
    """
    Tests that Dataset contains the correct number of timesteps
    for the data_type/time_period combination.
    """
    if time_period == "future":
        # bias corrected/downscaled data should have 2015 - 2100
        # CMIP6 future data has an additional ten years from the historical model run
        if data_type == "cmip6":
            assert (
                len(ds.time) == 35405
            ), "projection {} file is missing timesteps, only has {}".format(
                data_type, len(ds.time)
            )
        else:
            assert (
                len(ds.time) == 31390
            ), "projection {} file is missing timesteps, only has {}".format(
                data_type, len(ds.time)
            )
    elif time_period == "historical":
        # bias corrected/downscaled data should have 1950 - 2014
        # CMIP6 historical data has an additional ten years from SSP 370 (or 245 if 370 not available)
        if data_type == "cmip6":
            assert (
                len(ds.time) == 27740
            ), "historical {} file is missing timesteps, only has {}".format(
                data_type, len(ds.time)
            )
        else:
            assert (
                len(ds.time) == 23725
            ), "historical {} file is missing timesteps, only has {}".format(
                data_type, len(ds.time)
            )


def _test_variable_names(ds, var):
    """
    Test that the correct variable name exists in the file
    """
    assert var in ds.var(), "{} not in Dataset".format(var)


def _test_temp_range(ds, var):
    """
    Ensure temperature values are in a valid range
    """
    assert (ds[var].min() > 150) and (
        ds[var].max() < 350
    ), "{} values are invalid".format(var)


def _test_dtr_range(ds, var):
    """
    Ensure DTR values are in a valid range
    """
    assert (ds[var].min() > 0) and (
        ds[var].max() < 45
    ), "diurnal temperature range values are invalid"


def _test_negative_values(ds, var):
    """
    Tests for presence of negative values
    """
    # this is not set to 0 to deal with floating point error
    assert ds[var].where(ds[var] < -0.001).count() == 0, "there are negative values!"


def _test_maximum_precip(ds, var):
    """
    Tests that max precip is reasonable
    """
    threshold = 2000  # in mm, max observed is 1.825m --> maximum occurs between 0.5-0.8
    assert (
        ds[var].where(ds[var] > threshold).count() == 0
    ), "maximum precip exceeds 2000mm"
