"""Core logic for bias-correction and downscaling

Math stuff and business logic goes here. This is the "business logic".
"""


import warnings
import logging
import dask
import numpy as np
import xarray as xr
from xclim import sdba, set_options
from xclim.sdba.utils import equally_spaced_nodes
from xclim.core import units as xclim_units
from xclim.core.calendar import convert_calendar, get_calendar
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


def adjust_quantiledeltamapping(
    simulation,
    variable,
    qdm,
    years,
    astype=None,
    quantile_variable="sim_q",
    **kwargs,
):
    """Apply QDM to adjust a range of years within a simulation.

    Parameters
    ----------
    simulation : xr.Dataset
        Daily simulation data to be adjusted. Must have sufficient observations
        around `year` to adjust. Target variable must have a units attribute.
    variable : str
        Target variable in `simulation` to adjust. Adjusted output will
        share the same name.
    qdm : xr.Dataset or sdba.adjustment.QuantileDeltaMapping
        Trained ``xclim.sdba.adjustment.QuantileDeltaMapping``, or Dataset
        representation that will be instantiate
        ``xclim.sdba.adjustment.QuantileDeltaMapping``.
    years : sequence of ints
        Years of simulation to adjust, with rolling years and day grouping.
    astype : str, numpy.dtype, or None, optional
        Typecode or data-type to which the regridded output is cast.
    quantile_variable : str or None, optional
        Name of quantile coordinate to reset to data variable. Not reset
        if ``None``.
    kwargs :
        Keyword arguments passed to
        ``dodola.core.adjust_quantiledeltamapping_year``.

    Returns
    -------
    out : xr.Dataset
        QDM-adjusted values from `simulation`. May be a lazy-evaluated future, not
        yet computed. In addition to adjusted original variables, this includes
        "sim_q" variable giving quantiles from QDM biascorrection.
    """
    # This loop is a candidate for dask.delayed. Beware, xclim had issues with saturated scheduler.
    qdm_list = []
    for yr in years:
        adj = adjust_quantiledeltamapping_year(
            simulation=simulation, qdm=qdm, year=yr, variable=variable, **kwargs
        )
        if astype:
            adj = adj.astype(astype)
        qdm_list.append(adj)

    # Combine years and ensure output matches input data dimension order.
    adjusted_ds = xr.concat(qdm_list, dim="time").transpose(*simulation[variable].dims)

    if quantile_variable:
        adjusted_ds = adjusted_ds.reset_coords(quantile_variable)
        # Analysts said sim_q needed no attrs.
        adjusted_ds[quantile_variable].attrs = {}

    # Overwrite QDM output attrs with input simulation attrs.
    adjusted_ds.attrs = simulation.attrs
    for k, v in simulation.variables.items():
        if k in adjusted_ds:
            adjusted_ds[k].attrs = v.attrs

    return adjusted_ds


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
    """Train Quantile-Preserving, Localized Analogs Downscaling (QPLAD)

    Parameters
    ----------
    coarse_reference : xr.Dataset
        Dataset to use as resampled (to fine resolution) coarse reference.Target variable must have a units attribute.
    fine_reference : xr.Dataset
        Dataset to use as fine-resolution reference. Target variable must have a units attribute.
    variable : str
        Name of target variable to extract from `coarse_reference` and `fine_reference`.
    kind : {"+", "*"}
        Kind of variable. Used for creating QPLAD adjustment factors.
    quantiles_n : int, optional
        Number of quantiles for QPLAD.
    window_n : int, optional
        Centered window size for day-of-year grouping.

    Returns
    -------
    xclim.sdba.adjustment.QuantilePreservingAnalogDownscaling
    """

    # QPLAD method requires that the number of quantiles equals
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

    qplad = sdba.adjustment.QuantilePreservingAnalogDownscaling.train(
        ref=coarse_reference[variable],
        hist=fine_reference[variable],
        kind=str(kind),
        group=sdba.Grouper("time.dayofyear", window=int(window_n)),
        nquantiles=quantiles_n,
    )
    return qplad


def adjust_analogdownscaling(simulation, qplad, variable):
    """Apply QPLAD to downscale bias corrected output.

    Parameters
    ----------
    simulation : xr.Dataset
        Daily bias corrected data to be downscaled. Target variable must have a units attribute.
    qplad : xr.Dataset or sdba.adjustment.QuantilePreservingAnalogDownscaling
        Trained ``xclim.sdba.adjustment.QuantilePreservingAnalogDownscaling``, or
        Dataset representation that will instantiate
        ``xclim.sdba.adjustment.QuantilePreservingAnalogDownscaling``.
    variable : str
        Target variable in `simulation` to downscale. Downscaled output will share the
        same name.

    Returns
    -------
    out : xr.Dataset
        QPLAD-downscaled values from `simulation`. May be a lazy-evaluated future, not
        yet computed.
    """
    variable = str(variable)

    if isinstance(qplad, xr.Dataset):
        qplad = sdba.adjustment.QuantilePreservingAnalogDownscaling.from_dataset(qplad)

    out = qplad.adjust(simulation[variable]).to_dataset(name=variable)

    out = out.transpose(*simulation[variable].dims)
    # Overwrite QPLAD output attrs with input simulation attrs.
    out.attrs = simulation.attrs
    for k, v in simulation.variables.items():
        if k in out:
            out[k].attrs = v.attrs

    return out


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

    360 calendar conversion requires that there are no chunks in
    the 'time' dimension of `ds`.

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

    cal = get_calendar(ds)

    if (
        cal == "360_day" or leapday_removal
    ):  # calendar conversion is necessary in either case
        # if calendar is just integers, xclim cannot understand it
        if ds.time.dtype == "int64":
            ds_cleaned["time"] = xr.decode_cf(ds_cleaned).time
        if cal == "360_day":
            if leapday_removal:  # 360 day -> noleap
                ds_converted = xclim_convert_360day_calendar_interpolate(
                    ds=ds, target="noleap", align_on="random", interpolation="linear"
                )
            else:  # 360 day -> standard
                ds_converted = xclim_convert_360day_calendar_interpolate(
                    ds=ds, target="standard", align_on="random", interpolation="linear"
                )
        else:  # any -> noleap
            # remove leap days and update calendar
            ds_converted = xclim_remove_leapdays(ds_cleaned)

        # rechunk, otherwise chunks are different sizes
        ds_out = ds_converted.chunk(
            {"time": 730, "lat": len(ds.lat), "lon": len(ds.lon)}
        )

    else:
        ds_out = ds_cleaned

    return ds_out


def xclim_units_any2pint(ds, var):
    """
    Parameters
    ----------
    ds : xr.Dataset
    var : str

    Returns
    -------
    xr.Dataset with `var` units str attribute converted to xclim's pint registry format
    """

    logger.info(f"Reformatting {var} unit string representation")
    ds[var].attrs["units"] = str(xclim_units.units2pint(ds[var].attrs["units"]))
    return ds


def xclim_units_pint2cf(ds, var):
    """
    Parameters
    ----------
    ds : xr.Dataset
    var : str

    Returns
    -------
    xr.Dataset with `var` units str attribute converted to CF format
    """
    logger.info(f"Reformatting {var} unit string representation")
    ds[var].attrs["units"] = xclim_units.pint2cfunits(
        xclim_units.units2pint(ds[var].attrs["units"])
    )
    return ds


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


def xclim_convert_360day_calendar_interpolate(
    ds,
    target="noleap",
    align_on="random",
    interpolation="linear",
    return_indices=False,
    ignore_nans=True,
):
    """
    Parameters
    ----------
    ds : xr.Dataset
    target : str
        see xclim.core.calendar.convert_calendar
    align_on : str
        this determines which days in the calendar will have missing values or will be the product of interpolation, if there is.
        It could be every year the same calendar days, or the days could randomly change. see xclim.core.calendar.convert_calendar
    interpolation : None or str
        passed to xr.Dataset.interpolate_na if not None
    return_indices : bool
        on top of the converted dataset, return a list of the array indices identifying values that were inserted.
        This assumes there were no NaNs before conversion.
    ignore_nans : bool
        if False and there are any NaNs in `ds` variables, an assertion error will be raised. NaNs are ignored otherwise.
    Returns
    -------
    tuple(xr.Dataset, xr.Dataset) if return_indices is True, xr.Dataset otherwise.

    Notes
    -----
    The default values of `target`, `align_on` and `interpolation` mean that our default approach is equivalent to that of the LOCA
    calendar conversion [1] for conversion from 360 days calendars to noleap calendars. In that approach, 5 calendar days are added (noleap
    calendars always have 365 days) to each year. But those calendar days are not necessarily those that will have their value be the product
    of interpolation. The days for which we interpolate are selected randomly every block of 72 days, so that they change every year.

    [1] http://loca.ucsd.edu/loca-calendar/
    """

    if get_calendar(ds) != "360_day":
        raise ValueError(
            "tried to use 360 day calendar conversion for a non-360-day calendar dataset"
        )

    if not ignore_nans:
        for var in ds:
            assert (
                ds[var].isnull().sum() == 0
            ), "360 days calendar conversion with interpolation : there are nans !"

    ds_converted = convert_calendar(
        ds, target=target, align_on=align_on, missing=np.NaN
    )

    if interpolation:
        ds_out = ds_converted.interpolate_na("time", interpolation)
    else:
        ds_out = ds_converted

    if return_indices:
        return (ds_out, xr.ufuncs.isnan(ds_converted))
    else:
        return ds_out


def apply_wet_day_frequency_correction(ds, process, var="pr"):
    """

    Parameters
    ----------
    ds : xr.Dataset
    process : {"pre", "post"}
    var: str

    Returns
    -------
    xr.Dataset

    Notes
    -------
    [1] A.J. Cannon, S.R. Sobie, and T.Q. Murdock (2015), "Bias correction of GCM
        precipitation by quantile mapping: How well do methods preserve
        changes in quantiles and extremes?", Journal of Climate, vol.
        28, Issue 7, pp. 6938-6959.
    [2] S. Hempel, K. Frieler, L. Warszawski, J. Schewe, and F. Piotek (2013), "A trend-preserving bias correction - The ISI-MIP approach", Earth Syst. Dynam. vol. 4, pp. 219-236.
    """
    # threshold from Hempel et al 2013
    threshold = 1.0  # mm/day
    # adjusted "low" value from the original epsilon in Cannon et al 2015 to
    # avoid having some values get extremely large
    low = threshold / 2.0

    if process == "pre":
        # includes very small values that are negative in CMIP6 output
        ds[var] = ds[var].where(
            ds[var] >= threshold,
            np.random.uniform(low=low, high=threshold, size=ds[var].shape),
        )
    elif process == "post":
        ds[var] = ds[var].where(ds[var] >= threshold, 0.0)
    else:
        raise ValueError("this processing option is not implemented")
    return ds


def dtr_floor(ds, floor):
    """
    Converts all diurnal temperature range (DTR) values strictly below a floor
    to that floor.

    Parameters
    ----------
    ds : xr.Dataset
    floor : int or float

    Returns
    -------
    xr.Dataset

    """

    ds_corrected = ds.where(ds >= floor, floor)
    return ds_corrected


def non_polar_dtr_ceiling(ds, ceiling):
    """
    Converts all non-polar (regions between the 60th south and north parallel) diurnal temperature range (DTR) values strictly above a ceiling
    to that ceiling.

    Parameters
    ----------
    ds : xr.Dataset
    ceiling : int or float

    Returns
    -------
    xr.Dataset

    """

    ds_corrected = ds.where(
        xr.ufuncs.logical_or(
            ds <= ceiling, xr.ufuncs.logical_or(ds["lat"] <= -60, ds["lat"] >= 60)
        ),
        ceiling,
    )

    return ds_corrected


def apply_precip_ceiling(ds, ceiling):
    """
    Converts all precip values above a threshold to the threshold value, uniformly across space and time.

    Parameters
    ----------
    ds : xr.Dataset
    ceiling : int or float

    Returns
    -------
    xr.Dataset

    """
    ds_corrected = ds.where(ds <= ceiling, ceiling)
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
            _test_dtr_range(d, v, data_type)
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
    Tests that Dataset contains the correct number of timesteps (number of days on a noleap calendar)
    for the data_type/time_period combination.
    """
    if time_period == "future":
        # bias corrected/downscaled data has 2015 - 2099 or 2100 depending on the model
        # CMIP6 future data has an additional ten years from the historical model run
        if data_type == "cmip6":
            assert (
                len(ds.time) >= 35040  # some CMIP6 data ends in 2099
            ), "projection {} file is missing timesteps, has {}".format(
                data_type, len(ds.time)
            )
            if len(ds.time) > 35405:
                warnings.warn(
                    "projection {} file has excess timesteps, has {}".format(
                        data_type, len(ds.time)
                    )
                )
        else:
            assert (
                len(ds.time) >= 31025  # 2015 - 2099
            ), "projection {} file is missing timesteps, has {}".format(
                data_type, len(ds.time)
            )
            if len(ds.time) > 31390:  # 2015 - 2100
                warnings.warn(
                    "projection {} file has excess timesteps, has {}".format(
                        data_type, len(ds.time)
                    )
                )

    elif time_period == "historical":
        # bias corrected/downscaled data should have 1950 - 2014
        # CMIP6 historical data has an additional ten years from SSP 370 (or 245 if 370 not available)
        if data_type == "cmip6":
            assert (
                len(ds.time) >= 27740
            ), "historical {} file is missing timesteps, has {}".format(
                data_type, len(ds.time)
            )
            if len(ds.time) > 27740:
                warnings.warn(
                    "historical {} file has excess timesteps, has {}".format(
                        data_type, len(ds.time)
                    )
                )
        else:
            assert (
                len(ds.time) >= 23725
            ), "historical {} file is missing timesteps, has {}".format(
                data_type, len(ds.time)
            )
            if len(ds.time) > 23725:
                warnings.warn(
                    "historical {} file has excess timesteps, has {}".format(
                        data_type, len(ds.time)
                    )
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
    assert (ds[var].min() > 130) and (
        ds[var].max() < 360
    ), "{} values are invalid".format(var)


def _test_dtr_range(ds, var, data_type):
    """
    Ensure DTR values are in a valid range
    Test polar values separately since some polar values can be much higher post-bias correction.
    """
    # test that DTR values are greater than 0 or equal to 0 depending on the data type
    # note that some CMIP6 DTR will equal 0 at polar latitudes, this will be adjusted
    # before bias correction with the DTR small values correction

    dtr_min = ds[var].min()
    if data_type == "cmip6":
        # may be equal to zero in polar regions and if tasmax < tasmin (only occurs for GFDL models)
        assert (
            dtr_min >= 0
        ), "diurnal temperature range minimum is {} and thus not greater than or equal to 0 for CMIP6".format(
            dtr_min
        )
    else:
        # this must be greater than 0 for bias corrected and downscaled
        assert (
            dtr_min > 0
        ), "diurnal temperature range minimum is {} and must be greater than zero".format(
            dtr_min
        )

    # test polar DTR values
    southern_polar_max = ds[var].where(ds.lat < -60).max()
    if (southern_polar_max is not None) and (southern_polar_max >= 100):
        assert (
            southern_polar_max < 100
        ), "diurnal temperature range max is {} for polar southern latitudes".format(
            southern_polar_max
        )

    northern_polar_max = ds[var].where(ds.lat > 60).max()
    if (northern_polar_max is not None) and (northern_polar_max >= 100):
        assert (
            northern_polar_max < 100
        ), "diurnal temperature range max is {} for polar northern latitudes".format(
            northern_polar_max
        )

    # test all but polar regions
    non_polar_max = ds[var].where((ds.lat > -60) & (ds.lat < 60)).max()
    assert (
        non_polar_max <= 70
    ), "diurnal temperature range max is {} for non-polar regions".format(non_polar_max)


def _test_negative_values(ds, var):
    """
    Tests for presence of negative values
    """
    # this is not set to 0 to deal with floating point error
    neg_values = ds[var].where(ds[var] < -0.001).count()
    assert neg_values == 0, "there are {} negative values!".format(neg_values)


def _test_maximum_precip(ds, var):
    """
    Tests that max precip is reasonable
    """
    threshold = 3000  # in mm, max observed is 1.825m --> maximum occurs between 0.5-0.8
    max_precip = ds[var].max().load().values
    num_precip_values_over_threshold = (
        ds[var].where(ds[var] > threshold).count().load().values
    )
    assert (
        num_precip_values_over_threshold == 0
    ), "maximum precip is {} mm and there are {} values over 3000mm".format(
        max_precip, num_precip_values_over_threshold
    )
