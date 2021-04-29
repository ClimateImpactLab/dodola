"""Core logic for bias-correction and downscaling

Math stuff and business logic goes here. This is the "business logic".
"""

from datetime import timedelta
import logging
import cftime
from skdownscale.pointwise_models import PointWiseDownscaler, BcsdTemperature
import xarray as xr
from xclim import sdba
from xclim.core.calendar import convert_calendar
import xesmf as xe

logger = logging.getLogger(__name__)

# Break this down into a submodule(s) if needed.
# Assume data input here is generally clean and valid.

def qdm_rollingyearwindow(ds, halfyearwindow_n=10):
    """Get the first and last years for QDM of ds with an rolling yearly window

    Parameters
    ----------
    ds : xr.Dataset
        Must have "time" variable, populated with cftime.Datetime.
    halfyearwindow_n : int, optional
        Half the length of the rolling year-window for QDM.

    Returns
    -------
    firstyear : int
    lastyear : int
    """
    earliest_dt = min(ds["time"].values)
    latest_dt = max(ds["time"].values)
    logger.debug(f"Dataset earliest datetime: {earliest_dt}")
    logger.debug(f"Dataset latest datetime: {latest_dt}")

    earliest_year = earliest_dt.year
    latest_year = latest_dt.year

    assert earliest_dt.calendar == latest_dt.calendar, "time values should have the same calendar"

    # Figure out how many years we need to offset for our 1) year window
    # and 2) 15 day window for the earliest end of the window...
    early_limit = cftime.datetime(year=earliest_year, month=12, day=15, calendar=earliest_dt.calendar)
    additional_offset = 1  # +1 yr because we need 15 days from end of the first year:
    if early_limit < earliest_dt:
        # Require additional year offset if we have less than 15 days from the
        # *first* year, at the end of the year...
        additional_offset += 1
    firstyear = int(earliest_year + halfyearwindow_n + additional_offset)

    # Same as above but to find offset on the latest end of window.
    late_limit = cftime.datetime(year=latest_year, month=1, day=15, calendar=latest_dt.calendar)
    additional_offset = 1  # -1 yr because we need 15 days from beginning of the last year:
    if latest_dt < late_limit:
        # Use additional year offset if we fewer than 15 days from the
        # at the start of the year in the *last* year...
        additional_offset += 1
    lastyear = int(latest_year - halfyearwindow_n - additional_offset)

    logger.info(f"QDM window first year: {firstyear}")
    logger.info(f"QDM window last year: {lastyear}")

    if firstyear > lastyear:
        raise ValueError("firstyear must be <= lastyear to have years for QDM window.")
    # Safety against spending lots and *lots* of time and money:
    if abs(firstyear - lastyear) > 200:
        # Maybe this should be a different exception? Maybe should be validation?
        raise ValueError("dif between firstyear and lastyear seems too large, error?")

    return firstyear, lastyear


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
       BCSD and QDM methods are supported.

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
    method : {"BCSD", "QDM"}
        method to be used in the applied bias correction
    ds_predicted : Dataset
        bias corrected future model data
    """

    if method == "BCSD":
        # note that time_grouper='daily_nasa-nex' is what runs the
        # NASA-NEX version of daily BCSD
        # TO-DO: switch to NASA-NEX version once tests pass
        model = PointWiseDownscaler(BcsdTemperature(return_anoms=False))
        model.fit(gcm_training_ds[train_variable], obs_training_ds[train_variable])
        predicted = model.predict(gcm_predict_ds[train_variable]).load()
    elif method == "QDM":
        # instantiates a grouper class that groups by day of the year
        # centered window: +/-15 day group
        group = sdba.Grouper("time.dayofyear", window=31)
        model = sdba.adjustment.QuantileDeltaMapping(group=group, kind="+")
        model.train(
            ref=obs_training_ds[train_variable], hist=gcm_training_ds[train_variable]
        )
        predicted = model.adjust(sim=gcm_predict_ds[train_variable])
    else:
        raise ValueError("this method is not supported")
    ds_predicted = predicted.to_dataset(name=out_variable)
    return ds_predicted


def build_xesmf_weights_file(x, method, target_resolution, filename=None):
    """Build ESMF weights file for regridding x to a global grid

    Parameters
    ----------
    x : xr.Dataset
    method : str
        Method of regridding. Passed to ``xesmf.Regridder``.
    target_resolution: float
        Decimal-degree resolution of global grid to regrid to.
    filename : optional
        Local path to output netCDF weights file.

    Returns
    -------
    outfilename : str
        Path to resulting weights file.
    """
    out = xe.Regridder(
        x,
        xe.util.grid_global(target_resolution, target_resolution),
        method=method,
        filename=filename,
    )
    return str(out.filename)


def xesmf_regrid(x, domain, method, weights_path=None):
    """

    Parameters
    ----------
    x : xr.Dataset
    domain : xr.Dataset
        Domain to regrid to.
    method : str
        Method of regridding. Passed to ``xesmf.Regridder``.
    weights_path : str, optional
        Local path to netCDF file of pre-calculated XESMF regridding weights.

    Returns
    -------
    xr.Dataset
    """

    regridder = xe.Regridder(
        x,
        domain,
        method=method,
        filename=weights_path,
    )
    return regridder(x)


def standardize_gcm(ds, leapday_removal=True):
    """

    Parameters
    ----------
    x : xr.Dataset
    leapday_removal : bool, optional

    Returns
    -------
    xr.Dataset
    """
    dims_to_drop = []
    if "height" in ds.dims:
        dims_to_drop.append("height")
    if "member_id" in ds.dims:
        dims_to_drop.append("member_id")
    if "time_bnds" in ds.dims:
        dims_to_drop.append("time_bnds")

    if "member_id" in ds.dims:
        ds_cleaned = ds.isel(member_id=0).drop(dims_to_drop)
    else:
        ds_cleaned = ds.drop(dims_to_drop)

    if leapday_removal:
        # if calendar is just integers, xclim cannot understand it
        if ds.time.dtype == "int64":
            ds_cleaned["time"] = xr.decode_cf(ds_cleaned).time
        # remove leap days and update calendar
        ds_noleap = xclim_remove_leapdays(ds_cleaned)

        # rechunk, otherwise chunks are different sizes
        ds_out = ds_noleap.chunk(730, len(ds.lat), len(ds.lon))
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
