"""Core logic for bias-correction and downscaling

Math stuff and business logic goes here. This is the "business logic".
"""


import numpy as np
import logging
from skdownscale.spatial_models import SpatialDisaggregator
import xarray as xr
from xclim import sdba
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
        Dataset to use as model reference.
    historical : xr.Dataset
        Dataset to use as historical simulation.
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
    qdm = sdba.adjustment.QuantileDeltaMapping(
        kind=str(kind),
        group=sdba.Grouper("time.dayofyear", window=int(window_n)),
        nquantiles=int(quantiles_n),
    )
    qdm.train(ref=reference[variable], hist=historical[variable])
    return qdm


def adjust_quantiledeltamapping_year(
    simulation, qdm, year, variable, halfyearwindow_n=10
):
    """Apply QDM to adjust a year within a simulation.

    Parameters
    ----------
    simulation : xr.Dataset
        Daily simulation data to be adjusted. Must have sufficient observations
        around `year` to adjust.
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
    out = qdm.adjust(simulation, interp="nearest").sel(time=str(year))

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
        model = sdba.adjustment.QuantileDeltaMapping(group=group, kind="+")
        model.train(
            ref=obs_training_ds[train_variable], hist=gcm_training_ds[train_variable]
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
        A model dataset that has been downscaled from the bias correction resolution to specified domain file resolution.
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


def xesmf_regrid(x, domain, method, weights_path=None, astype=None):
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
    astype : str, numpy.dtype, or None, optional
        Typecode or data-type to which the regridded output is cast.

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
    if astype:
        return regridder(x).astype(astype)
    return regridder(x)


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
    [1] A.J. Cannon, S.R. Sobie, & T.Q. Murdock, "Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes?", Journal of Climate, vol. 28, Issue 7, pp. 6938-6959.
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
