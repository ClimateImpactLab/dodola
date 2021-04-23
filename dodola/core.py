"""Core logic for bias-correction and downscaling

Math stuff and business logic goes here. This is the "business logic".
"""

from skdownscale.pointwise_models import PointWiseDownscaler, BcsdTemperature
from skdownscale.spatial_models import SpatialDisaggregator, apply_weights
import xarray as xr
from xclim import sdba
from xclim.core.calendar import convert_calendar
import xesmf as xe

# Break this down into a submodule(s) if needed.
# Assume data input here is generally clean and valid.


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
    ds_predicted : Dataset
        bias corrected future model data
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
        bias corrected model data
    obs_climo_coarse : Dataset
        observation climatologies at coarse resolution
    obs_climo_fine : Dataset
        observation climatologies at fine resolution
    train_variable : str
        variable name used in obs data
    out_variable : str
        variable name used in downscaled output
    method : {"BCSD"}
        method to be used in the applied downscaling
    domain_fine : Dataset
        domain that specifies the fine resolution grid to downscale to
    weights_path : str, optional
        path to the weights file if they will be used in regridding,
        weights for downscaling to fine resolution
    ds_downscaled : Dataset
        downscaled model data
    adjustment_factors : Dataset
        adjustment factors produced in downscaling
    """

    if method == "BCSD":
        model = SpatialDisaggregator(var=train_variable)
        af_coarse = model.fit(bc_ds, obs_climo_coarse, var_name=train_variable)

        # regrid adjustment factors
        # BCSD uses bilinear interpolation for both temperature and precip to
        # regrid adjustment factors
        af_fine = xesmf_regrid(af_coarse, domain_fine, "bilinear", weights_path)

        # apply adjustment factors
        predicted = model.predict(af_fine, obs_climo_fine, var_name=train_variable)
    else:
        raise ValueError("this method is not supported")

    ds_downscaled = predicted.to_dataset(name=out_variable)
    return af_fine, ds_downscaled


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
