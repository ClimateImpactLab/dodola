"""Core logic for bias-correction and downscaling

Math stuff and business logic goes here. This is the "business logic".
"""

import os
from tempfile import TemporaryDirectory

from rechunker import rechunk
from skdownscale.pointwise_models import PointWiseDownscaler, BcsdTemperature
import xarray as xr
from xclim import sdba
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
        model.train(gcm_training_ds[train_variable], obs_training_ds[train_variable])
        predicted = model.adjust(gcm_predict_ds[train_variable])
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


def rechunk_ds(ds, target_chunks, max_mem):
    """Quickly (re)chunk a Dataset to specification

    Note, this rechunks the input into a Zarr store on the local disk. Be sure
    disk volume is available for this operation.

    Parameters
    ----------
    ds : xr.Dataset
    target_chunks : dict
        A dict of dicts. Top-level dict key maps variables name in `ds` to an
        inner dict {coordinate_name: chunk_size} mapping showing how data is
        to be rechunked.
    max_mem : int or str
        Maximum memory to use for rechunking (bytes).

    Returns
    -------
    out : xr.Dataset
    """
    max_mem = str(max_mem)  # To work around bug in rechunker.
    # Using tempdir for isolation/cleanup as rechunker dumps zarr files to disk.
    with TemporaryDirectory() as tmpdir:
        rechunkedzarr_path = os.path.join(tmpdir, "rechunk_out.zarr")
        tmpzarr_path = os.path.join(tmpdir, "rechunk_tmp.zarr")
        plan = rechunk(
            ds,
            target_chunks=target_chunks,
            target_store=rechunkedzarr_path,
            temp_store=tmpzarr_path,
            max_mem=max_mem,
        )
        plan.execute()
        out = xr.open_zarr(rechunkedzarr_path).load()
    return out
