"""Core logic for bias-correction and downscaling

Math stuff and business logic goes here. This is the "business logic".
"""

from skdownscale.pointwise_models import PointWiseDownscaler, BcsdTemperature
from xclim import sdba

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
    method : str
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
        raise ValueError("this method is not yet supported")
    ds_predicted = predicted.to_dataset(name=out_variable)
    return ds_predicted
