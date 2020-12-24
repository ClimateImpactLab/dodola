"""Core logic for bias-correction and downscaling

Math stuff and business logic goes here. This is the "business logic".
"""

from skdownscale.pointwise_models import PointWiseDownscaler, BcsdTemperature

# Break this down into a submodule(s) if needed.
# Assume data input here is generally clean and valid.


def bias_correct_bcsd(
    gcm_training_ds, obs_training_ds, gcm_predict_ds, train_variable, out_variable
):

    """Bias correct input model data using BCSD method,
       using either monthly or +/- 15 day time grouping.

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
    ds_predicted : Dataset
        bias corrected future model data
    """

    # note that time_grouper='daily_nasa-nex' is what runs the
    # NASA-NEX version of daily BCSD
    # TO-DO: switch to NASA-NEX version once tests pass
    model = PointWiseDownscaler(BcsdTemperature(return_anoms=False))
    model.fit(gcm_training_ds[train_variable], obs_training_ds[train_variable])
    predicted = model.predict(gcm_predict_ds[train_variable]).load()
    ds_predicted = predicted.to_dataset(name=out_variable)
    return ds_predicted
