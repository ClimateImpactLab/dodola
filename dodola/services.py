"""Used by the CLI or any UI to deliver services to our lovely users
"""

from dodola.core import bias_correct_bcsd
from dodola.repository import GcsRepository


def bias_correct(x, x_train, y_train, out, storage):
    """Bias correct input model data with IO to storage

    Parameters
    ----------
    x : str
        Storage URL to input data to bias correct.
    x_train : str
        Storage URL to input biased data to use for training bias-correction
        model.
    y_train : str
        Storage URL to input 'true' data or observations to use for training
        bias-correction model.
    out : str
        Storage URL to write bias-corrected output to.
    storage : RepositoryABC-like
        Storage abstraction for data IO.
    """
    gcm_training_ds = storage.read(x_train)
    obs_training_ds = storage.read(y_train)
    gcm_predict_ds = storage.read(x)

    # This is all made up demo. Just get the output dataset the user expects.
    bias_corrected_ds = bias_correct_bcsd(
        gcm_training_ds, obs_training_ds, gcm_predict_ds
    )

    storage.write(out, bias_corrected_ds)


def generate_weights(x, out, repo):
    """This is just an example. Please replace or delete."""
    raise NotImplementedError


def disaggregate(x, weights, out, repo):
    """This is just an example. Please replace or delete."""
    raise NotImplementedError
