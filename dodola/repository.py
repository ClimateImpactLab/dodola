"""Objects to read and write stored climate model data.
"""

import logging
from xarray import open_zarr


logger = logging.getLogger(__name__)


def read(url_or_path):
    """Read Dataset from Zarr store

    Parameters
    ----------
    url_or_path : str
        Location of Zarr store to read.

    Returns
    -------
    xr.Dataset
    """
    x = open_zarr(url_or_path)
    logger.info(f"Read {url_or_path}")
    return x


def write(url_or_path, x):
    """Write Dataset to Zarr store

    This opens Zarr store with mode "w" and is called with with
    ``compute=True``, so any lazy computations will be completed.

    Parameters
    ----------
    url_or_path : str
        Location to write Zarr store to.
    x : xr.Dataset
    """
    x.to_zarr(url_or_path, mode="w", compute=True)
    logger.info(f"Written {url_or_path}")
