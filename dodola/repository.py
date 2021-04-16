"""Objects to read and write stored climate model data.
"""

import logging
from fsspec import get_mapper as fs_get_mapper
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


def get_mapper(root, check=False, create=False):
    """Get fsspec.FSMap from wrapped FileSystemAbstraction

    Parameters
    ----------
    root : str
    check : bool
        Do touch at storage to check for write access.
    create : bool

    Returns
    -------
    out : fsspec.FSMap
    """
    return fs_get_mapper(root, check=check, create=create)
