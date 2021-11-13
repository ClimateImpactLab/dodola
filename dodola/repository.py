"""Objects to read and write stored climate model data.
"""

import json
import logging
import fsspec
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
    logger.debug(f"Reading {url_or_path}")
    x = open_zarr(url_or_path)
    logger.info(f"Read {url_or_path}")
    return x


def write(url_or_path, x, region=None):
    """Write Dataset to Zarr Store

    Note, any lazy computations will be evaluated.

    Parameters
    ----------
    url_or_path : str
        Location to write Zarr store to.
    x : xr.Dataset
    region : dict or None, optional
        Optional mapping from dimension names to integer slices along dataset
        dimensions to indicate the region of existing zarr array(s) in
        which to write this dataset’s data. Variables not sliced in the region
        are dropped.
    """
    logger.debug(f"Writing {url_or_path}")
    logger.debug(f"Output Dataset {x=}")

    if region:
        # TODO: This behavior needs a better, focused, unit test.
        logger.info(f"Writing to Zarr Store region, {region=}")

        # We need to drop all variables not sliced by the selected zarr_region.
        variables_to_drop = []
        region_variables = list(region.keys())
        for variable_name, variable in x.variables.items():
            if any(
                region_variable not in variable.dims
                for region_variable in region_variables
            ):
                variables_to_drop.append(variable_name)

        logger.info(
            f"Dropping variables before Zarr region write: {variables_to_drop=}"
        )
        x = x.drop_vars(variables_to_drop)

        x.to_zarr(url_or_path, region=region, mode="a", compute=True)
    else:
        x.to_zarr(url_or_path, mode="w", compute=True)
    logger.info(f"Written {url_or_path}")


def read_attrs(urlpath):
    """Read and deserialize JSON attrs file"""
    logger.debug(f"Reading attrs from {urlpath}")

    with fsspec.open(urlpath) as f:
        out = json.load(f)
        logger.info(f"Read attrs from {urlpath}")

    logger.debug(f"Read attrs {out}")
    return out
