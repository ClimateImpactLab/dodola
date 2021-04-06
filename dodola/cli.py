"""Commandline interface to the application.
"""

from os import getenv
import logging
import click
import dodola.services as services
from dodola.repository import adl_repository


logger = logging.getLogger(__name__)


def _authenticate_storage():
    storage = adl_repository(
        account_name=getenv("AZURE_STORAGE_ACCOUNT"),
        account_key=getenv("AZURE_STORAGE_KEY"),
        client_id=getenv("AZURE_CLIENT_ID"),
        client_secret=getenv("AZURE_CLIENT_SECRET"),
        tenant_id=getenv("AZURE_TENANT_ID"),
    )
    return storage


# Main entry point
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--debug/--no-debug", default=False, envvar="DODOLA_DEBUG")
def dodola_cli(debug):
    """GCM bias-correction and downscaling

    Authenticate with storage by setting the AZURE_STORAGE_ACCOUNT and
    AZURE_STORAGE_KEY environment variables for key-based authentication.
    Alternatively, set AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and
    AZURE_TENANT_ID for service principal-based authentication.
    """
    noisy_loggers = [
        "azure.core.pipeline.policies.http_logging_policy",
        "asyncio",
        "adlfs.spec",
        "chardet.universaldetector",
        "fsspec",
    ]
    for logger_name in noisy_loggers:
        nl = logging.getLogger(logger_name)
        nl.setLevel(logging.WARNING)

    loglevel = logging.INFO
    if debug:
        loglevel = logging.DEBUG

    logging.basicConfig(level=loglevel)


@dodola_cli.command(help="Bias-correct GCM on observations")
@click.argument("x", required=True)
@click.argument("xtrain", required=True)
@click.argument("trainvariable", required=True)
@click.argument("ytrain", required=True)
@click.argument("out", required=True)
@click.argument("outvariable", required=True)
@click.argument("method", required=True)
def biascorrect(x, xtrain, trainvariable, ytrain, out, outvariable, method):
    """Bias-correct GCM (x) to 'out' based on model (xtrain), obs (ytrain) using (method)"""
    services.bias_correct(
        x,
        xtrain,
        ytrain,
        out,
        storage=_authenticate_storage(),
        train_variable=trainvariable,
        out_variable=outvariable,
        method=method,
    )


@dodola_cli.command(help="Build NetCDF weights file for regridding")
@click.argument("x", required=True)
@click.option(
    "--method",
    "-m",
    required=True,
    help="Regridding method - 'bilinear' or 'conservative'",
)
@click.option(
    "--targetresolution", "-r", default=1.0, help="Global-grid resolution to regrid to"
)
@click.option("--outpath", "-o", default=None, help="Local path to write weights file")
def buildweights(x, method, targetresolution, outpath):
    """Generate local NetCDF weights file for regridding a target climate dataset

    Note, the output weights file is only written to the local disk. See
    https://xesmf.readthedocs.io/ for details on requirements for `x` with
    different methods.
    """
    # Configure storage while we have access to users configurations.
    services.build_weights(
        str(x),
        str(method),
        target_resolution=float(targetresolution),
        storage=_authenticate_storage(),
        outpath=str(outpath),
    )


@dodola_cli.command(help="Rechunk Zarr store")
@click.argument("x", required=True)
@click.option("--variable", "-v", required=True, help="Variable to rechunk")
@click.option(
    "--chunk", "-c", multiple=True, required=True, help="coord=chunksize to rechunk to"
)
@click.option(
    "--maxmemory", "-m", required=True, help="Max memory (bytes) to use for rechunking"
)
@click.option("--out", "-o", required=True)
def rechunk(x, variable, chunk, maxmemory, out):
    """Rechunk Zarr store"""
    # Convert ["k1=1", "k2=2"] into {k1: 1, k2: 2}
    coord_chunks = {c.split("=")[0]: int(c.split("=")[1]) for c in chunk}
    target_chunks = {variable: coord_chunks}

    services.rechunk(
        str(x),
        target_chunks=target_chunks,
        out=out,
        max_mem=maxmemory,
        storage=_authenticate_storage(),
    )


@dodola_cli.command(help="Build NetCDF weights file for regridding")
@click.argument("x", required=True)
@click.option("--out", "-o", required=True)
@click.option(
    "--method",
    "-m",
    required=True,
    help="Regridding method - 'bilinear' or 'conservative'",
)
@click.option("--domain_file", "-domain", help="Domain file to regrid to")
@click.option(
    "--weightspath",
    "-w",
    default=None,
    help="Local path to existing regrid weights file",
)
def regrid(x, out, method, domain_file, weightspath):
    """Regrid a target climate dataset

    Note, the weightspath only accepts paths to NetCDF files on the local disk. See
    https://xesmf.readthedocs.io/ for details on requirements for `x` with
    different methods.
    """
    # Configure storage while we have access to users configurations.
    services.regrid(
        str(x),
        out=str(out),
        method=str(method),
        storage=_authenticate_storage(),
        weights_path=weightspath,
        domain_file=domain_file,
    )
