"""Commandline interface to the application.
"""

import logging
import click
import dodola.services as services
from dodola.repository import AzureZarr


logger = logging.getLogger(__name__)


# Main entry point
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--debug/--no-debug", default=False, envvar="DODOLA_DEBUG")
def dodola_cli(debug):
    """GCM bias-correction and downscaling"""
    loglevel = logging.INFO
    if debug:
        loglevel = logging.DEBUG

    logging.basicConfig(level=loglevel)


@dodola_cli.command(help="Bias-correct GCM on observations")
@click.argument("x", required=True)
@click.argument("xtrain", required=True)
@click.argument("trainvariable", required=True)
@click.argument(
    "ytrain",
    required=True,
)
@click.argument("out", required=True)
@click.argument("outvariable", required=True)
@click.option(
    "--azstorageaccount",
    default=None,
    envvar="AZURE_STORAGE_ACCOUNT",
    help="Key-based Azure storage credential",
)
@click.option(
    "--azstoragekey",
    default=None,
    envvar="AZURE_STORAGE_KEY",
    help="Key-based Azure storage credential",
)
@click.option(
    "--azclientid",
    default=None,
    envvar="AZURE_CLIENT_ID",
    help="Service Principal-based Azure storage credential",
)
@click.option(
    "--azclientsecret",
    default=None,
    envvar="AZURE_CLIENT_SECRET",
    help="Service Principal-based Azure storage credential",
)
@click.option(
    "--aztenantid",
    default=None,
    envvar="AZURE_TENANT_ID",
    help="Service Principal-based Azure storage credential",
)
def biascorrect(
    x,
    xtrain,
    trainvariable,
    ytrain,
    out,
    outvariable,
    azstorageaccount,
    azstoragekey,
    azclientid,
    azclientsecret,
    aztenantid,
):
    """Bias-correct GCM (x) to 'out' based on model (xtrain), obs (ytrain)"""

    # Configure storage while we have access to users configurations.
    storage = AzureZarr(
        account_name=azstorageaccount,
        account_key=azstoragekey,
        client_id=azclientid,
        client_secret=azclientsecret,
        tenant_id=aztenantid,
    )
    services.bias_correct(
        x,
        xtrain,
        ytrain,
        out,
        storage,
        train_variable=trainvariable,
        out_variable=outvariable,
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
    "--targetresolution",
    "-r",
    default=1.0,
    help="Global-grid resolution to regrid to",
)
@click.option(
    "--outpath",
    "-o",
    default=None,
    help="Local path to write weights file",
)
@click.option(
    "--azstorageaccount",
    default=None,
    envvar="AZURE_STORAGE_ACCOUNT",
    help="Key-based Azure storage credential",
)
@click.option(
    "--azstoragekey",
    default=None,
    envvar="AZURE_STORAGE_KEY",
    help="Key-based Azure storage credential",
)
@click.option(
    "--azclientid",
    default=None,
    envvar="AZURE_CLIENT_ID",
    help="Service Principal-based Azure storage credential",
)
@click.option(
    "--azclientsecret",
    default=None,
    envvar="AZURE_CLIENT_SECRET",
    help="Service Principal-based Azure storage credential",
)
@click.option(
    "--aztenantid",
    default=None,
    envvar="AZURE_TENANT_ID",
    help="Service Principal-based Azure storage credential",
)
def buildweights(
    x,
    method,
    targetgrid,
    outpath,
    azstorageaccount,
    azstoragekey,
    azclientid,
    azclientsecret,
    aztenantid,
):
    """Generate local NetCDF weights file for regridding a target climate dataset

    Note, the output weights file is only written to the local disk. See
    https://xesmf.readthedocs.io/ for details on requirements for `x` with
    different methods.
    """

    # Configure storage while we have access to users configurations.
    storage = AzureZarr(
        account_name=azstorageaccount,
        account_key=azstoragekey,
        client_id=azclientid,
        client_secret=azclientsecret,
        tenant_id=aztenantid,
    )
    services.build_weights(
        str(x),
        str(method),
        target_resolution=float(targetgrid),
        storage=storage,
        outpath=str(outpath),
    )
