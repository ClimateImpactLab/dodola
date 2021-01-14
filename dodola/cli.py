"""Commandline interface to the application.
"""

import click
import dodola.services as services
from dodola.repository import AzureZarr


# Main entry point
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def dodola_cli():
    """GCM bias-correction and downscaling"""


@dodola_cli.command(help="Bias-correct GCM on observations")
@click.argument(
    "x",
    required=True,
    help="URL of Zarr store to extract independent variable or GCM data for training.",
)
@click.argument(
    "xtrain",
    required=True,
    help="URL of Zarr store to extract independent variable or obs for training.",
)
@click.argument(
    "trainvariable", required=True, help="Variable to extract for training."
)
@click.argument(
    "ytrain",
    required=True,
    help="URL of Zarr store to extract dependant variable for training.",
)
@click.argument("out", required=True, help="URL of store to write output Zarr data to.")
@click.argument(
    "outvariable", required=True, help="Variable name to assign output in output file."
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
