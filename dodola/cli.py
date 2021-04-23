"""Commandline interface to the application.
"""

import logging
import click
import dodola.services as services


logger = logging.getLogger(__name__)


# Main entry point
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--debug/--no-debug", default=False, envvar="DODOLA_DEBUG")
def dodola_cli(debug):
    """GCM bias-correction and downscaling

    Authenticate with storage by setting the appropriate environment variables
    for your fsspec-compatible URL library.
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


@dodola_cli.command(help="Clean up and standardize GCM")
@click.argument("x", required=True)
@click.argument("out", required=True)
@click.option(
    "--drop-leapdays/--no-drop-leapdays",
    default=True,
    help="Whether to remove leap days",
)
def cleancmip6(x, out, drop_leapdays):
    """Clean and standardize CMIP6 GCM to 'out'. If drop-leapdays option is set, remove leap days"""
    services.clean_cmip6(x, out, drop_leapdays)


@dodola_cli.command(help="Remove leap days and update calendar")
@click.argument("x", required=True)
@click.argument("out", required=True)
def removeleapdays(x, out):
    """ Remove leap days and update calendar attribute"""
    services.remove_leapdays(x, out)


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
        train_variable=trainvariable,
        out_variable=outvariable,
        method=method,
    )


@dodola_cli.command(help="Downscale bias-corrected GCM")
@click.argument("x", required=True)
@click.argument("trainvariable", required=True)
@click.argument("yclimocoarse", required=True)
@click.argument("yclimofine", required=True)
@click.argument("out", required=True)
@click.argument("af", required=False)
@click.argument("outvariable", required=True)
@click.argument("method", required=True)
@click.option("--domain-file", "-d", required=True, help="Domain file to regrid to")
@click.option(
    "--weightspath",
    "-w",
    default=None,
    help="Local path to existing regrid weights file",
)
def downscale(
    x,
    trainvariable,
    yclimocoarse,
    yclimofine,
    out,
    af,
    outvariable,
    method,
    domain_file,
    weightspath,
):
    """Downscale bias corrected GCM to 'out' based on obs climo (yclimocoarse, yclimofine) using (method) and (domain_file)"""
    services.downscale(
        x,
        yclimocoarse,
        yclimofine,
        out,
        af,
        storage=_authenticate_storage(),
        train_variable=trainvariable,
        out_variable=outvariable,
        method=method,
        domain_file=domain_file,
        weights_path=weightspath,
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
        outpath=str(outpath),
    )


@dodola_cli.command(help="Rechunk Zarr store")
@click.argument("x", required=True)
@click.option(
    "--chunk", "-c", multiple=True, required=True, help="coord=chunksize to rechunk to"
)
@click.option("--out", "-o", required=True)
def rechunk(x, chunk, out):
    """Rechunk Zarr store"""
    # Convert ["k1=1", "k2=2"] into {k1: 1, k2: 2}
    coord_chunks = {c.split("=")[0]: int(c.split("=")[1]) for c in chunk}

    services.rechunk(
        str(x),
        target_chunks=coord_chunks,
        out=out,
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
@click.option("--domain-file", "-d", help="Domain file to regrid to")
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
        domain_file=domain_file,
        weights_path=weightspath,
    )
