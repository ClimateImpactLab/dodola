"""Commandline interface to the application.
"""

import click
import dodola.services as services
from dodola.repository import GcsRepository


# Main entry point
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def dodola_cli():
    """GCM bias-correction and downscaling"""


@dodola_cli.command(help="Bias-correct GCM on observations")
@click.argument("x", required=True)
@click.argument("xtrain", required=True)
@click.argument("ytrain", required=True)
@click.argument("out", required=True)
def biascorrect(x, xtrain, ytrain, out):
    """Bias-correct GCM (x) to 'out' based on model (xtrain), obs (ytrain)"""
    storage = GcsRepository()
    services.bias_correct(x, xtrain, ytrain, out, storage)
