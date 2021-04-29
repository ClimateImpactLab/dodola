import pytest
from click.testing import CliRunner
import dodola.cli
import dodola.services


@pytest.mark.parametrize(
    "subcmd",
    [None, "biascorrect", "buildweights", "rechunk", "regrid", "find-qdm-ryw"],
    ids=(
        "--help",
        "biascorrect --help",
        "buildweights --help",
        "rechunk --help",
        "regrid --help",
        "find-qdm-ryw --help",
    ),
)
def test_cli_helpflags(subcmd):
    """Test that CLI commands and subcommands don't throw Error if given --help flag"""
    runner = CliRunner()

    # Setup CLI args
    cli_args = ["--help"]
    if subcmd is not None:
        cli_args = [subcmd, "--help"]

    result = runner.invoke(dodola.cli.dodola_cli, cli_args)
    assert "Error:" not in result.output
