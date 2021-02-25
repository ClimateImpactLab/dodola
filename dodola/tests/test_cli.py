import pytest
from click.testing import CliRunner
import dodola.cli
import dodola.services


@pytest.mark.parametrize(
    "subcmd",
    [None, "biascorrect", "buildweights", "rechunk"],
    ids=("--help", "biascorrect --help", "buildweights --help"),
)
def test_cli_helpflags(subcmd):
    """Test that CLI commands don't throw Error if given --help flag"""
    runner = CliRunner()

    # Setup CLI args
    cli_args = ["--help"]
    if subcmd is not None:
        cli_args = [subcmd, "--help"]

    result = runner.invoke(dodola.cli.dodola_cli, cli_args)
    assert "Error:" not in result.output
