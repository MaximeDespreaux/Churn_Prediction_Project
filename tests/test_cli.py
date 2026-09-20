"""The `churn` command dispatches to the pipeline modules."""

import pytest

from churn_prediction import __main__ as cli


def test_help_lists_every_command(capsys):
    assert cli.main([]) == 0
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for command in cli.COMMANDS:
        assert command in out


def test_unknown_command(capsys):
    assert cli.main(["deploy"]) == 2
    assert "unknown command 'deploy'" in capsys.readouterr().err


def test_dispatches_arguments_to_the_module(monkeypatch):
    calls = []
    fake = type("Module", (), {"main": staticmethod(calls.append)})
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: fake)
    assert cli.main(["predict", "--features", "x.parquet"]) == 0
    assert calls == [["--features", "x.parquet"]]


@pytest.mark.parametrize("command", list(cli.COMMANDS))
def test_every_command_has_its_own_help(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out
