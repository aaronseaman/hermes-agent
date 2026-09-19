from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from cli import HermesCLI
from hermes_cli.main_agent_cmds import cmd_insights
from hermes_state import _default_db_path


@pytest.fixture(autouse=True)
def _state_db_exists():
    # insights short-circuits before opening when state.db is absent (fresh install).
    _default_db_path().parent.mkdir(parents=True, exist_ok=True)
    _default_db_path().touch()


class _InsightsEngineStub:
    calls = []

    def __init__(self, db):
        self.db = db

    def generate(self, *, days=30, source=None):
        self.calls.append({"days": days, "source": source})
        return {"days": days, "source": source}

    def format_terminal(self, report):
        return f"days={report['days']} source={report['source']}"


def _run_show_insights(command: str):
    cli_obj = HermesCLI.__new__(HermesCLI)
    db = MagicMock()
    _InsightsEngineStub.calls = []
    with patch("hermes_state.SessionDB", return_value=db), \
         patch("agent.insights.InsightsEngine", _InsightsEngineStub):
        cli_obj._show_insights(command)
    return _InsightsEngineStub.calls, db


def test_cli_insights_accepts_positional_days(capsys):
    calls, db = _run_show_insights("/insights 7")

    assert calls == [{"days": 7, "source": None}]
    db.close.assert_called_once()
    assert "days=7 source=None" in capsys.readouterr().out


def test_cli_insights_keeps_days_flag_and_source(capsys):
    calls, db = _run_show_insights("/insights --days 14 --source discord")

    assert calls == [{"days": 14, "source": "discord"}]
    db.close.assert_called_once()
    assert "days=14 source=discord" in capsys.readouterr().out


def test_insights_skips_open_when_store_missing(capsys):
    # Fresh install: no state.db yet → no open at all (a read-only open needs an existing file).
    _default_db_path().unlink()
    for run in (lambda: HermesCLI.__new__(HermesCLI)._show_insights("/insights 7"),
                lambda: cmd_insights(SimpleNamespace(days=30, source=None))):
        with patch("hermes_state.SessionDB") as ctor:
            run()
        ctor.assert_not_called()
    assert "No session data yet." in capsys.readouterr().out


def test_show_insights_opens_read_only():
    cli_obj = HermesCLI.__new__(HermesCLI)
    db = MagicMock()
    with patch("hermes_state.SessionDB", return_value=db) as ctor, \
         patch("agent.insights.InsightsEngine", _InsightsEngineStub):
        _InsightsEngineStub.calls = []
        cli_obj._show_insights("/insights 7")
    ctor.assert_called_once_with(read_only=True)
    db.close.assert_called_once()


def test_subcommand_insights_opens_read_only():
    db = MagicMock()
    with patch("hermes_state.SessionDB", return_value=db) as ctor, \
         patch("agent.insights.InsightsEngine", _InsightsEngineStub):
        cmd_insights(SimpleNamespace(days=30, source=None))
    ctor.assert_called_once_with(read_only=True)
    db.close.assert_called_once()


def test_subcommand_insights_closes_database_when_generation_fails(capsys):
    db = MagicMock()
    with patch("hermes_state.SessionDB", return_value=db), \
         patch("agent.insights.InsightsEngine", side_effect=RuntimeError("boom")):
        cmd_insights(SimpleNamespace(days=30, source=None))

    db.close.assert_called_once()
    assert "Error generating insights: boom" in capsys.readouterr().out


def test_insights_ledger_reports_the_active_profiles_records(capsys):
    """`hermes insights --ledger --json` reads this profile's call_ledger/ and counts what was recorded."""
    import argparse
    import json

    from agent import call_ledger
    from agent.call_ledger_store import WRITER
    from hermes_cli.subcommands.insights import build_insights_parser

    class _Agent:
        _call_ledger_settings = call_ledger.LedgerSettings(enabled=True)
        session_id, platform, provider, model, api_mode = "cli-ledger", "cli", "p", "m", "chat_completions"

    agent = _Agent()
    for _ in range(2):
        token = call_ledger.begin_turn(agent, "task")
        call_ledger.record_model_call(agent, usage=None, cost_usd=None, latency_s=0.5)
        call_ledger.end_turn(token, outcome="success")
    assert WRITER.flush(timeout=5.0)

    parser = argparse.ArgumentParser()
    build_insights_parser(parser.add_subparsers(), cmd_insights=cmd_insights)
    args = parser.parse_args(["insights", "--ledger", "--json"])
    args.func(args)

    report = json.loads(capsys.readouterr().out)
    assert report["aggregate"]["turns"] == 2
    assert report["aggregate"]["model_calls_per_turn"] == 1
    assert set(report["sessions"]) == {"cli-ledger"}
