"""``hermes insights`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_insights_parser(subparsers, *, cmd_insights: Callable) -> None:
    """Attach the ``insights`` subcommand to ``subparsers``."""
    insights_parser = subparsers.add_parser(
        "insights", help="Show usage insights and analytics",
        description="Analyze session history to show token usage, costs, tool patterns, and activity trends",
    )
    insights_parser.add_argument(
        "--days", type=int, default=30, help="Number of days to analyze (default: 30)")
    insights_parser.add_argument(
        "--source", help="Filter by platform (cli, telegram, discord, etc.)")
    insights_parser.add_argument(
        "--ledger", action="store_true",
        help="Report the local call ledger (LLM/tool calls per turn, cache share, latency, cost) "
             "instead of session analytics; needs agent.call_ledger.enabled")
    insights_parser.add_argument(
        "--ledger-dir", action="append", metavar="DIR",
        help="Ledger directory to read (repeatable), e.g. another HERMES_HOME's call_ledger/ from an "
             "eval or batch_runner run (default: this profile's)")
    insights_parser.add_argument("--session", help="With --ledger: only this session id")
    insights_parser.add_argument("--json", action="store_true", help="With --ledger: print the report as JSON")
    insights_parser.set_defaults(func=cmd_insights)
