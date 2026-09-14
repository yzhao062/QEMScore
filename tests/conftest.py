from __future__ import annotations

from pathlib import Path

import pytest

from qemscore.protocol_trace import load_protocol_rules

_PASSED_GUARDS = pytest.StashKey[set[tuple[str, str]]]()
_ACTIVE_GUARDS = pytest.StashKey[set[tuple[str, str]]]()
_ITEM_GUARDS = pytest.StashKey[tuple[tuple[str, str], ...]]()
_TRACE = Path(__file__).resolve().parents[1] / "PROTOCOL-TRACE.md"


def _marker_ids(item: pytest.Item) -> set[str]:
    marker_ids = set()
    for marker in item.iter_markers(name="protocol"):
        if len(marker.args) != 1 or marker.kwargs or not isinstance(marker.args[0], str):
            raise pytest.UsageError(
                f"{item.nodeid}: protocol marker requires one string rule ID"
            )
        marker_ids.add(marker.args[0])
    return marker_ids


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_PASSED_GUARDS] = set()
    config.stash[_ACTIVE_GUARDS] = set()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    try:
        rules = load_protocol_rules(_TRACE)
    except (OSError, ValueError) as error:
        raise pytest.UsageError(str(error)) from error
    active_ids = {rule.rule_id for rule in rules}
    listed: dict[str, set[str]] = {}
    for rule in rules:
        for guard in rule.guards:
            listed.setdefault(guard, set()).add(rule.rule_id)
    collected_node_ids = {item.nodeid for item in items}
    config.stash[_ACTIVE_GUARDS] = {
        (rule.rule_id, guard)
        for rule in rules
        for guard in rule.guards
        if guard in collected_node_ids
    }
    for item in items:
        marker_ids = _marker_ids(item)
        unknown = marker_ids - active_ids
        if unknown:
            raise pytest.UsageError(
                f"{item.nodeid}: protocol marker names inactive rules {sorted(unknown)}"
            )
        item.stash[_ITEM_GUARDS] = tuple(
            (rule_id, item.nodeid)
            for rule_id in marker_ids
            if rule_id in listed.get(item.nodeid, set())
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.passed or hasattr(report, "wasxfail"):
        return
    item.config.stash[_PASSED_GUARDS].update(item.stash.get(_ITEM_GUARDS, ()))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    missing = sorted(
        session.config.stash[_ACTIVE_GUARDS]
        - session.config.stash[_PASSED_GUARDS]
    )
    if not missing:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            "active protocol guards without a passed call-phase: "
            + ", ".join(f"{rule_id} ({nodeid})" for rule_id, nodeid in missing),
            red=True,
        )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
