from __future__ import annotations

from pathlib import Path

import pytest

from qem_bench.protocol_trace import load_protocol_rules

pytest_plugins = ("pytester",)

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "PROTOCOL-TRACE.md"


def _marker_ids(item) -> set[str]:
    marker_ids = set()
    for marker in item.iter_markers(name="protocol"):
        assert len(marker.args) == 1
        assert not marker.kwargs
        assert isinstance(marker.args[0], str)
        marker_ids.add(marker.args[0])
    return marker_ids


def test_protocol_trace_structure(request):
    rules = load_protocol_rules(TRACE)
    collected = {item.nodeid: item for item in request.session.items}

    for rule in rules:
        for implementation in rule.implementations:
            path = implementation.split("::", 1)[0]
            assert (ROOT / path).is_file(), (
                f"{rule.rule_id}: implementation file is missing: {path}"
            )
        for guard in rule.guards:
            path = guard.split("::", 1)[0]
            assert (ROOT / path).is_file(), f"{rule.rule_id}: guard file is missing: {path}"
            assert guard in collected, f"{rule.rule_id}: guard was not collected: {guard}"
            assert rule.rule_id in _marker_ids(collected[guard]), (
                f"{guard}: missing @pytest.mark.protocol({rule.rule_id!r})"
            )


def _make_protocol_project(pytester, guard_source: str) -> None:
    tests = pytester.path / "tests"
    tests.mkdir()
    source_conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    (tests / "conftest.py").write_text(source_conftest, encoding="utf-8")
    (tests / "test_guards.py").write_text(guard_source, encoding="utf-8")
    (pytester.path / "guard_impl.py").write_text("VALUE = 1\n", encoding="utf-8")
    (pytester.path / "PROTOCOL-TRACE.md").write_text(
        """# Protocol Trace

<!-- protocol-trace:begin -->
| Rule ID | Authority | Protocol Rule | Implementation | Guarding Test |
|---|---|---|---|---|
| QEM-P001 | Test | Both guards pass | `guard_impl.py` | `tests/test_guards.py::test_first`<br>`tests/test_guards.py::test_second` |
<!-- protocol-trace:end -->
""",
        encoding="utf-8",
    )
    pytester.makeini(
        """[pytest]
markers =
    protocol(id): guarding test for a protocol rule
"""
    )


def test_protocol_guard_does_not_mask_a_skipped_sibling(pytester):
    _make_protocol_project(
        pytester,
        """import pytest

@pytest.mark.protocol("QEM-P001")
def test_first():
    pass

@pytest.mark.protocol("QEM-P001")
@pytest.mark.skip(reason="masking probe")
def test_second():
    pass
""",
    )

    result = pytester.runpytest("tests", "-q", "--strict-markers", "-p", "no:cacheprovider")

    result.assert_outcomes(passed=1, skipped=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        [
            "*active protocol guards without a passed call-phase:*QEM-P001 "
            "(tests/test_guards.py::test_second)*"
        ]
    )


def test_protocol_guard_does_not_mask_an_expected_failure(pytester):
    _make_protocol_project(
        pytester,
        """import pytest

@pytest.mark.protocol("QEM-P001")
def test_first():
    pass

@pytest.mark.protocol("QEM-P001")
@pytest.mark.xfail(reason="masking probe", strict=True)
def test_second():
    assert False
""",
    )

    result = pytester.runpytest("tests", "-q", "--strict-markers", "-p", "no:cacheprovider")

    result.assert_outcomes(passed=1, xfailed=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "QEM-P001 (tests/test_guards.py::test_second)" in result.stdout.str()


def test_protocol_rule_passes_when_all_guards_pass(pytester):
    _make_protocol_project(
        pytester,
        """import pytest

@pytest.mark.protocol("QEM-P001")
def test_first():
    pass

@pytest.mark.protocol("QEM-P001")
def test_second():
    pass
""",
    )

    result = pytester.runpytest("tests", "-q", "--strict-markers", "-p", "no:cacheprovider")

    result.assert_outcomes(passed=2)
    assert result.ret == pytest.ExitCode.OK
