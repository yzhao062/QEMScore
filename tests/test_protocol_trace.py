from __future__ import annotations

import ast
from pathlib import Path

import pytest

from qemscore.protocol_trace import load_protocol_rules

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


def _source_marker_ids(guard: str) -> set[str]:
    path, *nodes = guard.split("::")
    assert len(nodes) == 1, f"unsupported guard node reference: {guard}"
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == nodes[0]
        ),
        None,
    )
    assert function is not None, f"guard function is missing: {guard}"
    marker_ids = set()
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call) or not isinstance(
            decorator.func, ast.Attribute
        ):
            continue
        owner = decorator.func.value
        if (
            decorator.func.attr == "protocol"
            and isinstance(owner, ast.Attribute)
            and owner.attr == "mark"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "pytest"
            and len(decorator.args) == 1
            and isinstance(decorator.args[0], ast.Constant)
            and isinstance(decorator.args[0].value, str)
            and not decorator.keywords
        ):
            marker_ids.add(decorator.args[0].value)
    return marker_ids


def _assert_implementation_symbol(implementation: str, rule_id: str) -> None:
    parts = implementation.split("::")
    assert len(parts) == 2 and all(parts), (
        f"{rule_id}: implementation reference must be path::symbol: {implementation}"
    )
    path, symbol = parts
    source = ROOT / path
    assert source.is_file(), f"{rule_id}: implementation file is missing: {path}"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definition = next(
        (
            node
            for node in tree.body
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and node.name == symbol
        ),
        None,
    )
    assert definition is not None, (
        f"{rule_id}: implementation symbol is missing: {implementation}"
    )


def _is_full_suite_invocation(request) -> bool:
    if len(request.config.args) != 1:
        return False
    selector = request.config.args[0]
    if "::" in selector:
        return False
    selected_path = Path(selector).resolve()
    return selected_path in {ROOT, ROOT / "tests"}


def test_protocol_trace_source_catalog_inspects_source_text():
    rules = load_protocol_rules(TRACE)
    assert rules, "PROTOCOL-TRACE.md must contain at least one active rule"

    for rule in rules:
        for implementation in rule.implementations:
            _assert_implementation_symbol(implementation, rule.rule_id)
        for guard in rule.guards:
            path = guard.split("::", 1)[0]
            assert (ROOT / path).is_file(), (
                f"{rule.rule_id}: guard file is missing: {path}"
            )
            assert rule.rule_id in _source_marker_ids(guard), (
                f"{guard}: source is missing @pytest.mark.protocol({rule.rule_id!r})"
            )


def test_protocol_trace_rejects_a_missing_implementation_symbol():
    with pytest.raises(AssertionError, match="implementation symbol is missing"):
        _assert_implementation_symbol(
            "qemscore/runner/metrics.py::definitely_missing", "QEM-P001"
        )


@pytest.mark.parametrize(
    "implementation",
    (
        "qemscore/runner/metrics.py",
        "qemscore/runner/metrics.py::headline_metrics::extra",
    ),
)
def test_protocol_trace_requires_exact_path_symbol_references(implementation):
    with pytest.raises(AssertionError, match="must be path::symbol"):
        _assert_implementation_symbol(implementation, "QEM-P001")


def test_protocol_trace_structure(request):
    rules = load_protocol_rules(TRACE)
    assert rules, "PROTOCOL-TRACE.md must contain at least one active rule"
    if not _is_full_suite_invocation(request):
        pytest.skip(
            "runtime protocol-trace verification requires a full-suite collection; "
            "the source catalog has a separate static test"
        )
    collected = {item.nodeid: item for item in request.session.items}

    for rule in rules:
        for guard in rule.guards:
            assert guard in collected, (
                f"{rule.rule_id}: guard was not collected: {guard}"
            )
            marker_ids = _marker_ids(collected[guard])
            assert rule.rule_id in marker_ids, (
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
