"""Report strata and split identities, from metric cells to rendered artifacts."""

import copy
import json
import math
from pathlib import Path

import pytest

from qem_bench.datasets.generate import PRESETS, generate
from qem_bench.datasets.schema import FAMILY_STRATA
from qem_bench.datasets.split_generate import generate_split
from qem_bench.datasets.splits import SPLIT_AXES, SplitSpec
from qem_bench.reports import generate_report
from qem_bench.reports.generate import (
    METHOD_LABELS,
    _merge_cell_records,
    _report_description,
)
from qem_bench.runner.metrics import build_cell_records
from qem_bench.runner.run import _run_artifact_id, run, validate_run_artifact


@pytest.mark.parametrize("grouping", ["six-part", "observable-excluded"])
@pytest.mark.parametrize("role", ["train", "validation", "test"])
@pytest.mark.parametrize("family", ["tfi", "random_clifford"])
def test_cells_keep_every_axis_and_role_in_both_groupings(grouping, role, family):
    records = []
    for split_id, split_axis in SPLIT_AXES.items():
        for replica in range(2):
            items = [
                {
                    "item_id": f"{split_id}-{replica}-{observable}",
                    "circuit_id": f"circuit-{split_id}-{replica}",
                    "split": role,
                    "split_id": split_id,
                    "split_axis": split_axis,
                    "family": family,
                    "stratum": FAMILY_STRATA[family],
                    "noise_family": "depolarizing_readout",
                    "severity": "L1",
                    "observable": observable,
                    "ideal_expectation": 0.0,
                }
                for observable in ("z_mid", "zz_mid")
            ]
            records.extend(build_cell_records(
                "raw", items, [0.2, 0.6], [0.2, 0.6],
                artifact_id=f"artifact-{split_id}-{replica}", grouping=grouping,
            ))
    merged = _merge_cell_records(records)
    n_observables = 2 if grouping == "six-part" else 1
    assert len(merged) == 7 * n_observables
    assert len({record["cell_id"] for record in merged}) == 7 * n_observables
    assert {record["key"]["split_id"] for record in merged} == set(SPLIT_AXES)
    for record in merged:
        assert record["key"]["split"] == role
        assert len(record["source_artifact_ids"]) == 2
        assert len(record["items"]) == 4 // n_observables
        expected = {"z_mid": 0.2, "zz_mid": 0.6}.get(record["key"].get("observable"), 0.4)
        assert record["metrics"]["mae"] == pytest.approx(expected)


def _write_manifest(root, name, runs):
    path = root / f"{name}.json"
    path.write_text(json.dumps({
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [{"results": relative} for relative, _ in runs],
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize("grouping", ["six-part", "observable-excluded"])
@pytest.mark.parametrize("defect", ["missing-id", "missing-axis", "wrong-axis", "mixed-legacy"])
def test_split_cells_reject_incomplete_or_mixed_axis_metadata(grouping, defect):
    item = {
        "item_id": "item-0", "circuit_id": "circuit-0", "split": "test",
        "family": "tfi", "stratum": "continuous_regression",
        "noise_family": "depolarizing_readout", "severity": "L1",
        "observable": "z_mid", "ideal_expectation": 0.0,
        "dataset_schema_version": "split-v2",
        "split_id": "S4", "split_axis": "family_native_depth",
    }
    items = [item]
    if defect == "missing-id":
        item.pop("split_id")
    elif defect == "missing-axis":
        item.pop("split_axis")
    elif defect == "wrong-axis":
        item["split_axis"] = "shots"
    else:
        legacy = {key: value for key, value in item.items() if key not in {
            "dataset_schema_version", "split_id", "split_axis",
        }}
        items.append(legacy)
    with pytest.raises(ValueError, match="split-aware cell records require"):
        build_cell_records(
            "raw", items, [0.1] * len(items), [0.1] * len(items),
            artifact_id="artifact-0", grouping=grouping,
        )


@pytest.mark.parametrize("grouping", ["six-part", "observable-excluded"])
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize(
    "marker",
    [
        "schema-only", "split-id-only", "split-axis-only", "complete",
        # Present but empty is malformed metadata, not absent metadata. Detection
        # by truthiness rather than by key presence would read these as legacy and
        # merge the batch silently, which is the one outcome that loses data.
        "split-id-none", "split-id-empty", "split-axis-none", "split-axis-empty",
    ],
)
def test_one_split_looking_row_anywhere_in_a_legacy_batch_is_refused(
    grouping, position, marker
):
    """A single split-looking row must activate the split contract from any position.

    The implementation scans the whole batch. A first-row-only scan would read a
    legacy-first batch as legacy and merge it into one cell instead of refusing it,
    so every case here places the marked row at a different index and asserts the
    refusal that such a scan would lose.
    """
    def _legacy(index):
        return {
            "item_id": f"item-{index}", "circuit_id": f"circuit-{index}",
            "split": "test", "family": "tfi", "stratum": "continuous_regression",
            "noise_family": "depolarizing_readout", "severity": "L1",
            "observable": "z_mid", "ideal_expectation": 0.0,
        }

    items = [_legacy(index) for index in range(3)]
    marked = items[position]
    if marker == "schema-only":
        marked["dataset_schema_version"] = "split-v2"
    elif marker == "split-id-only":
        marked["split_id"] = "S4"
    elif marker == "split-axis-only":
        marked["split_axis"] = "family_native_depth"
    elif marker == "split-id-none":
        marked["split_id"] = None
    elif marker == "split-id-empty":
        marked["split_id"] = ""
    elif marker == "split-axis-none":
        marked["split_axis"] = None
    elif marker == "split-axis-empty":
        marked["split_axis"] = ""
    else:
        marked.update({
            "dataset_schema_version": "split-v2",
            "split_id": "S4", "split_axis": "family_native_depth",
        })

    with pytest.raises(ValueError, match="split-aware cell records require"):
        build_cell_records(
            "raw", items, [0.1] * len(items), [0.1] * len(items),
            artifact_id="artifact-0", grouping=grouping,
        )


def test_every_section_is_its_own_float_and_no_row_is_clipped(stratum_runs, tmp_path):
    """A section that does not fit the page must move, never disappear.

    One unbreakable float holding every section silently drops whatever overflows:
    resizebox scales width only, a table float cannot break, and pdflatex still
    exits 0. The compiled page then omits the trailing sections and the artifact
    note while every number stays in the trace, so an exit-code assertion passes
    the broken output. This checks the structure that prevents it, and, where a
    compiler exists, that the compiler itself reports no oversized float.
    """
    import re
    import shutil
    import subprocess

    root, version, runs = stratum_runs
    methods = tuple(runs[0][1]["methods"])
    manifest = _write_manifest(root, "mixed-pagination", runs)
    outputs = generate_report(
        manifest, tmp_path / "report", methods=methods, bootstrap_resamples=20
    )
    tex = outputs["table1"].read_text(encoding="utf-8")

    # The expected sections come from the fixture, not from the rendered captions:
    # a renderer that drops a section, duplicates one, or emits a header without
    # its body would still agree with a structure read back out of its own output.
    split_ids = [] if version == "legacy-v1" else ["S0", "S4", "S6"]
    expected_sections = [
        label if split_id is None else f"{label}: {split_id}"
        for label in (
            "Continuous regression (headline)",
            "Clifford control (excluded from headline)",
        )
        for split_id in (None, *split_ids)
    ]
    expected_labels = [METHOD_LABELS.get(method, method) for method in methods]
    assert re.findall(r"Section: ([^.]+)\.", tex) == expected_sections
    assert len(re.findall(r"\\begin\{table\}", tex)) == len(expected_sections)
    assert len(re.findall(r"Method & Cells & Macro MAE", tex)) == len(expected_sections)
    # The artifact note rides the last float, so it cannot be orphaned onto a page
    # whose table was dropped.
    assert tex.count("Run artifacts:") == 1
    last_float = tex.rsplit(r"\begin{table}", 1)[1]
    assert "Run artifacts:" in last_float
    # Every method appears once in every section, checked inside each float rather
    # than summed over the document: a section rendered empty and another rendered
    # twice keeps the document-wide row count correct.
    floats = tex.split(r"\begin{table}")[1:]
    assert len(floats) == len(expected_sections)
    for block, section in zip(floats, expected_sections, strict=True):
        assert block.count(f"Section: {section}.") == 1
        assert block.count("Method & Cells & Macro MAE") == 1
        rows = re.findall(r"(?m)^([^&\n]+) & \d+ & ", block)
        assert [row.strip() for row in rows] == expected_labels

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        pytest.skip("pdflatex is not installed")
    build = tmp_path / "latex"
    build.mkdir()
    shutil.copyfile(outputs["table1"], build / "table1.tex")
    completed = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-file-line-error", "table1.tex"],
        cwd=build, check=False, capture_output=True, text=True,
    )
    log = (build / "table1.log").read_text(encoding="utf-8", errors="replace")
    assert completed.returncode == 0, log[-3000:]
    assert "Float too large for page" not in log, log[-3000:]
    assert "Overfull \\vbox" not in log, log[-3000:]


def test_figure_legend_stays_clear_of_the_panels_and_the_title(
    stratum_runs, monkeypatch, tmp_path
):
    """A legend that outgrows its panels covers them, and nothing reports it.

    Color and marker carry the method, so a split ID can only reach the reader
    through the legend text. Adding it multiplies the entries by the number of
    splits while the panel height stays fixed by the stratum count: at eight
    methods and three splits the one-column legend measured nearly three times
    the height of the panel holding it, ran across the neighbouring plot and the
    report title, and still wrote a PDF that opens. Geometry is what detects it.
    The collection counts, the point total and the PDF signature were all correct
    while the figure was unreadable, so this measures the laid-out boxes.
    """
    import re

    from matplotlib.figure import Figure

    root, version, runs = stratum_runs
    methods = tuple(runs[0][1]["methods"])
    measured = []
    original = Figure.savefig

    def savefig(figure, *args, **kwargs):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        measured.append({
            "legend": figure.legends[0].get_window_extent(renderer),
            "entries": [text.get_text() for text in figure.legends[0].get_texts()],
            "panels": [axis.get_tightbbox(renderer) for axis in figure.axes],
            "title": figure._suptitle.get_window_extent(renderer),
            "canvas": figure.get_window_extent(renderer),
        })
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    manifest = _write_manifest(root, "legend-geometry", runs)
    outputs = generate_report(
        manifest, tmp_path / "report", methods=methods, bootstrap_resamples=20
    )

    assert len(measured) == (1 if version == "legacy-v1" else 3)
    # Counting the drawn figures is not the same as counting the pages that reach
    # the file: one page per split has to survive into the PDF itself.
    rendered = outputs["figure1"].read_bytes()
    assert rendered.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page[^s]", rendered)) == len(measured)
    for page in measured:
        legend = page["legend"]
        # One entry per method. Per method and split is the defect itself.
        assert page["entries"] == [
            METHOD_LABELS.get(method, method) for method in methods
        ]
        for panel in page["panels"]:
            assert not legend.overlaps(panel), (legend.extents, panel.extents)
        assert not legend.overlaps(page["title"]), (
            legend.extents, page["title"].extents
        )
        # Reserved space that is too small pushes the legend off the canvas
        # rather than into a panel, which is just as unreadable.
        assert page["canvas"].y0 <= legend.y0 and legend.y1 <= page["canvas"].y1
        assert page["canvas"].x0 <= legend.x0 and legend.x1 <= page["canvas"].x1


@pytest.fixture(scope="module", params=["legacy-v1", "split-v2"])
def stratum_runs(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("strata")
    runs = []
    for family in ("tfi", "random_clifford"):
        split_ids = (None,) if request.param == "legacy-v1" else ("S0", "S4", "S6")
        for split_id in split_ids:
            name = f"{family}-{split_id}"
            data = root / name
            if split_id is None:
                preset = "t0-micro" if family == "tfi" else "t0-rc-micro"
                # A dict configuration has no preset name to classify by.
                generate(copy.deepcopy(PRESETS[preset]), data)
            else:
                fixed = {
                    "circuit_family": [family],
                    "noise_family": ["depolarizing_readout"],
                    "noise_strength": ["L1"],
                    "family_native_depth": [1],
                    "observable_class": ["z_mid", "zz_mid"],
                    "shots": [64],
                }
                source = {"circuit_instance": ["sampled"]}
                target = {"circuit_instance": ["sampled"]}
                if split_id != "S0":
                    axis = SPLIT_AXES[split_id]
                    source = {axis: fixed.pop(axis)}
                    target = {axis: [2] if split_id == "S4" else [16]}
                generate_split(SplitSpec(
                    split_id=split_id, source_domain=source, target_domain=target,
                    fixed_axes=fixed, n_qubits=[3],
                    role_counts={"train": 2, "validation": 2, "test": 2},
                    family_parameters={family: {"dt": 0.2} if family == "tfi" else {}}, budget_tier="H",
                ), data)
            relative = f"{name}-run/results.json"
            result = run(data, root / f"{name}-run")
            assert validate_run_artifact(result) is result
            runs.append((relative, result))
    return root, request.param, runs


def test_report_strata_are_invariant_to_the_other_stratum(stratum_runs, monkeypatch):
    from matplotlib.figure import Figure

    root, version, runs = stratum_runs
    captured = []
    original = Figure.savefig

    def savefig(figure, *args, **kwargs):
        captured.append((figure._suptitle.get_text(), [
            (axis.get_title(), len(axis.collections)) for axis in figure.axes
        ]))
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    methods = tuple(runs[0][1]["methods"])
    traces = {}
    for selection in ("continuous_regression", "clifford_control", "mixed"):
        selected = [entry for entry in runs if selection == "mixed" or entry[1]["stratum"] == selection]
        manifest = _write_manifest(root, selection, selected)
        outputs = generate_report(manifest, root / selection, methods=methods, bootstrap_resamples=20)
        traces[selection] = json.loads(outputs["trace"].read_text(encoding="utf-8"))
        tex = outputs["table1"].read_text(encoding="utf-8")
        assert outputs["figure1"].read_bytes().startswith(b"%PDF")
        if selection == "clifford_control":
            assert traces[selection]["table1"] == {}
            assert "no continuous-regression headline is available" in tex
        if version == "legacy-v1":
            assert "Legacy-v1 walking-skeleton results" in tex
            assert "no declared S0 to S6 split axis" in tex
        else:
            assert "Split-v2 results (S0, S4, S6)" in tex
            assert "walking-skeleton" not in tex
            assert "do not yet implement" not in tex
            assert "Validation uses the source domain" in tex
    mixed = traces["mixed"]
    assert mixed["table1"] == traces["continuous_regression"]["table1"]
    for stratum in ("continuous_regression", "clifford_control"):
        assert mixed["strata"][stratum] == traces[stratum]["strata"][stratum]
        stratum_sources = [result for _, result in runs if result["stratum"] == stratum]
        for method in methods:
            summary = mixed["strata"][stratum]["table1"][method]
            assert summary["n_expectations"] == sum(result["n_test_items"] for result in stratum_sources)
            for bucket in ("B_train", "B_extra", "B_pred"):
                assert summary["ledger"][bucket] == sum(result["methods"][method]["ledger"][bucket] for result in stratum_sources)
    assert len(mixed["figure1"]["points"]) == len(runs) * len(methods)
    for point in mixed["figure1"]["points"]:
        source = next(result for _, result in runs if result["artifact_id"] == point["artifact_id"])
        assert point["stratum"] == source["stratum"]
        raw_mae = source["methods"]["raw"]["metrics"]["mae"]
        if raw_mae == 0:
            assert point["relative_macro_mae"] is None
        else:
            assert point["relative_macro_mae"] == pytest.approx(
                source["methods"][point["method"]]["metrics"]["mae"] / raw_mae
            )
    # One page per declared split, in split order; the mixed report is the last
    # of the three, so its pages are the tail of the capture.
    split_ids = [None] if version == "legacy-v1" else ["S0", "S4", "S6"]
    assert len(captured) == 3 * len(split_ids)
    for split_id, (title, panels) in zip(split_ids, captured[-len(split_ids):], strict=True):
        assert len(panels) == 4
        assert all("Continuous regression" in label for label, _ in panels[:2])
        assert all("Clifford control" in label for label, _ in panels[2:])
        # A page holds one run per stratum, so a panel carries the roster once.
        assert all(count == len(methods) for _, count in panels)
        if split_id is None:
            assert "Page" not in title and "walking-skeleton" in title
        else:
            assert "walking-skeleton" not in title
            assert "S0, S4, S6" in title and f"split {split_id}" in title
    by_page = {}
    for point in mixed["figure1"]["points"]:
        by_page.setdefault(point["page"], set()).add(
            (point["split_id"], point["artifact_id"], point["method"])
        )
    assert sorted(by_page) == list(range(1, len(split_ids) + 1))
    for page, split_id in zip(sorted(by_page), split_ids, strict=True):
        members = by_page[page]
        assert {split for split, _, _ in members} == {split_id}
        assert {method for _, _, method in members} == set(methods)
        expected_artifacts = {
            result["artifact_id"] for _, result in runs
            if (None if version == "legacy-v1" else result["test_items"][0]["split_id"]) == split_id
        }
        assert {artifact for _, artifact, _ in members} == expected_artifacts
        assert len(members) == len(expected_artifacts) * len(methods)
    if version == "split-v2":
        assert any(point["relative_macro_mae"] is None for point in mixed["figure1"]["points"])
        for stratum, section in mixed["strata"].items():
            assert set(section["by_split"]) == {"S0", "S4", "S6"}
            for method in methods:
                cells = section["table1"][method]["cells"]
                assert len(cells) == 6
                assert {cell["key"]["split_id"] for cell in cells} == {"S0", "S4", "S6"}
            for split_id, summary in section["by_split"].items():
                source = next(result for _, result in runs if result["stratum"] == stratum and result["test_items"][0]["split_id"] == split_id)
                assert f"{'Continuous regression (headline)' if stratum == 'continuous_regression' else 'Clifford control (excluded from headline)'}: {split_id}" in tex
                for method in methods:
                    errors = [abs(prediction - item["ideal_expectation"]) for prediction, item in zip(source["methods"][method]["predictions"], source["test_items"], strict=True)]
                    # Both observables have exactly two items, so the direct mean
                    # is also the independently specified equal-cell macro mean.
                    assert summary[method]["metrics"]["macro_mae"]["mean"] == pytest.approx(math.fsum(errors) / 4)


def _resign_identity(result):
    result["artifact_id"] = _run_artifact_id(
        result["dataset_hash"], result["preset"], result["dataset_item_stream_hashes"],
        result["test_items"], result["methods"],
        dataset_environment_contract=result["dataset_environment_contract"],
        environment_contract=result["environment_contract"],
        role_assignment=result.get("role_assignment"), budget=result.get("budget"),
    )


def test_run_rejects_resigned_stratum_relabels(stratum_runs):
    _, _, runs = stratum_runs
    for _, valid in runs:
        other = "clifford_control" if valid["stratum"] == "continuous_regression" else "continuous_regression"
        for relabel_items in (False, True):
            changed = copy.deepcopy(valid)
            changed["stratum"] = other
            changed["methods"]["raw"]["config"]["run_artifact_identity"]["stratum"] = other
            if relabel_items:
                for item in changed["test_items"]:
                    item["stratum"] = other
            _resign_identity(changed)
            with pytest.raises(ValueError, match="stratum must match every test item and its family"):
                validate_run_artifact(changed)


def test_split_run_requires_consistent_bound_axis_metadata(stratum_runs):
    _, version, runs = stratum_runs
    for _, valid in runs:
        if version == "legacy-v1":
            assert valid["analysis_contract"]["metric_schema"] == "qem-bench-cell-metrics-v2"
            assert all("split_id" not in record["key"] for method in valid["methods"].values() for record in method["cell_records"])
            continue
        assert valid["analysis_contract"]["metric_schema"] == "qem-bench-cell-metrics-v3"
        assert all("split_id" in fields for fields in valid["analysis_contract"]["cell_groupings"].values())
        changed = copy.deepcopy(valid)
        for item in changed["test_items"]:
            item.pop("split_id")
            item.pop("split_axis")
        with pytest.raises(ValueError, match="lacks split identity metadata; rerun the dataset"):
            validate_run_artifact(changed)

        changed = copy.deepcopy(valid)
        changed["test_items"][0]["split_axis"] = "wrong-axis"
        with pytest.raises(ValueError, match="split_axis must match split_id"):
            validate_run_artifact(changed)
        changed = copy.deepcopy(valid)
        replacement = "S6" if valid["test_items"][0]["split_id"] != "S6" else "S4"
        for item in changed["test_items"]:
            item["split_id"] = replacement
            item["split_axis"] = SPLIT_AXES[replacement]
        _resign_identity(changed)
        assert changed["artifact_id"] != valid["artifact_id"]
        with pytest.raises(ValueError, match="budget and test items disagree on split or stratum"):
            validate_run_artifact(changed)


def test_caption_names_only_supplied_domains(stratum_runs):
    _, version, runs = stratum_runs
    if version == "legacy-v1":
        title, note = _report_description([result for _, result in runs])
        assert title == "Legacy-v1 walking-skeleton results"
        assert note == "These legacy-v1 runs have no declared S0 to S6 split axis."
        return
    for split_ids in ({"S0"}, {"S4", "S6"}):
        selected = [result for _, result in runs if result["test_items"][0]["split_id"] in split_ids]
        title, note = _report_description(selected)
        assert title == "Split-v2 results (" + ", ".join(sorted(split_ids)) + ")"
        if "S0" in split_ids:
            assert "S0 is in-distribution." in note
            assert "out-of-distribution" not in note
        else:
            assert "S4, S6 are out-of-distribution." in note
            assert "S0" not in note


def test_committed_example_excludes_clifford_from_headline():
    root = Path(__file__).parents[1] / "examples" / "report-walking-skeleton"
    trace = json.loads((root / "report-trace.json").read_text(encoding="utf-8"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in (root / "runs").glob("*/results.json")]
    by_cell = {}
    for result in runs:
        for record in result["methods"]["ridge"]["cell_records"]:
            key = (result["stratum"], tuple(sorted(record["key"].items())))
            by_cell.setdefault(key, []).extend(abs(item["prediction"] - item["target"]) for item in record["items"])
    means = {key: math.fsum(errors) / len(errors) for key, errors in by_cell.items()}
    assert math.fsum(means.values()) / len(means) == pytest.approx(0.053048609009202115)
    for stratum in ("continuous_regression", "clifford_control"):
        selected = [value for (role, _), value in means.items() if role == stratum]
        summary = trace["strata"][stratum]["table1"]["ridge"]
        assert summary["metrics"]["macro_mae"]["mean"] == pytest.approx(math.fsum(selected) / len(selected))
    assert trace["table1"]["ridge"]["metrics"]["macro_mae"]["mean"] == pytest.approx(0.056577102850636725)
    assert trace["table1"]["ridge"]["n_cells"] == 18
    assert trace["strata"]["clifford_control"]["table1"]["ridge"]["n_cells"] == 4
