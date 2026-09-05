"""Generate Table 1, Figure 1, and their artifact/cell trace."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from qem_bench.datasets.generate import UNKNOWN_IDENTITY_ENCODING_PROFILE
from qem_bench.runner.metrics import pooled_method_metrics
from qem_bench.runner.run import (
    circuit_evaluation_ratio,
    run_identity_encoding_profiles,
    validate_run_artifact,
)
from qem_bench.stats import circuit_blocked_bootstrap, macro_mean_iqr, mean_ranks

HEADLINE_METHODS = ("raw", "ridge", "zne")
METHOD_LABELS = {"raw": "Raw", "ridge": "Ridge", "zne": "Local digital ZNE"}
STRATUM_LABELS = {
    "continuous_regression": "Continuous regression (headline)",
    "clifford_control": "Clifford control (excluded from headline)",
}


def generate_report(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    methods: Sequence[str] = HEADLINE_METHODS,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20260818,
) -> dict[str, Path]:
    """Render reports from a manifest containing one or more runner artifacts."""

    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    resolved_manifest_path = manifest_path.resolve()
    source_runs: list[dict] = []
    runs = _load_runs(
        manifest,
        resolved_manifest_path.parent,
        source_runs=source_runs,
    )
    methods = tuple(methods)
    if len(set(methods)) != len(methods) or not methods:
        raise ValueError("methods must be a nonempty unique sequence")
    for run in runs:
        missing = set(methods) - set(run["methods"])
        if missing:
            raise ValueError(f"artifact {run['artifact_id']} lacks methods {sorted(missing)}")

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace = {
        "schema_version": "qem-bench-report-trace-v2",
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": (
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        ),
        "source_runs": source_runs,
        "artifacts": [run["artifact_id"] for run in runs],
        "headline_stratum": "continuous_regression",
        "table1": {},
        "strata": {},
        "figure1": {"points": []},
    }
    sections = []
    for stratum, label in STRATUM_LABELS.items():
        stratum_runs = [run for run in runs if run["stratum"] == stratum]
        if not stratum_runs:
            continue
        summaries, summary_trace = _summarize_runs(
            stratum_runs, methods, bootstrap_resamples, bootstrap_seed
        )
        sections.append((label, summaries))
        stratum_trace = {"table1": summary_trace, "by_split": {}}
        trace["strata"][stratum] = stratum_trace
        if stratum == trace["headline_stratum"]:
            trace["table1"] = summary_trace
        if stratum_runs[0]["dataset_schema_version"] == "split-v2":
            for split_id in sorted({_run_split_id(run) for run in stratum_runs}):
                axis_runs = [
                    run for run in stratum_runs if _run_split_id(run) == split_id
                ]
                axis_summaries, axis_trace = _summarize_runs(
                    axis_runs, methods, bootstrap_resamples, bootstrap_seed
                )
                sections.append((f"{label}: {split_id}", axis_summaries))
                stratum_trace["by_split"][split_id] = axis_trace

    table_path = output / "table1.tex"
    table_path.write_text(_table_tex(sections, runs), encoding="utf-8")
    figure_path = output / "figure1.pdf"
    _figure_pdf(runs, methods, figure_path, trace)
    trace_path = output / "report-trace.json"
    trace_path.write_text(json.dumps(trace, indent=2, allow_nan=False), encoding="utf-8")
    return {"table1": table_path, "figure1": figure_path, "trace": trace_path}


def _run_split_id(run: Mapping[str, object]) -> str | None:
    if run["dataset_schema_version"] == "legacy-v1":
        return None
    return str(run["test_items"][0]["split_id"])


def _summarize_runs(
    runs: Sequence[dict],
    methods: Sequence[str],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict, dict]:
    """Summarize a single stratum, optionally restricted to one split."""

    all_records = {
        method: _merge_cell_records(
            [
                record
                for run in runs
                for record in run["methods"][method]["cell_records"]
            ]
        )
        for method in methods
    }
    ranks = mean_ranks(
        [record for method in methods for record in all_records[method]], metric="mae"
    )

    summaries = {}
    trace = {}
    for method_index, method in enumerate(methods):
        records = all_records[method]
        metrics = macro_mean_iqr(records)
        interval = circuit_blocked_bootstrap(
            records,
            metric="mae",
            confidence=0.95,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed + method_index,
        )
        ledgers = [run["methods"][method]["ledger"] for run in runs]
        n_expectations = sum(int(run["n_test_items"]) for run in runs)
        ledger = {
            "B_train": sum(int(value["B_train"]) for value in ledgers),
            "B_extra": sum(int(value["B_extra"]) for value in ledgers),
            "B_pred": sum(int(value["B_pred"]) for value in ledgers),
            "nominal_total": sum(
                int(value.get("nominal_total", value["B_train"] + value["B_pred"]))
                for value in ledgers
            ),
            "realized_total": sum(int(value["total"]) for value in ledgers),
            "nominal_test_total": sum(
                int(value.get("nominal_test_total", value["B_pred"])) for value in ledgers
            ),
            "realized_test_total": sum(
                int(value.get("realized_test_total", value["B_extra"] + value["B_pred"]))
                for value in ledgers
            ),
        }
        ledger["test_budget_ratio"] = circuit_evaluation_ratio(
            ledger["realized_test_total"], ledger["nominal_test_total"]
        )
        ledger["circuit_evals_per_mitigated_expectation"] = (
            ledger["realized_test_total"] / n_expectations
        )
        summaries[method] = {
            "metrics": metrics,
            "mae_interval": interval,
            "mean_rank": ranks[method],
            "ledger": ledger,
            "n_cells": len(records),
            "n_expectations": n_expectations,
        }
        cell_ids = [str(record["cell_id"]) for record in records]
        artifact_ids = sorted(
            {
                str(artifact_id)
                for record in records
                for artifact_id in record["source_artifact_ids"]
            }
        )
        trace[method] = {
            "cell_ids": cell_ids,
            "cells": [
                {"cell_id": record["cell_id"], "key": record["key"]}
                for record in records
            ],
            "artifact_ids": artifact_ids,
            "n_cells": len(records),
            "n_expectations": n_expectations,
            "metrics": {
                "macro_mae": metrics["mae"],
                "mae_circuit_bootstrap_interval": interval.__dict__,
                "mean_rank": ranks[method],
                "macro_rmse": metrics["rmse"],
                "excess_loss_mean": metrics["excess_loss_mean"],
                "overcorrection_rate": metrics["overcorrection_rate"],
            },
            "ledger": ledger,
        }

    return summaries, trace


def _merge_cell_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Merge source records into stable declared cells for report statistics."""

    if not records:
        raise ValueError("at least one cell record is required")
    source_artifact_ids = sorted({str(record["artifact_id"]) for record in records})
    source_set_payload = json.dumps(
        {
            "schema": "qem-bench-report-source-set-v1",
            "artifact_ids": source_artifact_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    report_artifact_id = "sha256:" + hashlib.sha256(source_set_payload).hexdigest()

    grouped: dict[tuple[object, ...], dict] = {}
    for record in records:
        artifact_id = str(record["artifact_id"])
        grouping = str(record["grouping"])
        key_fields = tuple(str(field) for field in record["key_fields"])
        key = dict(record["key"])
        signature = (
            grouping,
            key_fields,
            json.dumps(key, sort_keys=True, separators=(",", ":")),
        )
        merged = grouped.setdefault(
            signature,
            {
                "grouping": grouping,
                "key_fields": key_fields,
                "key": key,
                "source_artifact_ids": set(),
                "items": [],
            },
        )
        merged["source_artifact_ids"].add(artifact_id)
        for source_item in record["items"]:
            item = dict(source_item)
            item["artifact_id"] = artifact_id
            item["item_id"] = _source_scoped_id(artifact_id, item["item_id"])
            item["circuit_id"] = str(item["circuit_id"])
            merged["items"].append(item)

    merged_records = []
    for signature in sorted(grouped, key=repr):
        merged = grouped[signature]
        items = sorted(
            merged["items"],
            key=lambda item: (str(item["artifact_id"]), str(item["item_id"])),
        )
        predictions = np.asarray([item["prediction"] for item in items], dtype=float)
        raw_predictions = np.asarray(
            [item["raw_prediction"] for item in items], dtype=float
        )
        targets = np.asarray([item["target"] for item in items], dtype=float)
        cell_payload = json.dumps(
            {
                "grouping": merged["grouping"],
                "key_fields": list(merged["key_fields"]),
                "key": merged["key"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        merged_records.append(
            {
                "cell_id": "sha256:" + hashlib.sha256(cell_payload).hexdigest(),
                "artifact_id": report_artifact_id,
                "source_artifact_ids": sorted(merged["source_artifact_ids"]),
                "grouping": merged["grouping"],
                "key_fields": list(merged["key_fields"]),
                "key": merged["key"],
                "metrics": pooled_method_metrics(
                    predictions, raw_predictions, targets
                ),
                "circuit_ids": sorted({item["circuit_id"] for item in items}),
                "item_ids": [item["item_id"] for item in items],
                "items": items,
            }
        )
    return merged_records


def _source_scoped_id(artifact_id: str, identifier: object) -> str:
    return f"{artifact_id}/{identifier}"


def _load_runs(
    manifest: Mapping[str, object],
    base_dir: Path,
    *,
    source_runs: list[dict] | None = None,
) -> list[dict]:
    if manifest.get("schema_version") != "qem-bench-report-manifest-v1":
        raise ValueError("unsupported report manifest schema_version")
    entries = manifest.get("runs", [])
    if not isinstance(entries, list):
        raise ValueError("report manifest runs must be an array")
    resolved_base = base_dir.resolve()
    runs = []
    artifact_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("report manifest run entries must be objects")
        declared = entry.get("results")
        if not isinstance(declared, str) or not declared:
            raise ValueError("report manifest run results must be nonempty paths")
        relative_path = Path(declared)
        if relative_path.is_absolute() or relative_path.drive:
            raise ValueError("report manifest run results paths must be relative")
        display_path = relative_path.as_posix()
        try:
            resolved_path = (resolved_base / relative_path).resolve()
            resolved_path.relative_to(resolved_base)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"report run result path escapes manifest directory: {display_path}"
            ) from exc
        try:
            payload = resolved_path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"invalid runner result at {display_path}: unable to read result"
            ) from exc
        try:
            run = validate_run_artifact(json.loads(payload))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid runner result at {display_path}: {exc}"
            ) from exc
        artifact_id = str(run["artifact_id"])
        if artifact_id in artifact_ids:
            raise ValueError(
                f"report manifest contains duplicate run artifact_id {artifact_id!r}"
            )
        artifact_ids.add(artifact_id)
        runs.append(run)
        if source_runs is not None:
            source_runs.append(
                {
                    "path": display_path,
                    "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                    "artifact_id": artifact_id,
                }
            )
    if not runs:
        raise ValueError("report manifest contains no runs")
    schema_versions = {
        str(run["dataset_schema_version"])
        for run in runs
    }
    if len(schema_versions) != 1:
        raise ValueError(
            "report manifest cannot mix dataset schema versions; "
            f"found {sorted(schema_versions)}"
        )
    profiles_by_family: dict[str, list[str]] = {}
    for run in runs:
        for family, profile in run_identity_encoding_profiles(run).items():
            profiles_by_family.setdefault(family, []).append(profile)
    for family, profiles in sorted(profiles_by_family.items()):
        if len(profiles) < 2:
            continue
        if (
            UNKNOWN_IDENTITY_ENCODING_PROFILE in profiles
            or len(set(profiles)) > 1
        ):
            raise ValueError(
                "report manifest cannot merge physical identity encoding "
                f"profiles for family {family!r}; found {sorted(profiles)}"
            )
    return runs


def _table_rows(summaries: Mapping[str, Mapping[str, object]]) -> list[str]:
    rows = []
    for method, summary in summaries.items():
        metrics = summary["metrics"]
        interval = summary["mae_interval"]
        ledger = summary["ledger"]
        test_ratio = ledger["test_budget_ratio"]
        test_ratio_text = (
            "undefined" if test_ratio is None else f"{test_ratio:.1f}x"
        )
        rows.append(
            " & ".join(
                [
                    _latex_escape(METHOD_LABELS.get(method, method)),
                    str(summary["n_cells"]),
                    f"{metrics['mae']['mean']:.4f} [{interval.lower:.4f}, {interval.upper:.4f}]",
                    f"[{metrics['mae']['q1']:.4f}, {metrics['mae']['q3']:.4f}]",
                    f"{summary['mean_rank']:.2f}",
                    f"{metrics['rmse']['mean']:.4f}",
                    f"{metrics['excess_loss_mean']['mean']:.4f}",
                    f"{metrics['overcorrection_rate']['mean']:.3f}",
                    f"{ledger['B_train']:,}",
                    f"{ledger['B_extra']:,}",
                    f"{ledger['B_pred']:,}",
                    f"{ledger['nominal_total']:,}",
                    f"{ledger['realized_total']:,}",
                    test_ratio_text,
                    f"{ledger['circuit_evals_per_mitigated_expectation']:.1f}",
                ]
            )
            + r" \\"
        )
    return rows


def _report_description(runs: Sequence[dict]) -> tuple[str, str]:
    if runs[0]["dataset_schema_version"] == "legacy-v1":
        return (
            "Legacy-v1 walking-skeleton results",
            "These legacy-v1 runs have no declared S0 to S6 split axis.",
        )
    split_ids = sorted({_run_split_id(run) for run in runs})
    domains = ["S0 is in-distribution."] if "S0" in split_ids else []
    ood = [split_id for split_id in split_ids if split_id != "S0"]
    if ood:
        verb = "are" if len(ood) > 1 else "is"
        domains.append(", ".join(ood) + f" {verb} out-of-distribution.")
    return (
        "Split-v2 results (" + ", ".join(split_ids) + ")",
        "Rows are grouped by the declared split ID. " + " ".join(domains)
        + " Validation uses the source domain.",
    )


_CAPTION_PRESET_LIMIT = 8
_TABLE_COLUMNS = r"lrrrrrrrrrrrrrr"
_TABLE_HEADER = (
    r"Method & Cells & Macro MAE [95\% CI] & MAE IQR & Rank & Macro RMSE "
    r"& Harm mean & OCR & $B_{train}$ & $B_{extra}$ & $B_{pred}$ "
    r"& Nominal total & Realized total & Test ratio & Eval./expect. \\"
)


def _table_tex(sections: Sequence[tuple[str, dict]], runs: Sequence[dict]) -> str:
    description, split_note = _report_description(runs)
    stratum_note = (
        "The headline includes only continuous regression. Clifford controls "
        "are reported separately and excluded from all headline statistics and costs."
    )
    if not any(run["stratum"] == "continuous_regression" for run in runs):
        stratum_note = (
            "Only Clifford controls were supplied; "
            "no continuous-regression headline is available."
        )
    artifact_text = ", ".join(run["artifact_id"][7:19] for run in runs)
    # The caption names the distinct presets, not one descriptor per run. A
    # campaign of a few hundred runs is ordinary, and every custom-configured
    # run contributed the same twenty-character phrase: at 160 runs the caption
    # grew until pdflatex failed with "Dimension too large" and emitted no PDF,
    # before body pagination could help. Beyond the display limit the caption
    # carries a count; per-run identity stays in the artifact note and the trace.
    presets = list(dict.fromkeys(
        str(run["preset"]) if run["preset"] is not None else "custom configuration"
        for run in runs
    ))
    preset_text = ", ".join(presets[:_CAPTION_PRESET_LIMIT])
    if len(presets) > _CAPTION_PRESET_LIMIT:
        preset_text += f", and {len(presets) - _CAPTION_PRESET_LIMIT} more"
    # One float per section. A single unbreakable float silently drops whatever
    # does not fit the page: resizebox scales width only, and a table float cannot
    # break, so an oversized body is clipped while pdflatex still exits 0. That
    # defeats the stratum and split separation this report exists to show, so each
    # section is placed on its own with the column header repeated.
    caption = (
        f"{_latex_escape(description)}. Run presets: {_latex_escape(preset_text)}. "
        "MAE confidence intervals use a circuit-blocked bootstrap. Nominal and "
        "realized circuit-evaluation totals are reported; budgets are not "
        f"equalized. {_latex_escape(stratum_note)}"
    )
    footer = (
        rf"\par\vspace{{2pt}}\raggedright\tiny Run artifacts: "
        rf"{_latex_escape(artifact_text)}. Full cell identifiers and numeric "
        rf"sources are in \texttt{{report-trace.json}}. {_latex_escape(split_note)}"
    )
    blocks = []
    for index, (label, summaries) in enumerate(sections):
        body = chr(10).join(_table_rows(summaries))
        if index == 0:
            section_caption = f"{caption} Section: {_latex_escape(label)}."
        else:
            section_caption = (
                rf"{_latex_escape(description)} (continued). "
                rf"Section: {_latex_escape(label)}."
            )
        tail = footer if index == len(sections) - 1 else ""
        blocks.append(rf"""\begin{{table}}[ht]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2.2pt}}
\caption{{{section_caption}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{{_TABLE_COLUMNS}}}
{_TABLE_HEADER}
\hline
{body}
\hline
\end{{tabular}}}}
{tail}
\end{{table}}""")
    return rf"""\documentclass{{article}}
\usepackage[margin=0.35in,landscape]{{geometry}}
\usepackage{{graphicx}}
\begin{{document}}
{(chr(10) + chr(10)).join(blocks)}
\end{{document}}
"""


_FIGURE_WIDTH_IN = 10.0
_FIGURE_PANEL_HEIGHT_IN = 3.8
_FIGURE_TITLE_HEIGHT_IN = 0.5
_FIGURE_NOTE_HEIGHT_IN = 0.3
_LEGEND_COLUMNS = 4
_LEGEND_ROW_HEIGHT_IN = 0.3
_LEGEND_PAD_IN = 0.4


def _method_cost(run: Mapping[str, object], method: str) -> float:
    ledger = run["methods"][method]["ledger"]
    return float(
        ledger.get(
            "circuit_evals_per_mitigated_expectation",
            (ledger["B_extra"] + ledger["B_pred"]) / run["n_test_items"],
        )
    )


def _figure_pages(runs: Sequence[dict]) -> list[tuple[str | None, list[dict]]]:
    """Group runs into one page per declared split.

    Color and marker carry the method, so the split ID reaches the reader only
    through the legend text. Overlaying every split on one page multiplies the
    legend entries by the number of splits, and the legend then outgrows the
    panels it labels: at eight methods and three splits it covers the panels and
    the report title, and matplotlib reports nothing. One page per split holds
    the legend at one entry per method. The cost is that splits are compared
    across pages; the trace records the page of every point.
    """

    pages: dict[str | None, list[dict]] = {}
    for run in runs:
        pages.setdefault(_run_split_id(run), []).append(run)
    return sorted(pages.items(), key=lambda page: page[0] or "")


def _figure_page(
    page_runs: Sequence[dict],
    methods: Sequence[str],
    geometry: Mapping[str, object],
    trace: dict,
    page_number: int,
):
    """Draw one page: every stratum present in these runs, for one split."""

    from matplotlib import pyplot as plt

    colors = {"raw": "#555555", "ridge": "#2b6cb0", "zne": "#c05621"}
    markers = {"raw": "o", "ridge": "s", "zne": "^"}
    strata = [
        stratum for stratum in STRATUM_LABELS
        if any(run["stratum"] == stratum for run in page_runs)
    ]
    # The legend, note and title bands are sized from the roster before the
    # panels are placed, so the panels never share space with them however many
    # methods are reported. Fixing the column count bounds the legend width as
    # well: a wider roster adds rows, and the page grows to hold them.
    legend_rows = -(-len(methods) // _LEGEND_COLUMNS)
    legend_height = legend_rows * _LEGEND_ROW_HEIGHT_IN + _LEGEND_PAD_IN
    undefined = any(
        float(run["methods"]["raw"]["metrics"]["mae"]) == 0.0 for run in page_runs
    )
    note_height = _FIGURE_NOTE_HEIGHT_IN if undefined else 0.0
    height = (
        _FIGURE_PANEL_HEIGHT_IN * len(strata)
        + legend_height
        + note_height
        + _FIGURE_TITLE_HEIGHT_IN
    )
    figure, panels = plt.subplots(
        len(strata), 2, figsize=(_FIGURE_WIDTH_IN, height), squeeze=False
    )
    zero_control_x = float(geometry["zero_control_x"])
    for run in page_runs:
        axes = panels[strata.index(run["stratum"])]
        raw_mae = float(run["methods"]["raw"]["metrics"]["mae"])
        for method in methods:
            spec = run["methods"][method]
            x = _method_cost(run, method)
            plot_x = zero_control_x if x == 0 else x
            relative_mae = (
                float(spec["metrics"]["mae"]) / raw_mae if raw_mae else None
            )
            harm = float(spec["metrics"]["excess_loss_mean"])
            axes[0].scatter(
                plot_x,
                relative_mae if relative_mae is not None else np.nan,
                color=colors.get(method),
                marker=markers.get(method, "o"),
                alpha=0.75,
                label=METHOD_LABELS.get(method, method),
            )
            axes[1].scatter(
                plot_x,
                harm,
                color=colors.get(method),
                marker=markers.get(method, "o"),
                alpha=0.75,
            )
            trace["figure1"]["points"].append(
                {
                    "artifact_id": run["artifact_id"],
                    "preset": run["preset"],
                    "stratum": run["stratum"],
                    "split_id": _run_split_id(run),
                    "page": page_number,
                    "method": method,
                    "cell_ids": [record["cell_id"] for record in spec["cell_records"]],
                    "circuit_evals_per_mitigated_expectation": x,
                    "plot_x": plot_x,
                    "zero_cost_control": x == 0,
                    "relative_macro_mae": relative_mae,
                    "excess_loss_mean": harm,
                }
            )
    ticks = list(geometry["ticks"])
    x_labels = dict(geometry["x_labels"])
    for axis in panels.flat:
        axis.set_xscale("log")
        axis.set_xticks(ticks, [x_labels[value] for value in ticks])
        axis.minorticks_off()
        axis.grid(True, which="both", color="0.9", linewidth=0.6)
        label = "Circuit evaluations per mitigated expectation"
        if geometry["has_zero_cost"]:
            label += "\nZero-cost controls occupy the separate left column"
        axis.set_xlabel(label)
    for stratum, axes in zip(strata, panels, strict=True):
        axes[0].axhline(1.0, color="0.6", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("Macro MAE relative to raw")
        axes[0].set_title(
            STRATUM_LABELS[stratum] + "\nAccuracy-cost frontier", fontsize=10
        )
        axes[1].set_ylabel("Macro excess absolute loss")
        axes[1].set_title(
            STRATUM_LABELS[stratum] + "\nHarm-cost frontier", fontsize=10
        )
    figure.suptitle(str(geometry["title"]), fontsize=10)
    handles, labels = panels[0][0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.4 * _LEGEND_PAD_IN / height),
        ncol=_LEGEND_COLUMNS,
        fontsize=8,
    )
    if undefined:
        figure.text(
            0.5,
            (legend_height + 0.35 * note_height) / height,
            "Relative MAE is undefined when raw MAE is zero; "
            "see the harm panel for those points.",
            ha="center", fontsize=8,
        )
    figure.tight_layout(
        rect=(
            0.0,
            (legend_height + note_height) / height,
            1.0,
            1.0 - _FIGURE_TITLE_HEIGHT_IN / height,
        )
    )
    return figure


def _figure_pdf(
    runs: Sequence[dict], methods: Sequence[str], path: Path, trace: dict
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    costs = [_method_cost(run, method) for run in runs for method in methods]
    if any(cost < 0 for cost in costs):
        raise ValueError("circuit-evaluation costs must be nonnegative")
    positive_costs = [cost for cost in costs if cost > 0]
    zero_control_x = min(positive_costs) / 4.0 if positive_costs else 1.0
    # Ticks are collected over every run rather than per page, so the pages
    # share one x axis and a cost read off one page means the same on the next.
    x_labels: dict[float, str] = {}
    for cost in costs:
        plot_x = zero_control_x if cost == 0 else cost
        x_labels[plot_x] = "0\n(control)" if cost == 0 else f"{cost:,.0f}"
    description, _ = _report_description(runs)
    geometry = {
        "zero_control_x": zero_control_x,
        "has_zero_cost": any(cost == 0 for cost in costs),
        "ticks": sorted(x_labels),
        "x_labels": x_labels,
    }
    pages = _figure_pages(runs)
    with PdfPages(path) as pdf:
        for page_number, (split_id, page_runs) in enumerate(pages, start=1):
            title = description + ": budgets are not equalized"
            if split_id is not None:
                title += (
                    f"\nPage {page_number} of {len(pages)}: split {split_id}"
                )
            figure = _figure_page(
                page_runs, methods, {**geometry, "title": title},
                trace, page_number,
            )
            pdf.savefig(figure)
            plt.close(figure)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)
