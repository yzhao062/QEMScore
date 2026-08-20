"""Generate Table 1, Figure 1, and their artifact/cell trace."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from qem_bench.runner.metrics import pooled_method_metrics
from qem_bench.runner.run import circuit_evaluation_ratio, validate_run_artifact
from qem_bench.stats import circuit_blocked_bootstrap, macro_mean_iqr, mean_ranks

HEADLINE_METHODS = ("raw", "ridge", "zne")
METHOD_LABELS = {"raw": "Raw", "ridge": "Ridge", "zne": "Local digital ZNE"}


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
    trace = {
        "schema_version": "qem-bench-report-trace-v1",
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": (
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        ),
        "source_runs": source_runs,
        "artifacts": [run["artifact_id"] for run in runs],
        "table1": {},
        "figure1": {"points": []},
    }
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
        trace["table1"][method] = {
            "cell_ids": cell_ids,
            "artifact_ids": artifact_ids,
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

    table_path = output / "table1.tex"
    table_path.write_text(_table_tex(summaries, runs), encoding="utf-8")
    figure_path = output / "figure1.pdf"
    _figure_pdf(runs, methods, figure_path, trace)
    trace_path = output / "report-trace.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return {"table1": table_path, "figure1": figure_path, "trace": trace_path}


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
    return runs


def _table_tex(summaries: Mapping[str, Mapping[str, object]], runs: Sequence[dict]) -> str:
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
    artifact_text = ", ".join(run["artifact_id"][7:19] for run in runs)
    preset_text = ", ".join(str(run["preset"]) for run in runs)
    return rf"""\documentclass{{article}}
\usepackage[margin=0.35in,landscape]{{geometry}}
\usepackage{{graphicx}}
\begin{{document}}
\begin{{table}}[ht]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2.2pt}}
\caption{{Walking-skeleton results on shipped presets ({_latex_escape(preset_text)}). MAE confidence intervals use a circuit-blocked bootstrap. Nominal and realized totals expose current budget inequality.}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrrrrrrrr}}
Method & Cells & Macro MAE [95\% CI] & MAE IQR & Rank & Macro RMSE & Harm mean & OCR & $B_{{train}}$ & $B_{{extra}}$ & $B_{{pred}}$ & Nominal total & Realized total & Test ratio & Eval./expect. \\
\hline
{chr(10).join(rows)}
\end{{tabular}}}}
\par\vspace{{2pt}}\raggedright\tiny Run artifacts: {_latex_escape(artifact_text)}. Full cell identifiers and numeric provenance are in \texttt{{report-trace.json}}. These shipped presets do not yet implement the S0 to S6 grammar.
\end{{table}}
\end{{document}}
"""


def _figure_pdf(
    runs: Sequence[dict], methods: Sequence[str], path: Path, trace: dict
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    colors = {"raw": "#555555", "ridge": "#2b6cb0", "zne": "#c05621"}
    markers = {"raw": "o", "ridge": "s", "zne": "^"}
    costs = [
        float(
            run["methods"][method]["ledger"].get(
                "circuit_evals_per_mitigated_expectation",
                (
                    run["methods"][method]["ledger"]["B_extra"]
                    + run["methods"][method]["ledger"]["B_pred"]
                )
                / run["n_test_items"],
            )
        )
        for run in runs
        for method in methods
    ]
    if any(cost < 0 for cost in costs):
        raise ValueError("circuit-evaluation costs must be nonnegative")
    positive_costs = [cost for cost in costs if cost > 0]
    zero_control_x = min(positive_costs) / 4.0 if positive_costs else 1.0
    has_zero_cost = any(cost == 0 for cost in costs)
    x_values: set[float] = set()
    x_labels: dict[float, str] = {}
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for run in runs:
        raw_mae = float(run["methods"]["raw"]["metrics"]["mae"])
        for method in methods:
            spec = run["methods"][method]
            ledger = spec["ledger"]
            x = float(
                ledger.get(
                    "circuit_evals_per_mitigated_expectation",
                    (ledger["B_extra"] + ledger["B_pred"]) / run["n_test_items"],
                )
            )
            plot_x = zero_control_x if x == 0 else x
            x_values.add(plot_x)
            x_labels[plot_x] = "0\n(control)" if x == 0 else f"{x:,.0f}"
            relative_mae = float(spec["metrics"]["mae"]) / raw_mae if raw_mae else np.nan
            harm = float(spec["metrics"]["excess_loss_mean"])
            axes[0].scatter(
                plot_x,
                relative_mae,
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
                    "method": method,
                    "cell_ids": [record["cell_id"] for record in spec["cell_records"]],
                    "circuit_evals_per_mitigated_expectation": x,
                    "plot_x": plot_x,
                    "zero_cost_control": x == 0,
                    "relative_macro_mae": relative_mae,
                    "excess_loss_mean": harm,
                }
            )
    for axis in axes:
        axis.set_xscale("log")
        ticks = sorted(x_values)
        axis.set_xticks(ticks, [x_labels[value] for value in ticks])
        axis.minorticks_off()
        axis.grid(True, which="both", color="0.9", linewidth=0.6)
        label = "Circuit evaluations per mitigated expectation"
        if has_zero_cost:
            label += "\nZero-cost controls occupy the separate left column"
        axis.set_xlabel(label)
    axes[0].axhline(1.0, color="0.6", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Macro MAE relative to raw")
    axes[0].set_title("Accuracy-cost frontier")
    axes[1].set_ylabel("Macro excess absolute loss")
    axes[1].set_title("Harm-cost frontier")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axes[0].legend(unique.values(), unique.keys(), fontsize=8)
    figure.suptitle("Shipped presets: current unequal walking-skeleton budgets", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
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
