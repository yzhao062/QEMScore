# Protocol Trace

Active rows cover only frozen rules that have direct executable guards. Open design
decisions and scientific-judgment claims are intentionally absent. IDs are stable and
retired IDs are never reused. No public paper authority for these rules is tracked in
this repository yet; the implementation and guarding-test columns identify the current
public executable contracts.

<!-- protocol-trace:begin -->
| Rule ID | Authority | Protocol Rule | Implementation | Guarding Test |
|---|---|---|---|---|
| QEM-P001 | No public paper authority exists yet. | Headline statistics use equal-weight macro cells rather than pooled items. | `qem_bench/runner/metrics.py::headline_metrics`<br>`qem_bench/stats/descriptive.py::macro_mean_iqr` | `tests/test_metrics_stats_reports.py::test_pooled_diagnostic_differs_from_macro_on_unequal_cells` |
| QEM-P002 | No public paper authority exists yet. | The hierarchical bootstrap resamples physical circuits as top-level blocks and keeps every severity row for a sampled circuit together. | `qem_bench/stats/bootstrap.py::circuit_blocked_bootstrap` | `tests/test_metrics_stats_reports.py::test_bootstrap_blocks_split_physical_circuits_across_severities` |
| QEM-P003 | No public paper authority exists yet. | OOD validation remains in the source domain. | `qem_bench/validation.py::validate_axis_contract` | `tests/test_splits.py::test_source_only_validation_is_enforced` |
| QEM-P004 | No public paper authority exists yet. | Dataset item rows are closed to the declared circuit family and dataset schema version, including after complete artifact rehashing. | `qem_bench/datasets/schema.py::validate_item` | `tests/test_schema_closure.py::test_rehashed_row_rejects_foreign_family_field`<br>`tests/test_schema_closure.py::test_rehashed_legacy_row_rejects_split_only_field` |
| QEM-P005 | No public paper authority exists yet. | Run validation enforces exact nonnegative ledger buckets and conservation of every derived ledger field. | `qem_bench/runner/run.py::_validate_ledger` | `tests/test_metrics_stats_reports.py::test_run_validator_rejects_resigned_invalid_ledgers` |
| QEM-P006 | No public paper authority exists yet. | Every split row seed is recoverable from the manifest master seed and named spawn formula. | `qem_bench/validation.py::_validate_item_semantics` | `tests/test_validation_split_regressions.py::test_split_manifest_seed_contract_is_recomputed` |
<!-- protocol-trace:end -->
