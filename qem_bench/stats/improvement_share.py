"""Joint attribution of a test-set improvement across a ladder of predictors.

The gain contrast in :mod:`qem_bench.stats.gain_contrast` estimates one ratio
between two methods. The question here has three predictors and an additive
decomposition rather than a ratio. With A the affine feature-only error, C the
nonlinear feature-only error and F the full nonlinear error, all measured on
identical rows with identical aggregation,

    T = A - F      K = A - C      D = C - F      T = K + D      S = K / T

so T is the improvement the full method makes over the affine control, K is the
part a nonlinear feature-only control already reproduces, D is the part that
remains once the noisy observation enters, and S is the classical share of T.

The controlled instrument this module serves is frozen as:

    At the shipped evolution steps, we will quantify how much improvement over
    an affine feature-only control is reproduced by a nonlinear feature-only
    control, and how much remains from adding the noisy observation, using
    paired test-set MAE decompositions and uncertainty intervals for both
    spin-chain families at each of three declared seeds.

The core takes an ordered ladder of two or more rungs rather than a fixed A, C
and F. A second half of the same estimand needs a four-rung nested ladder, and
writing two estimators would duplicate the bootstrap, the pairing and the
denominator rule. The ladder itself is declared by the campaign layer, so this
module never fixes which methods occupy which rung.

Every rung, every declared span and every reference method shares one
circuit-index resample per family, which is what makes the decomposition
paired: a drawn physical circuit carries all of its severity and observable
rows into every quantity of that draw. S is formed inside each draw from that
draw's macro errors, never as an average of per-row ratios and never clipped.
No draw is discarded. When the denominator cannot carry a fraction, the share
is withheld and the denominator diagnostics say why, while the rung errors, the
total and every gap keep their intervals, because a negative gap is a result
rather than a failure.

What the intervals do not carry: they condition on the fitted and selected
models, so training and model-selection variation stays outside them, and they
are pointwise rather than simultaneous over settings, families or quantities.
The declared seeds describe training-draw variation separately.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json
import math

import numpy as np

from qem_bench.datasets.schema import FAMILY_STRATA
# The single data constant is shared so the two modules cannot drift on what a
# cell is. Nothing else is imported from the gain contrast: its estimator, its
# table builder and its bootstrap encode a two-method ratio and must not be
# pressed into service as a three-predictor estimator.
from qem_bench.stats.gain_contrast import CELL_FIELDS

SCHEMA_VERSION = "qem-bench-improvement-share-v1"

# A family-level refusal withholds every interval of that family. A share-level
# refusal withholds the share alone, because the reviewer requires the absolute
# quantities retained whichever way the fraction falls.
FAMILY_FAILURE_REASONS = (
    "family_missing_from_the_setting",
    "fewer_than_two_circuits",
    "empty_cell_in_a_draw",
    "nonfinite_macro_in_a_draw",
)
SHARE_FAILURE_REASONS = (
    "nonpositive_point_total",
    "nonfinite_point_share",
    "nonpositive_total_in_a_draw",
    "nonfinite_share_in_a_draw",
    "total_interval_reaches_zero",
)
FAILURE_REASONS = FAMILY_FAILURE_REASONS + SHARE_FAILURE_REASONS

# Draws are evaluated in blocks. The index matrix is drawn in one call and only
# the evaluation is chunked, so no published number depends on this constant.
_EVAL_CHUNK = 512
_EPS = 2.220446049250313e-16
# The identity T = sum(gaps) is exact in real arithmetic and holds to a few
# rounding steps in binary64. Thirty-two epsilons of the largest macro error
# leave room for the summation without admitting a real aggregation mismatch.
_IDENTITY_TOLERANCE_FACTOR = 32.0
_RESERVED_LABELS = ("total", "share")
# What a caller may say a decomposition is for. `rehearsal` is here because a
# run that can never be published needs a role of its own: labelling it
# `secondary` would put it in the same class as a reported-but-not-headline
# result, and labelling it `primary` is what let a whole grid read as the
# committed one. The caller assigns the value; this list only bounds it.
_SHARE_ROLES = ("primary", "secondary", "rehearsal")


@dataclass(frozen=True, eq=False)
class LadderTable:
    """One family's validated per-circuit, per-cell counts and summed errors.

    ``counts`` is a single array shared by every rung. That is the structural
    form of the requirement to compute all errors on identical rows with
    identical aggregation: one count array per rung would let the row sets drift
    apart while every other check still passed.

    ``errors[m][c, k]`` is the sum of absolute errors of method ``m`` over the
    rows of circuit ``c`` in cell ``k``, and ``counts[c, k]`` is the number of
    those rows. Comparison is disabled because the numpy fields make the
    generated ``__eq__`` and ``__hash__`` ambiguous rather than useful.
    """

    family: str
    circuit_ids: tuple[str, ...]
    cell_keys: tuple[tuple[str, ...], ...]
    counts: np.ndarray
    errors: Mapping[str, np.ndarray]
    n_rows: int
    # Carried rather than assumed: an adapter may key cells by its own fields,
    # and equal-weight aggregation makes that choice part of the estimand.
    cell_fields: tuple[str, ...] = CELL_FIELDS

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cell_fields", _distinct_names(self.cell_fields, "cell_fields")
        )
        object.__setattr__(self, "circuit_ids", tuple(self.circuit_ids))
        object.__setattr__(
            self, "cell_keys", tuple(tuple(key) for key in self.cell_keys)
        )
        object.__setattr__(self, "errors", MappingProxyType(dict(self.errors)))
        _validate_table(self)

    @classmethod
    def from_cell_errors(
        cls,
        cells: Mapping[tuple[str, tuple[str, ...]], Mapping[str, object]],
        *,
        family: str,
        methods: Sequence[str],
        cell_fields: Sequence[str] = CELL_FIELDS,
    ) -> "LadderTable":
        """Build a table from errors a caller already aggregated.

        ``cells`` maps ``(unit_id, cell_key)`` to
        ``{"count": rows, "errors": {method: summed_absolute_error}}``. The
        nested form is used rather than a flat one so that a rung may carry any
        name without colliding with the count. A pair absent from ``cells`` is
        an empty cell of that unit, which validation admits; a builder that
        needs a complete grid has to refuse an incomplete one itself, because
        an incomplete grid only surfaces here as a family-level refusal once
        some draw misses the single unit that populates a cell.

        This constructor never accepts raw rows, so the strict row path in
        :func:`build_ladder_tables` has no bypass.
        """

        declared = _distinct_names(methods, "methods")
        if not isinstance(cells, Mapping) or not cells:
            raise ValueError("from_cell_errors requires a nonempty mapping of cells")
        parsed: dict[tuple[str, tuple[str, ...]], tuple[float, dict[str, float]]] = {}
        units: set[str] = set()
        keys: set[tuple[str, ...]] = set()
        for identifier, value in cells.items():
            if not isinstance(identifier, tuple) or len(identifier) != 2:
                raise ValueError("cells must be keyed by (unit id, cell key) pairs")
            unit, cell = identifier
            if not isinstance(unit, str) or not unit:
                raise ValueError("cells require a nonempty unit id")
            cell = tuple(cell)
            if not cell or any(not isinstance(part, str) or not part for part in cell):
                raise ValueError("a cell key must hold nonempty strings")
            if not isinstance(value, Mapping) or set(value) != {"count", "errors"}:
                raise ValueError("each cell needs exactly a count and an errors map")
            errors = value["errors"]
            if not isinstance(errors, Mapping) or set(errors) != set(declared):
                raise ValueError("each cell must carry exactly the declared methods")
            parsed[(unit, cell)] = (
                float(value["count"]),
                {method: float(errors[method]) for method in declared},
            )
            units.add(unit)
            keys.add(cell)

        circuit_ids = tuple(sorted(units))
        cell_keys = tuple(sorted(keys))
        shape = (len(circuit_ids), len(cell_keys))
        circuit_index = {name: index for index, name in enumerate(circuit_ids)}
        cell_index = {key: index for index, key in enumerate(cell_keys)}
        counts = np.zeros(shape, dtype=float)
        errors = {method: np.zeros(shape, dtype=float) for method in declared}
        for (unit, cell), (count, values) in parsed.items():
            row = circuit_index[unit]
            column = cell_index[cell]
            counts[row, column] = count
            for method in declared:
                errors[method][row, column] = values[method]
        total = float(counts.sum())
        if not math.isfinite(total) or total != round(total):
            raise ValueError("cell counts must be whole numbers")
        return cls(
            family=_nonempty(family, "family"),
            circuit_ids=circuit_ids,
            cell_keys=cell_keys,
            counts=counts,
            errors=errors,
            n_rows=int(total),
            cell_fields=cell_fields,
        )


def build_ladder_tables(
    items: Sequence[Mapping[str, object]],
    predictions_by_method: Mapping[str, Mapping[str, float]],
    *,
    methods: Sequence[str],
    cell_fields: Sequence[str] = CELL_FIELDS,
) -> dict[str, LadderTable]:
    """Turn untouched split-v2 test rows into one validated table per family.

    This path is strict and carries no relaxation parameter. Every rung scores
    the same rows, so a prediction map that misses a row or carries an unknown
    identifier raises rather than quietly shrinking one rung's row set. Rows
    outside the continuous regression stratum are skipped; a family is built
    only from its own continuous rows.
    """

    declared = _distinct_names(methods, "methods")
    fields = _distinct_names(cell_fields, "cell_fields")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("build_ladder_tables requires a sequence of test rows")
    if not isinstance(predictions_by_method, Mapping):
        raise ValueError("predictions must be keyed by method")
    for method in declared:
        if method not in predictions_by_method:
            raise ValueError(f"predictions lack the ladder rung {method!r}")
        if not isinstance(predictions_by_method[method], Mapping):
            raise ValueError("predictions must be keyed by test item ID")

    rows_by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    identities: set[str] = set()
    circuit_families: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.get("split") != "test":
            raise ValueError("the improvement share requires untouched test rows only")
        if item.get("dataset_schema_version") != "split-v2":
            raise ValueError("untouched test rows require split-v2 metadata")
        item_id = _identity(item, "item_id")
        if item_id in identities:
            raise ValueError("test item IDs must be unique within a setting")
        identities.add(item_id)
        family = _identity(item, "family")
        if family not in FAMILY_STRATA or item.get("stratum") != FAMILY_STRATA[family]:
            raise ValueError("family and stratum must match the dataset contract")
        if item["stratum"] != "continuous_regression":
            continue
        circuit = _identity(item, "circuit_id")
        circuit_families[circuit].add(family)
        for field in fields:
            _identity(item, field)
        target = float(item["ideal_expectation"])
        if not math.isfinite(target):
            raise ValueError("test targets must be finite")
        rows_by_family[family].append(item)
    if any(len(names) != 1 for names in circuit_families.values()):
        raise ValueError("physical circuits must not span families")

    continuous_ids = {
        item["item_id"] for rows in rows_by_family.values() for item in rows
    }
    for method in declared:
        predictions = predictions_by_method[method]
        missing = continuous_ids - predictions.keys()
        if missing:
            raise ValueError(
                f"rung {method!r} lacks predictions for {len(missing)} test rows"
            )
        surplus = predictions.keys() - identities
        if surplus:
            raise ValueError(
                f"rung {method!r} carries {len(surplus)} identifiers "
                "absent from the test rows"
            )
        if any(not math.isfinite(float(predictions[key])) for key in continuous_ids):
            raise ValueError("predictions must be finite")

    tables: dict[str, LadderTable] = {}
    for family, rows in rows_by_family.items():
        circuit_ids = tuple(sorted({item["circuit_id"] for item in rows}))
        cell_keys = tuple(sorted(
            {tuple(item[field] for field in fields) for item in rows}
        ))
        circuit_index = {name: index for index, name in enumerate(circuit_ids)}
        cell_index = {key: index for index, key in enumerate(cell_keys)}
        shape = (len(circuit_ids), len(cell_keys))
        counts = np.zeros(shape, dtype=float)
        errors = {method: np.zeros(shape, dtype=float) for method in declared}
        for item in rows:
            row = circuit_index[item["circuit_id"]]
            column = cell_index[tuple(item[field] for field in fields)]
            counts[row, column] += 1.0
            for method in declared:
                errors[method][row, column] += _finite_absolute_error(
                    predictions_by_method[method][item["item_id"]],
                    item["ideal_expectation"],
                )
        tables[family] = LadderTable(
            family=family,
            circuit_ids=circuit_ids,
            cell_keys=cell_keys,
            counts=counts,
            errors=errors,
            n_rows=len(rows),
            cell_fields=fields,
        )
    return dict(sorted(tables.items()))


def draw_matrix(n_circuits: int, *, seed: int, n_resamples: int) -> np.ndarray:
    """One circuit-index resample. Shape (n_resamples, n_circuits), dtype int64.

    The whole matrix is drawn in a single call and only the evaluation is
    chunked afterwards. Chunking the drawing instead would tie every published
    interval to the value of a performance constant.

    The resample is public because it is an object of the contract: an oracle
    has to consume the same draws without touching any aggregation code.
    """

    if type(n_circuits) is not int or n_circuits < 1:
        raise ValueError("n_circuits must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(n_resamples) is not int or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    return np.random.default_rng(seed).integers(
        0, n_circuits, size=(n_resamples, n_circuits)
    )


def ladder_stream_seed(root_seed: int, components: Sequence[object]) -> int:
    """Derive one reproducible stream from the declared key components.

    A string component is digested rather than looked up in a declared tuple.
    An index-based key would move every published interval the moment a name is
    added to the declared family list, and that movement would be invisible.

    The training size is deliberately excluded from the key by the caller. Two
    sizes inside one regime share their test circuits, so giving them the same
    seed, the same circuit count and the same sorted circuit order gives them
    the same drawn circuits, which is what pairs the size comparison.
    """

    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a nonnegative integer")
    entropy = [root_seed]
    for component in components:
        entropy.append(_component_entropy(component))
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def estimate_improvement_share(
    tables: Mapping[str, LadderTable],
    *,
    ladder: Sequence[str],
    rung_labels: Sequence[str],
    gap_labels: Sequence[str],
    spans: Mapping[str, tuple[str, str]] = MappingProxyType({}),
    required_families: Sequence[str],
    setting_label: str,
    evaluation_role: str,
    share_role: str,
    share_interpretation: str,
    stream_components: Sequence[object],
    reference_methods: Sequence[str] = (),
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    root_seed: int,
) -> dict:
    """Decompose the improvement over the first rung across the ladder.

    ``ladder`` runs from the most restricted rung to the full rung and names
    methods; ``rung_labels``, ``gap_labels`` and the keys of ``spans`` name the
    reported quantities. Every gap, every span, every rung error and the share
    come from one circuit-index resample per family, so the decomposition is
    paired inside a setting.

    A required family absent from ``tables`` is reported as not estimable
    rather than omitted, and a family present in ``tables`` but undeclared
    raises: a partial result must never read as a complete one. The returned
    dictionary serializes under ``json.dumps(result, allow_nan=False)`` on
    every branch.
    """

    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if type(n_resamples) is not int or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a nonnegative integer")

    ladder = _distinct_names(ladder, "ladder")
    if len(ladder) < 2:
        raise ValueError("the ladder needs at least two rungs")
    rung_labels = _distinct_names(rung_labels, "rung_labels")
    if len(rung_labels) != len(ladder):
        raise ValueError("rung_labels must label every rung of the ladder")
    gap_labels = _distinct_names(gap_labels, "gap_labels")
    if len(gap_labels) != len(ladder) - 1:
        raise ValueError("gap_labels must label every adjacent gap of the ladder")
    quantity_labels = set(rung_labels) | set(gap_labels)
    if len(quantity_labels) != len(rung_labels) + len(gap_labels):
        raise ValueError("rung labels and gap labels must not collide")
    if quantity_labels & set(_RESERVED_LABELS):
        raise ValueError(f"{_RESERVED_LABELS} are reserved quantity names")
    reference_methods = _distinct_names(
        reference_methods, "reference_methods", allow_empty=True
    )
    if set(reference_methods) & set(ladder):
        raise ValueError("a reference method must not name a ladder rung")

    span_pairs = _span_pairs(spans, rung_labels, quantity_labels)

    _nonempty(setting_label, "setting_label")
    _nonempty(evaluation_role, "evaluation_role")
    if share_role not in _SHARE_ROLES:
        raise ValueError(f"share_role must be one of {_SHARE_ROLES}")
    _nonempty(share_interpretation, "share_interpretation")

    required = _distinct_names(required_families, "required_families")
    # The caller keeps its declared order, which the campaign design does not
    # sort, and the report gives the sorted order.
    required_sorted = tuple(sorted(required))
    components = tuple(stream_components)
    for component in components:
        _component_entropy(component)

    if not isinstance(tables, Mapping):
        raise ValueError("tables must map a family name to its ladder table")
    undeclared = sorted(set(tables) - set(required_sorted))
    if undeclared:
        raise ValueError(
            "every family in tables must be declared in required_families; "
            f"undeclared: {undeclared}"
        )
    for name, table in tables.items():
        if not isinstance(table, LadderTable):
            raise ValueError(f"family {name!r} must carry a LadderTable")
        if table.family != name:
            raise ValueError(f"family {name!r} carries a table for {table.family!r}")
        absent = [
            method
            for method in tuple(ladder) + tuple(reference_methods)
            if method not in table.errors
        ]
        if absent:
            raise ValueError(f"family {name!r} lacks errors for {absent}")

    # One report describes one aggregation, so two families cannot disagree
    # about what a cell is.
    declared_cell_fields = {table.cell_fields for table in tables.values()}
    if len(declared_cell_fields) > 1:
        raise ValueError("all family tables must use the same cell_fields")
    fields = next(iter(declared_cell_fields), tuple(CELL_FIELDS))

    alpha = 1.0 - confidence
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    result = {
        "schema_version": SCHEMA_VERSION,
        "setting": setting_label,
        "evaluation_role": evaluation_role,
        "ladder": list(ladder),
        "rung_labels": list(rung_labels),
        "gap_labels": list(gap_labels),
        "span_labels": list(span_pairs),
        # The machine-readable companion of span_definitions, so a contrast can
        # rebuild a span without parsing a rendered expression.
        "span_rungs": {
            label: [rung_labels[start], rung_labels[end]]
            for label, (start, end) in span_pairs.items()
        },
        "reference_methods": list(reference_methods),
        "total_definition": f"T = error({rung_labels[0]}) - error({rung_labels[-1]})",
        "gap_definitions": {
            label: f"error({rung_labels[index]}) - error({rung_labels[index + 1]})"
            for index, label in enumerate(gap_labels)
        },
        "span_definitions": {
            label: f"error({rung_labels[start]}) - error({rung_labels[end]})"
            for label, (start, end) in span_pairs.items()
        },
        "share_definition": (
            f"S = {gap_labels[0]} / T, formed inside each draw from that "
            "draw's macro errors"
        ),
        "share_role": share_role,
        "share_interpretation": share_interpretation,
        "aggregation": (
            "absolute error per row; pooled over rows inside each cell; "
            "equal weight over cells"
        ),
        "resample_unit": (
            "physical circuit, carrying every severity and observable row of "
            "that circuit"
        ),
        "pairing": (
            "every rung, span and reference method shares one circuit-index "
            "resample per family"
        ),
        "interval_method": "percentile, numpy linear interpolation, frozen",
        "interval_scope": (
            "pointwise; conditional on the fitted and selected models; paired "
            "across rungs within a setting; not simultaneous over settings, "
            "families or quantities; the three declared seeds describe "
            "training-draw variation separately"
        ),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "root_seed": root_seed,
        "stream_components": list(components),
        # Excluded by construction, not by the caller: see ladder_stream_seed.
        "stream_components_excluded": ["size"],
        "cell_fields": list(fields),
        "required_families": list(required_sorted),
        "status": "not_estimable",
        "reasons": [],
        "families": {},
    }

    reasons: set[str] = set()
    for family in required_sorted:
        family_result = _evaluate_family(
            tables.get(family),
            family=family,
            ladder=ladder,
            rung_labels=rung_labels,
            gap_labels=gap_labels,
            span_pairs=span_pairs,
            reference_methods=reference_methods,
            stream_components=components,
            root_seed=root_seed,
            n_resamples=n_resamples,
            quantiles=quantiles,
        )
        result["families"][family] = family_result
        # The top-level list is the union of the family-level and share-level
        # reasons, so a run whose only failure is a withheld share still reports
        # why rather than an empty list.
        reasons.update(family_result["reasons"])
        reasons.update(family_result["share"]["reasons"])

    estimated = all(
        value["status"] == "estimated" and value["share"]["status"] == "estimated"
        for value in result["families"].values()
    )
    result["reasons"] = sorted(reasons)
    result["status"] = "estimated" if estimated else "not_estimable"
    return result


def contrast_improvement_share(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    left_tables: Mapping[str, LadderTable],
    right_tables: Mapping[str, LadderTable],
    quantity: str,
    pairing: str,
    confidence: float = 0.95,
) -> dict:
    """Contrast one quantity between two settings, right minus left.

    ``left`` and ``right`` are results of :func:`estimate_improvement_share` and
    the two table mappings are the ones those calls consumed. Each side's draw
    matrix is rebuilt from the stream seed and resample count the result
    records, so the recomputation is reproducible from published fields alone,
    and a draw-by-draw contrast stays available where two summaries could not
    supply one.

    ``pairing`` is checked against the data rather than trusted. Two sizes in
    one regime share their test circuits and therefore their circuit order
    digest; two regimes redraw their pools and therefore cannot share it. Both
    mismatches raise, which turns a statistical design rule into an assertion
    the code makes about its own inputs.
    """

    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if pairing not in ("paired", "independent"):
        raise ValueError("pairing must be 'paired' or 'independent'")
    for name, side in (("left", left), ("right", right)):
        if not isinstance(side, Mapping) or side.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"both sides must be {SCHEMA_VERSION} results")
        if type(side.get("n_resamples")) is not int or side["n_resamples"] <= 0:
            raise ValueError(f"the {name} n_resamples must be a positive integer")
    # A cell definition, a weighting and a resample unit each pick out which
    # estimand the number is, so two sides that differ on any of them are two
    # different quantities rather than one quantity under two settings.
    for key in ("ladder", "rung_labels", "gap_labels", "span_rungs", "n_resamples",
                "cell_fields", "aggregation", "resample_unit"):
        if left[key] != right[key]:
            raise ValueError(f"the two sides disagree on {key}")

    ladder = tuple(left["ladder"])
    rung_labels = tuple(left["rung_labels"])
    gap_labels = tuple(left["gap_labels"])
    span_rungs = {
        label: tuple(pair) for label, pair in dict(left["span_rungs"]).items()
    }
    span_pairs = _span_pairs(span_rungs, rung_labels, set(rung_labels) | set(gap_labels))
    if quantity == "total":
        selector = ("total", None)
    elif quantity in gap_labels:
        selector = ("gaps", quantity)
    elif quantity in span_pairs:
        selector = ("spans", quantity)
    else:
        raise ValueError(f"{quantity!r} is not a reported gap, span or the total")

    n_resamples = left["n_resamples"]
    alpha = 1.0 - confidence
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    families = sorted(set(left["required_families"]) | set(right["required_families"]))
    contrast = {
        "schema_version": "qem-bench-improvement-share-contrast-v1",
        "quantity": quantity,
        "quantity_definition": _quantity_definition(left, quantity),
        "pairing": pairing,
        "pairing_check": (
            "table digests, circuit order digests, circuit counts and stream "
            "seeds compared on every contrasted family"
        ),
        "contrast_definition": "right setting minus left setting",
        "left_setting": left["setting"],
        "right_setting": right["setting"],
        # Carried so a reader of the contrast alone can see which estimand the
        # two sides agreed on, rather than having to fetch both results.
        "cell_fields": list(left["cell_fields"]),
        "aggregation": left["aggregation"],
        "resample_unit": left["resample_unit"],
        "confidence": confidence,
        "n_resamples": n_resamples,
        "interval_method": "percentile, numpy linear interpolation, frozen",
        "interval_scope": (
            "pointwise; conditional on the fitted and selected models; "
            f"{pairing} across the two settings"
        ),
        "families": {},
    }

    for family in families:
        sides = {}
        for name, result, tables in (
            ("left", left, left_tables), ("right", right, right_tables)
        ):
            record = result["families"].get(family)
            if record is None or record["status"] != "estimated":
                raise ValueError(
                    f"family {family!r} is not estimated on the {name} side"
                )
            # An imported result can carry a decimal string where the estimator
            # wrote an integer. int() would coerce it back and replay the very
            # stream the equality checks below exist to tell apart, so the value
            # is refused rather than repaired.
            for field in ("stream_seed", "n_circuits"):
                if type(record.get(field)) is not int or record[field] < 0:
                    raise ValueError(
                        f"the {name} {field} must be a nonnegative integer"
                    )
            table = tables.get(family) if isinstance(tables, Mapping) else None
            if not isinstance(table, LadderTable):
                raise ValueError(f"the {name} tables lack family {family!r}")
            # Every field the macro errors were built from, not the circuit names
            # alone. A result written before the table digest existed carries no
            # binding at all and is refused here rather than granted a replay it
            # cannot support.
            if (table.family != family
                    or len(table.circuit_ids) != record["n_circuits"]
                    or list(table.cell_fields) != result["cell_fields"]
                    or _circuit_order_digest(table.circuit_ids)
                        != record["circuit_order_digest"]
                    or _table_digest(table) != record.get("table_digest")):
                raise ValueError(
                    f"the {name} table for {family!r} is not the one that was scored"
                )
            sides[name] = (record, table)

        left_record = sides["left"][0]
        right_record = sides["right"][0]
        shared = (
            left_record["circuit_order_digest"] == right_record["circuit_order_digest"]
        )
        pairable = (
            shared
            and left_record["stream_seed"] == right_record["stream_seed"]
            and left_record["n_circuits"] == right_record["n_circuits"]
        )
        if pairing == "paired":
            if not pairable:
                raise ValueError(
                    f"family {family!r} cannot be paired: the two settings do not "
                    "share their circuits, their order or their stream"
                )
        else:
            overlap = (
                set(sides["left"][1].circuit_ids)
                & set(sides["right"][1].circuit_ids)
            )
            if overlap:
                raise ValueError(
                    f"family {family!r} shares physical circuits across the "
                    "two settings, so independent resampling is invalid"
                )
            # Equal seeds and equal circuit counts reproduce one index matrix,
            # so the difference would cancel the sampling variation it claims to
            # measure and report a falsely narrow interval. Refusing is the only
            # correct answer here: reseeding one side inside the contrast would
            # publish an interval no declared stream can replay.
            if left_record["stream_seed"] == right_record["stream_seed"]:
                raise ValueError(
                    f"family {family!r} requires distinct resample streams "
                    "for an independent contrast"
                )

        values = []
        points = []
        for record, table in (sides["left"], sides["right"]):
            indices = draw_matrix(
                len(table.circuit_ids),
                seed=int(record["stream_seed"]),
                n_resamples=n_resamples,
            )
            drawn = _evaluate(
                table, ladder=ladder, reference_methods=(), span_pairs=span_pairs,
                indices=indices, origin="draw",
            )
            point = _evaluate(
                table, ladder=ladder, reference_methods=(), span_pairs=span_pairs,
                indices=_identity_indices(len(table.circuit_ids)), origin="point",
            )
            values.append(_select(drawn, selector, gap_labels))
            points.append(float(_select(point, selector, gap_labels)[0]))

        draws = values[1] - values[0]
        estimate = points[1] - points[0]
        contrast["families"][family] = {
            "n_draws": n_resamples,
            "left_point": points[0],
            "right_point": points[1],
            "left_stream_seed": left_record["stream_seed"],
            "right_stream_seed": right_record["stream_seed"],
            "circuit_order_digest_shared": bool(shared),
            "contrast": _quantile_interval(estimate, draws, quantiles),
        }
    return contrast


def _evaluate_family(
    table,
    *,
    family,
    ladder,
    rung_labels,
    gap_labels,
    span_pairs,
    reference_methods,
    stream_components,
    root_seed,
    n_resamples,
    quantiles,
) -> dict:
    """Estimate one family and report why anything was withheld."""

    seed = ladder_stream_seed(root_seed, list(stream_components) + [family])
    family_result = {
        "status": "not_estimable",
        "reasons": [],
        "n_rows": None,
        "n_circuits": None,
        "n_cells": None,
        "rows_per_cell": None,
        "excluded_rows": None,
        "circuit_order_digest": None,
        "table_digest": None,
        "stream_seed": seed,
        "errors": None,
        "reference_errors": None,
        "total": None,
        "gaps": None,
        "spans": None,
        "point_values": None,
        "share": {
            "status": "not_estimable", "reasons": [], "point": None, "interval": None,
        },
        "denominator_diagnostics": _diagnostics(),
        "identity_check": {
            "max_absolute_residual": None, "tolerance": None, "held": True,
        },
    }
    if table is None:
        family_result["reasons"] = ["family_missing_from_the_setting"]
        return family_result

    n_circuits = len(table.circuit_ids)
    family_result.update({
        "n_rows": table.n_rows,
        "n_circuits": n_circuits,
        "n_cells": len(table.cell_keys),
        "rows_per_cell": {
            "|".join(key): int(value)
            for key, value in zip(table.cell_keys, table.counts.sum(axis=0))
        },
        # A family holds rows of one stratum only, because the dataset contract
        # gives each family a single stratum. A family that reaches a table
        # therefore has no skipped rows of its own.
        "excluded_rows": 0,
        "circuit_order_digest": _circuit_order_digest(table.circuit_ids),
        "table_digest": _table_digest(table),
    })

    point = _evaluate(
        table, ladder=ladder, reference_methods=reference_methods,
        span_pairs=span_pairs, indices=_identity_indices(n_circuits),
        origin="point",
    )
    point_macro = {method: float(values[0]) for method, values in point["macro"].items()}
    point_total = float(point["total"][0])
    point_gaps = [float(values[0]) for values in point["gaps"]]
    point_spans = {label: float(values[0]) for label, values in point["spans"].items()}
    point_share = float(point["share"][0])
    # The point values are published on every branch, including the branches
    # that emit no interval. A value binary64 cannot represent is null here for
    # the same reason it is null in the diagnostics: the result has to serialize
    # strictly, and the family-level refusal below already says what happened.
    family_result["point_values"] = {
        "errors": {
            label: _json_float(point_macro[ladder[index]])
            for index, label in enumerate(rung_labels)
        },
        "reference_errors": {
            method: _json_float(point_macro[method]) for method in reference_methods
        },
        "total": _json_float(point_total),
        "gaps": {
            label: _json_float(point_gaps[index])
            for index, label in enumerate(gap_labels)
        },
        "spans": {label: _json_float(point_spans[label]) for label in span_pairs},
    }

    family_reasons: list[str] = []
    # The point estimate is the identity draw of the same evaluator, so a
    # nonfinite point macro is the same defect as a nonfinite drawn one and is
    # refused by the same rule rather than left to poison an interval.
    nonfinite_macro = any(not math.isfinite(value) for value in point_macro.values())
    drawn = None
    if n_circuits < 2:
        # A single circuit cannot estimate circuit sampling uncertainty, so no
        # draw is taken and the counts and point values stand alone.
        family_reasons.append("fewer_than_two_circuits")
        diagnostics = _diagnostics(total_point=_json_float(point_total))
    else:
        indices = draw_matrix(n_circuits, seed=seed, n_resamples=n_resamples)
        drawn = _evaluate(
            table, ladder=ladder, reference_methods=reference_methods,
            span_pairs=span_pairs, indices=indices, origin="draw",
        )
        empty = drawn["empty"]
        evaluated = ~empty
        if bool(empty.any()):
            # An empty cell refuses rather than skipping the offending draws,
            # because skipping is discarding.
            family_reasons.append("empty_cell_in_a_draw")
        for values in drawn["macro"].values():
            nonfinite_macro = nonfinite_macro or bool(
                np.any(evaluated & ~np.isfinite(values))
            )
        diagnostics = _diagnostics(
            n_draws=n_resamples,
            total_point=_json_float(point_total),
            totals=drawn["total"],
            shares=drawn["share"],
            evaluated=evaluated,
            n_empty=int(empty.sum()),
        )
    if nonfinite_macro:
        family_reasons.append("nonfinite_macro_in_a_draw")
    family_result["denominator_diagnostics"] = diagnostics
    family_result["identity_check"] = _identity_check(point, drawn)
    family_result["reasons"] = sorted(family_reasons)
    if family_reasons:
        # Every interval of the family is withheld, and with it the share: the
        # share's own reasons describe a denominator this run never measured.
        return family_result

    family_result["status"] = "estimated"
    family_result["errors"] = {
        label: _quantile_interval(
            point_macro[ladder[index]], drawn["macro"][ladder[index]], quantiles
        )
        for index, label in enumerate(rung_labels)
    }
    family_result["reference_errors"] = {
        method: _quantile_interval(point_macro[method], drawn["macro"][method], quantiles)
        for method in reference_methods
    }
    total_interval = _quantile_interval(point_total, drawn["total"], quantiles)
    family_result["total"] = total_interval
    family_result["gaps"] = {
        label: _quantile_interval(point_gaps[index], drawn["gaps"][index], quantiles)
        for index, label in enumerate(gap_labels)
    }
    family_result["spans"] = {
        label: _quantile_interval(point_spans[label], drawn["spans"][label], quantiles)
        for label in span_pairs
    }
    diagnostics["total_interval"] = total_interval
    diagnostics["total_interval_reaches_zero"] = bool(total_interval["lower"] <= 0.0)

    share_reasons: list[str] = []
    # A nonfinite point total is already refused above as a nonfinite macro, so
    # the finiteness clause here is a guard rather than a live branch.
    if not (math.isfinite(point_total) and point_total > 0.0):
        share_reasons.append("nonpositive_point_total")
    if not math.isfinite(point_share):
        share_reasons.append("nonfinite_point_share")
    # A draw total that is not both finite and strictly positive cannot carry a
    # fraction. The two diagnostics counts stay separate so a reader can tell
    # which condition fired.
    if (diagnostics["n_draws_with_nonpositive_total"]
            or diagnostics["n_draws_with_nonfinite_total"]):
        share_reasons.append("nonpositive_total_in_a_draw")
    if diagnostics["n_draws_with_nonfinite_share"]:
        share_reasons.append("nonfinite_share_in_a_draw")
    if diagnostics["total_interval_reaches_zero"]:
        share_reasons.append("total_interval_reaches_zero")

    share = family_result["share"]
    share["reasons"] = sorted(share_reasons)
    if not share_reasons:
        share["status"] = "estimated"
        share["point"] = point_share
        share["interval"] = _quantile_interval(point_share, drawn["share"], quantiles)
    return family_result


def _evaluate(
    table: LadderTable,
    *,
    ladder: Sequence[str],
    reference_methods: Sequence[str],
    span_pairs: Mapping[str, tuple[int, int]],
    indices: np.ndarray,
    origin: str,
    chunk: int | None = None,
) -> dict:
    """Evaluate every reported quantity on each index vector of ``indices``.

    The point estimate and every bootstrap draw run through this one function,
    called on the identity index vector for the point. Two aggregation paths
    would let a point estimate and its draws disagree whenever a cell's rows
    spread unevenly over circuits; one path makes that divergence impossible
    rather than merely tested.
    """

    counts = table.counts
    methods = tuple(ladder) + tuple(reference_methods)
    # The block size is read here rather than bound at definition, so a test
    # can vary it and confirm that it moves no returned number.
    block_size = max(1, int(_EVAL_CHUNK if chunk is None else chunk))
    n = int(indices.shape[0])
    n_gaps = len(ladder) - 1
    macro = {method: np.empty(n, dtype=float) for method in methods}
    empty = np.empty(n, dtype=bool)
    total = np.empty(n, dtype=float)
    gaps = [np.empty(n, dtype=float) for _ in range(n_gaps)]
    spans = {label: np.empty(n, dtype=float) for label in span_pairs}
    share = np.empty(n, dtype=float)
    max_residual = None
    max_tolerance = None

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = indices[start:stop]
        drawn_counts = counts[block].sum(axis=1)
        # The empty-cell mark is taken before any division, so a zero
        # denominator cannot reach the identity check and no draw carries two
        # competing reasons for one condition.
        block_empty = np.any(drawn_counts <= 0.0, axis=1)
        empty[start:stop] = block_empty
        # A draw that sums error entries near the binary64 maximum overflows,
        # and a zero cell count divides by zero. Both are conditions this
        # module reports rather than warnings, so a warning filter set to error
        # cannot change the result.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            block_macro = {}
            for method in methods:
                drawn_errors = table.errors[method][block].sum(axis=1)
                values = (drawn_errors / drawn_counts).mean(axis=1)
                block_macro[method] = values
                macro[method][start:stop] = values
            block_total = block_macro[ladder[0]] - block_macro[ladder[-1]]
            block_gaps = [
                block_macro[ladder[index]] - block_macro[ladder[index + 1]]
                for index in range(n_gaps)
            ]
            block_spans = {
                label: block_macro[ladder[first]] - block_macro[ladder[second]]
                for label, (first, second) in span_pairs.items()
            }
            block_share = block_gaps[0] / block_total
            # A draw that left a cell empty has no decomposition at all, so its
            # derived values are marked undefined rather than carried forward.
            if bool(block_empty.any()):
                for array in (block_total, block_share, *block_gaps,
                              *block_spans.values()):
                    array[block_empty] = np.nan

            checked = ~block_empty
            if bool(checked.any()):
                scale = np.maximum(1.0, np.max(
                    np.stack([np.abs(block_macro[rung]) for rung in ladder]), axis=0
                ))
                tolerance = _IDENTITY_TOLERANCE_FACTOR * _EPS * scale
                residual = block_total - sum(block_gaps)
                _refuse_identity_violation(
                    table.family, origin, start, checked, residual, tolerance,
                    "the total against the sum of its gaps",
                )
                for label, (first, second) in span_pairs.items():
                    span_residual = block_spans[label] - sum(block_gaps[first:second])
                    _refuse_identity_violation(
                        table.family, origin, start, checked, span_residual, tolerance,
                        f"span {label} against the sum of the gaps it covers",
                    )
                    max_residual, max_tolerance = _widest(
                        max_residual, max_tolerance, checked, span_residual, tolerance
                    )
                max_residual, max_tolerance = _widest(
                    max_residual, max_tolerance, checked, residual, tolerance
                )

        total[start:stop] = block_total
        share[start:stop] = block_share
        for index in range(n_gaps):
            gaps[index][start:stop] = block_gaps[index]
        for label in span_pairs:
            spans[label][start:stop] = block_spans[label]

    return {
        "macro": macro,
        "empty": empty,
        "total": total,
        "gaps": gaps,
        "spans": spans,
        "share": share,
        "max_absolute_residual": max_residual,
        "tolerance": max_tolerance,
    }


def _refuse_identity_violation(
    family, origin, offset, checked, residual, tolerance, description,
) -> None:
    """Raise when an additive identity fails beyond rounding.

    A violation means the parts were aggregated differently from the whole,
    which is a code defect rather than an awkward dataset, so it raises instead
    of setting a status.
    """
    with np.errstate(invalid="ignore"):
        violations = checked & (np.abs(residual) > tolerance)
    if not bool(violations.any()):
        return
    index = int(np.flatnonzero(violations)[0])
    raise ValueError(
        f"family {family!r} {origin} {offset + index}: {description} leaves a "
        f"residual of {float(residual[index])!r}, above the tolerance "
        f"{float(tolerance[index])!r}"
    )


def _widest(max_residual, max_tolerance, checked, residual, tolerance):
    """Track the largest residual and tolerance actually compared."""
    with np.errstate(invalid="ignore"):
        usable = checked & np.isfinite(residual) & np.isfinite(tolerance)
    if not bool(usable.any()):
        return max_residual, max_tolerance
    residual_here = float(np.max(np.abs(residual[usable])))
    tolerance_here = float(np.max(tolerance[usable]))
    if max_residual is None or residual_here > max_residual:
        max_residual = residual_here
    if max_tolerance is None or tolerance_here > max_tolerance:
        max_tolerance = tolerance_here
    return max_residual, max_tolerance


def _identity_check(point: Mapping[str, object], drawn) -> dict:
    residuals = [point["max_absolute_residual"]]
    tolerances = [point["tolerance"]]
    if drawn is not None:
        residuals.append(drawn["max_absolute_residual"])
        tolerances.append(drawn["tolerance"])
    residuals = [value for value in residuals if value is not None]
    tolerances = [value for value in tolerances if value is not None]
    return {
        "max_absolute_residual": max(residuals) if residuals else None,
        "tolerance": max(tolerances) if tolerances else None,
        # A violation raises, so a returned check has always held. The field
        # records that the check ran.
        "held": True,
    }


def _diagnostics(
    *,
    n_draws: int = 0,
    total_point=None,
    totals=None,
    shares=None,
    evaluated=None,
    n_empty: int = 0,
) -> dict:
    """Summarize the denominator, published whether or not the share survives.

    Every summary over an empty subset is null rather than a nonfinite float,
    because the result has to serialize with ``allow_nan=False`` on exactly the
    branches where a subset can be empty. Nothing is filtered out of the
    estimate: a nonfinite draw simply has no serializable summary, and its
    count is published beside the summaries it cannot enter.
    """

    diagnostics = {
        "n_draws": int(n_draws),
        "total_point": total_point,
        "total_interval": None,
        "total_interval_reaches_zero": None,
        "n_draws_with_nonpositive_total": 0,
        "n_draws_with_nonfinite_total": 0,
        "n_draws_with_nonfinite_share": 0,
        "n_draws_with_an_empty_cell": int(n_empty),
        "n_draws_finite_total": 0,
        "min_finite_total": None,
        "max_finite_total": None,
        "smallest_positive_total": None,
        "share_min_finite": None,
        "share_max_finite": None,
    }
    if totals is None:
        return diagnostics

    finite_total = evaluated & np.isfinite(totals)
    finite_share = evaluated & np.isfinite(shares)
    diagnostics.update({
        "n_draws_with_nonpositive_total": int(
            np.count_nonzero(finite_total & (totals <= 0.0))
        ),
        "n_draws_with_nonfinite_total": int(
            np.count_nonzero(evaluated & ~np.isfinite(totals))
        ),
        "n_draws_with_nonfinite_share": int(
            np.count_nonzero(evaluated & ~np.isfinite(shares))
        ),
        "n_draws_finite_total": int(np.count_nonzero(finite_total)),
        "min_finite_total": _summary(totals, finite_total, np.min),
        "max_finite_total": _summary(totals, finite_total, np.max),
        "smallest_positive_total": _summary(
            totals, finite_total & (totals > 0.0), np.min
        ),
        "share_min_finite": _summary(shares, finite_share, np.min),
        "share_max_finite": _summary(shares, finite_share, np.max),
    })
    return diagnostics


def _summary(values: np.ndarray, mask: np.ndarray, reducer):
    selected = values[mask]
    if selected.size == 0:
        return None
    return float(reducer(selected))


def _quantile_interval(estimate: float, values: np.ndarray, quantiles) -> dict:
    """Percentile interval from one quantity's own per-draw values.

    The interpolation method is passed explicitly and frozen, so a future change
    of the numpy default cannot move a published number. Every quantity takes
    its interval from its own draws of the one shared resample; no bound is ever
    assembled from another quantity's marginals.
    """
    lower = float(np.quantile(values, quantiles[0], method="linear"))
    upper = float(np.quantile(values, quantiles[1], method="linear"))
    return _interval(estimate, lower, upper)


def _interval(estimate: float, lower: float, upper: float) -> dict:
    for value in (estimate, lower, upper):
        if not math.isfinite(value):
            raise ValueError("reported intervals must be finite")
    return {
        "estimate": float(estimate),
        "lower": float(lower),
        "upper": float(upper),
        "excludes_zero_above": bool(lower > 0),
        "excludes_zero_below": bool(upper < 0),
    }


def _span_pairs(spans, rung_labels, quantity_labels) -> dict[str, tuple[int, int]]:
    """Resolve declared spans to ladder index pairs.

    A span names a contiguous stretch of the ladder, so its value is formed
    inside a draw from that draw's macro errors. Assembling one afterwards from
    two separately intervalled gaps is the construction the review forbids.
    """
    if not isinstance(spans, Mapping):
        raise ValueError("spans must map a label to a pair of rung labels")
    index_of = {label: index for index, label in enumerate(rung_labels)}
    pairs: dict[str, tuple[int, int]] = {}
    for label, endpoints in spans.items():
        _nonempty(label, "a span label")
        if label in quantity_labels or label in _RESERVED_LABELS:
            raise ValueError(
                f"span label {label!r} collides with a reported quantity name"
            )
        endpoints = tuple(endpoints)
        if len(endpoints) != 2 or any(name not in index_of for name in endpoints):
            raise ValueError(f"span {label!r} must name two rung labels")
        first, second = (index_of[name] for name in endpoints)
        if first >= second:
            raise ValueError(
                f"span {label!r} must run from an earlier rung to a later one"
            )
        pairs[label] = (first, second)
    return pairs


def _select(values: Mapping[str, object], selector, gap_labels):
    kind, label = selector
    if kind == "total":
        return values["total"]
    if kind == "gaps":
        return values["gaps"][list(gap_labels).index(label)]
    return values["spans"][label]


def _quantity_definition(result: Mapping[str, object], quantity: str) -> str:
    if quantity == "total":
        return str(result["total_definition"])
    definitions = dict(result["gap_definitions"])
    definitions.update(dict(result["span_definitions"]))
    return str(definitions[quantity])


def _identity_indices(n_circuits: int) -> np.ndarray:
    return np.arange(n_circuits, dtype=np.int64).reshape(1, n_circuits)


def _table_digest(table: LadderTable) -> str:
    """Digest the whole payload a table contributed to its result.

    The circuit-order digest names only which circuits were scored. Everything a
    macro error is actually built from, the counts and the summed errors, sits
    outside it, and ``@dataclass(frozen=True)`` does not freeze a numpy array, so
    an in-place edit leaves that digest intact. Binding the complete payload is
    what lets a replay assert that the table in hand is the one that was scored.
    """
    payload = {
        "family": table.family,
        "circuit_ids": list(table.circuit_ids),
        "cell_fields": list(table.cell_fields),
        "cell_keys": [list(key) for key in table.cell_keys],
        "n_rows": table.n_rows,
        "counts": table.counts.tolist(),
        "errors": {
            method: table.errors[method].tolist()
            for method in sorted(table.errors)
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _circuit_order_digest(circuit_ids: Sequence[str]) -> str:
    """Digest the circuit order injectively.

    Joining on a separator is not injective over identifiers that may contain
    it, so two disjoint pools can collide and be accepted as paired, which
    publishes a zero-width interval for a comparison sharing no circuit.
    Canonical campaign identifiers are hex and cannot collide, but a field
    adapter supplies its own, so the encoding carries the guarantee rather
    than the caller.
    """
    encoded = json.dumps(
        list(circuit_ids), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _component_entropy(component: object) -> int:
    """Map one declared key component to a stable 32-bit integer."""
    if type(component) is int:
        if component < 0:
            raise ValueError("integer stream components must be nonnegative")
        return component
    if isinstance(component, str) and component:
        return int.from_bytes(
            hashlib.sha256(component.encode("utf-8")).digest()[:4], "big"
        )
    raise ValueError(
        "stream components must be nonempty strings or nonnegative integers"
    )


def _distinct_names(values, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence of names")
    names = tuple(values)
    if not names and not allow_empty:
        raise ValueError(f"{label} must be nonempty")
    for name in names:
        _nonempty(name, label)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not repeat a name")
    return names


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _identity(item: Mapping[str, object], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"test rows require nonempty {field}")
    return value


def _finite_absolute_error(prediction: object, target: object) -> float:
    error = abs(float(prediction) - float(target))
    if not math.isfinite(error):
        raise ValueError("derived absolute errors must be finite")
    return error


def _json_float(value: float):
    """Return a float only when strict JSON can carry it, otherwise null."""
    return float(value) if math.isfinite(value) else None


def _validate_table(table: LadderTable) -> None:
    _nonempty(table.family, "family")
    for name, values in (
        ("circuit_ids", table.circuit_ids), ("cell_keys", table.cell_keys)
    ):
        if not values:
            raise ValueError(f"{name} must be nonempty")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not repeat an entry")
        if list(values) != sorted(values):
            raise ValueError(f"{name} must be sorted")
    for key in table.cell_keys:
        if not key or any(not isinstance(part, str) or not part for part in key):
            raise ValueError("a cell key must hold nonempty strings")
    if len({len(key) for key in table.cell_keys}) != 1:
        raise ValueError("every cell key must name the same fields")
    if any(len(key) != len(table.cell_fields) for key in table.cell_keys):
        raise ValueError("cell keys must match the declared cell_fields")
    if type(table.n_rows) is not int or table.n_rows < 0:
        raise ValueError("n_rows must be a nonnegative integer")

    shape = (len(table.circuit_ids), len(table.cell_keys))
    arrays = [("counts", table.counts)]
    arrays.extend((f"errors[{method!r}]", array) for method, array in table.errors.items())
    for name, array in arrays:
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if array.dtype != np.float64:
            raise ValueError(f"{name} must hold float64 values")

    counts = table.counts
    if not np.all(np.isfinite(counts)):
        raise ValueError("counts must be finite")
    if np.any(counts < 0.0) or np.any(counts != np.rint(counts)):
        raise ValueError("counts must hold nonnegative whole numbers")
    if float(counts.sum()) != float(table.n_rows):
        raise ValueError("counts must sum to n_rows")
    if not np.all(counts.sum(axis=0) > 0.0):
        raise ValueError("every cell must hold at least one row")

    unpopulated = counts == 0.0
    for method, array in table.errors.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"errors[{method!r}] must be finite")
        if np.any(array < 0.0):
            raise ValueError(f"errors[{method!r}] must be nonnegative")
        if np.any(array[unpopulated] != 0.0):
            raise ValueError(
                f"errors[{method!r}] must be zero wherever its count is zero"
            )
