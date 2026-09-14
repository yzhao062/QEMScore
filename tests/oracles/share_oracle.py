"""Exact-rational reference calculation for the ladder decomposition.

Why this file exists
--------------------
The joint attribution estimator reports a decomposition of one rung's error
advantage into a classical part and a residual part, with a bootstrap interval on
every quantity. An estimator checked only against its own aggregation helper is
checked against nothing: a wrong weighting appears on both sides of the
comparison and cancels. This module recomputes the same numbers a second time,
from the written contract rather than from the estimator source, and does so in
a form a person can read line by line and confirm by hand.

Three properties are deliberate.

Every number is a ``fractions.Fraction``. A float input is converted with
``Fraction(value)``, which takes the exact binary value the estimator itself
received, so the oracle is exact on exactly the inputs under test. The estimator
works in binary64, so the two agree to a relative tolerance rather than exactly,
and the caller supplies that tolerance.

Every loop is a plain loop over circuits and cells. There is no vectorization and
no reuse of a shared aggregation step between the point estimate and the draws.
The point estimate is the same evaluator called on the identity index vector,
which is the one piece of structure worth sharing, because two aggregation paths
that disagree are the defect this file is meant to find.

Nothing from ``qemscore`` is imported, and numpy is not imported. The estimator
depends on numpy, so a numpy result reached through this module would be a shared
failure mode rather than a second opinion.

What the calculation is
-----------------------
For one family, ``counts[c][k]`` is the number of rows contributed by circuit
``c`` in cell ``k``, and ``errors[method][c][k]`` is the sum of the absolute
errors of that method over those rows. For an index vector ``v``, pool counts and
errors over the drawn circuits, form each cell's mean, and take an equal-weight
mean over cells. That is one rung's macro error.

On a ladder of rung errors ordered from the most restricted rung to the full
rung, the total is the first rung minus the last, each gap is one adjacent
difference, each declared span is a wider difference formed inside the same draw,
and the share is the first gap divided by the total. For the three-rung
controlled ladder with rungs A, C and F this reads::

    T = A - F      K = A - C      D = C - F      T = K + D      S = K / T

Every quantity of a draw comes from that draw's own macro errors, so a common
per-circuit scale cancels out of the share and leaves the share invariant under
the resample.

What this oracle does NOT check
-------------------------------
Agreement with this file is agreement on the numbers alone. It is not
verification of the estimator, and the following are all outside it.

1. The result schema, the status values, the failure reason strings and the
   published diagnostics. This module returns the point value and the per-draw
   values, and the caller applies the refusal rules to them. A run whose numbers
   agree here may still report them under the wrong status.
2. The row ingestion path. Counts and summed errors arrive already aggregated, so
   the split filter, the schema version filter, duplicate identifiers, missing or
   surplus prediction keys, and the family and stratum agreement are all
   untouched. A builder that joins predictions positionally rather than by
   identifier produces a table this module will happily decompose.
3. The resample itself. The draw matrix is consumed as data. A resample of the
   wrong shape, one drawn without replacement, or one drawn from the wrong seed
   is reproduced here and agrees. The hand-enumerated support test and the
   longhand seed replay test cover that ground; this file cannot.
4. The seed derivation and the pairing rule that keeps two training sizes on one
   stream.
5. The identity residual. In exact rational arithmetic the gaps telescope, so the
   residual is identically zero and no tolerance rule can be probed from here. An
   estimator that weights cells differently for one gap shows up as a value
   disagreement instead.
6. Anything that turns on binary64 overflow or underflow. Rationals do not
   overflow, so a case built to drive the estimator's share to an infinite value
   carries a large finite value here and the two disagree by construction. Such
   cases belong in a hand-written test, not in a battery against this file.
7. Serialization, chunking, memory use and running time.

Entry points
------------
``decompose`` is the calculation. ``percentile`` is the interval rule written out
from its definition. ``self_check`` runs the hand-computed examples recorded at
the bottom of this file and is also what ``python share_oracle.py`` runs.
"""

from __future__ import annotations

import math
from fractions import Fraction

__all__ = ["decompose", "percentile", "self_check"]


TOTAL_LABEL = "total"
SHARE_LABEL = "share"


# --------------------------------------------------------------------------
# Reading inputs into exact rationals
# --------------------------------------------------------------------------


def _exact(value, where):
    """Return ``value`` as an exact Fraction, or raise naming ``where``.

    A float becomes its exact binary value rather than its decimal appearance,
    because that binary value is what the estimator was handed. Objects that are
    not Python numbers are accepted when they convert through ``float``, which
    covers a numpy scalar without importing numpy.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{where}: cannot read {value!r} as a number") from None
    if not math.isfinite(as_float):
        raise ValueError(f"{where}: {value!r} is not finite")
    return Fraction(as_float)


def _read_matrix(matrix, where, *, n_circuits=None, n_cells=None):
    """Copy a rectangular circuit-by-cell table into exact rationals."""
    rows = [list(row) for row in matrix]
    if not rows:
        raise ValueError(f"{where} holds no circuits")
    width = len(rows[0])
    if width == 0:
        raise ValueError(f"{where} holds no cells")
    exact = []
    for circuit, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"{where} is not rectangular: circuit 0 has {width} cells and "
                f"circuit {circuit} has {len(row)}")
        exact.append([_exact(cell, f"{where}[{circuit}][{index}]")
                      for index, cell in enumerate(row)])
    if n_circuits is not None and len(exact) != n_circuits:
        raise ValueError(
            f"{where} has {len(exact)} circuits, expected {n_circuits}")
    if n_cells is not None and width != n_cells:
        raise ValueError(f"{where} has {width} cells, expected {n_cells}")
    return exact


def _read_draws(draw_matrix, where, *, n_circuits):
    """Copy a draw matrix into lists of integer circuit indices.

    The row length is taken as given rather than required to equal the circuit
    count, because this module consumes the resample as data and has no standing
    to rule on its shape.
    """
    drawn = []
    for index, row in enumerate(draw_matrix):
        circuits = [int(entry) for entry in row]
        if not circuits:
            raise ValueError(f"{where} draw {index} is empty")
        for circuit in circuits:
            if not 0 <= circuit < n_circuits:
                raise ValueError(
                    f"{where} draw {index} names circuit {circuit}, outside "
                    f"0 to {n_circuits - 1}")
        drawn.append(circuits)
    if not drawn:
        raise ValueError(f"{where} holds no draws")
    return drawn


# --------------------------------------------------------------------------
# The aggregation, written the way a person would do it by hand
# --------------------------------------------------------------------------


def _macro_errors(counts, errors, methods, index_vector):
    """Return one macro error per method for one index vector, or None.

    Pool counts and summed errors over the drawn circuits, divide within each
    cell, then average over cells with equal weight. None is returned when the
    draw leaves a cell with no rows, because that cell has no mean and the whole
    draw is therefore undefined.
    """
    n_cells = len(counts[0])

    pooled_counts = [Fraction(0)] * n_cells
    for circuit in index_vector:
        for cell in range(n_cells):
            pooled_counts[cell] += counts[circuit][cell]
    for cell in range(n_cells):
        if pooled_counts[cell] == 0:
            return None

    macros = {}
    for method in methods:
        table = errors[method]
        running = Fraction(0)
        for cell in range(n_cells):
            pooled_error = Fraction(0)
            for circuit in index_vector:
                pooled_error += table[circuit][cell]
            running += pooled_error / pooled_counts[cell]
        macros[method] = running / n_cells
    return macros


def _quantities(macros, ladder, rung_labels, gap_labels, spans,
                reference_methods, method_of_rung):
    """Assemble every reported quantity from one draw's macro errors."""
    values = {}
    for label, method in zip(rung_labels, ladder):
        values[label] = macros[method]
    for method in reference_methods:
        values[method] = macros[method]

    total = macros[ladder[0]] - macros[ladder[-1]]
    values[TOTAL_LABEL] = total

    running = Fraction(0)
    for position, label in enumerate(gap_labels):
        gap = macros[ladder[position]] - macros[ladder[position + 1]]
        values[label] = gap
        running += gap
    # Exact arithmetic makes the telescoping identity hold with no tolerance.
    # A failure here would mean this module built the gaps off the wrong rungs.
    assert running == total, "the oracle gaps do not sum to the oracle total"

    for label, (start, end) in spans.items():
        values[label] = macros[method_of_rung[start]] - macros[method_of_rung[end]]

    first_gap = values[gap_labels[0]]
    values[SHARE_LABEL] = None if total == 0 else first_gap / total
    return values


# --------------------------------------------------------------------------
# The interval rule, written from its definition
# --------------------------------------------------------------------------


def percentile(values, q):
    """Return the linearly interpolated quantile of ``values`` at ``q``.

    Sort the values, set ``h = (n - 1) * q``, take the floor of ``h``, and
    interpolate between that order statistic and the next by the fractional part.
    This is the rule numpy documents as method ``"linear"``, reimplemented here
    so the oracle does not lean on the same library call the estimator makes.

    ``values`` is a sequence of Fractions and ``q`` is a Fraction in ``[0, 1]``.
    """
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("percentile of an empty sequence")
    if not Fraction(0) <= q <= Fraction(1):
        raise ValueError(f"quantile {q} lies outside 0 to 1")
    position = (n - 1) * q
    lower_index = math.floor(position)
    if lower_index >= n - 1:
        return ordered[n - 1]
    weight = position - lower_index
    return ordered[lower_index] + weight * (
        ordered[lower_index + 1] - ordered[lower_index])


def _interval(draw_values, confidence):
    """Return the percentile interval, or None when a draw value is missing."""
    if any(value is None for value in draw_values):
        return None
    alpha = Fraction(1) - Fraction(confidence)
    half = alpha / 2
    return (percentile(draw_values, half), percentile(draw_values, 1 - half))


# --------------------------------------------------------------------------
# The public calculation
# --------------------------------------------------------------------------


def decompose(families, draws, *, ladder, rung_labels, gap_labels,
              spans=None, reference_methods=(), confidence=0.95):
    """Recompute the ladder decomposition for every family, exactly.

    Parameters
    ----------
    families
        Mapping of family name to ``{"counts": counts, "errors": errors}``.
        ``counts[c][k]`` is the number of rows circuit ``c`` contributes to cell
        ``k``. ``errors[method][c][k]`` is the sum of the absolute errors of that
        method over those rows. Both are plain nested sequences of numbers, of
        the same shape, and every ladder rung and reference method must appear in
        ``errors``.
    draws
        Mapping of family name to that family's circuit-index draw matrix, a
        sequence of sequences of integer circuit indices. Every rung of a family
        is evaluated on the same drawn indices, which is the pairing under test.
    ladder
        Ordered method names, from the most restricted rung to the full rung. Two
        or more, distinct.
    rung_labels
        Short names for the rungs, parallel to ``ladder``. The controlled call
        passes ``("A", "C", "F")``.
    gap_labels
        Names for the adjacent gaps, one shorter than the ladder. The controlled
        call passes ``("K", "D")``.
    spans
        Optional mapping of span label to a pair of rung labels, the first
        strictly earlier in the ladder than the second. A span is formed inside
        each draw from that draw's macro errors, never assembled afterwards.
    reference_methods
        Extra method names scored on the same rows and the same draws, reported
        under their own names and excluded from every gap, every span and the
        total.
    confidence
        Interval confidence. The interval runs from quantile ``alpha / 2`` to
        quantile ``1 - alpha / 2`` with ``alpha = 1 - confidence``. The quantile
        is formed in exact arithmetic from the exact binary value of
        ``confidence``, so it can differ from a binary64 quantile in the last
        bits. The interpolation is continuous in the quantile, so the effect is
        at the rounding level except where two order statistics tie.

    Returns
    -------
    dict
        Mapping of family name to a dict with four keys.

        ``labels``
            Every quantity label in report order: the rung labels, the reference
            method names, ``"total"``, the gap labels, the span labels, and
            ``"share"``.
        ``point``
            Mapping of label to the Fraction from the identity index vector
            ``[0, 1, ..., n_circuits - 1]``, which is the point estimate. The
            share is None when the total is zero. Every entry is None when the
            full index vector leaves a cell with no rows.
        ``draws``
            Mapping of label to a list holding one value per draw, in the order
            the draw matrix supplies. An entry is None when the draw left a cell
            with no rows, and the share entry is also None when that draw's total
            is zero.
        ``interval``
            Mapping of label to a ``(lower, upper)`` pair of Fractions, or None
            when any draw value for that label is None. The share interval is
            therefore None whenever any draw has a zero total.

        The refusal rules of the contract are not applied. A caller derives them
        from the per-draw totals and shares returned here.
    """
    ladder = tuple(ladder)
    rung_labels = tuple(rung_labels)
    gap_labels = tuple(gap_labels)
    reference_methods = tuple(reference_methods)
    spans = dict(spans or {})

    if len(ladder) < 2:
        raise ValueError("the ladder needs two or more rungs")
    if len(set(ladder)) != len(ladder):
        raise ValueError("the ladder repeats a method name")
    if len(rung_labels) != len(ladder):
        raise ValueError("rung_labels and ladder differ in length")
    if len(set(rung_labels)) != len(rung_labels):
        raise ValueError("rung_labels repeats a name")
    if len(gap_labels) != len(ladder) - 1:
        raise ValueError("gap_labels must be one shorter than the ladder")
    if len(set(gap_labels)) != len(gap_labels):
        raise ValueError("gap_labels repeats a name")
    for method in reference_methods:
        if method in ladder:
            raise ValueError(f"reference method {method!r} is also a ladder rung")

    method_of_rung = dict(zip(rung_labels, ladder))
    rung_position = {label: index for index, label in enumerate(rung_labels)}
    for label, pair in spans.items():
        start, end = pair
        if start not in rung_position or end not in rung_position:
            raise ValueError(f"span {label!r} names a rung outside the ladder")
        if rung_position[start] >= rung_position[end]:
            raise ValueError(f"span {label!r} is not in ladder order")

    labels = (rung_labels + reference_methods + (TOTAL_LABEL,) + gap_labels
              + tuple(spans) + (SHARE_LABEL,))
    if len(set(labels)) != len(labels):
        raise ValueError("two reported quantities carry the same label")

    methods = ladder + reference_methods
    results = {}
    for family in sorted(families):
        table = families[family]
        counts = _read_matrix(table["counts"], f"{family} counts")
        n_circuits = len(counts)
        n_cells = len(counts[0])

        errors = {}
        for method in methods:
            if method not in table["errors"]:
                raise ValueError(f"{family} carries no errors for {method!r}")
            errors[method] = _read_matrix(
                table["errors"][method], f"{family} errors[{method}]",
                n_circuits=n_circuits, n_cells=n_cells)

        if family not in draws:
            raise ValueError(f"{family} has no draw matrix")
        drawn = _read_draws(draws[family], f"{family} draws",
                            n_circuits=n_circuits)

        identity = list(range(n_circuits))
        point_macros = _macro_errors(counts, errors, methods, identity)
        if point_macros is None:
            point = {label: None for label in labels}
        else:
            point = _quantities(point_macros, ladder, rung_labels, gap_labels,
                                spans, reference_methods, method_of_rung)

        per_draw = {label: [] for label in labels}
        for index_vector in drawn:
            macros = _macro_errors(counts, errors, methods, index_vector)
            if macros is None:
                for label in labels:
                    per_draw[label].append(None)
                continue
            values = _quantities(macros, ladder, rung_labels, gap_labels, spans,
                                 reference_methods, method_of_rung)
            for label in labels:
                per_draw[label].append(values[label])

        intervals = {label: _interval(per_draw[label], confidence)
                     for label in labels}
        results[family] = {"labels": labels, "point": point,
                           "draws": per_draw, "interval": intervals}
    return results


# --------------------------------------------------------------------------
# Hand-computed examples
#
# Every expected value below was worked out on paper before the code ran, using
# error values that are exact in binary64 so the expectation is an exact
# rational rather than an approximation.
# --------------------------------------------------------------------------


LADDER = ("feat-only", "liao-feat-only", "liao")
RUNGS = ("A", "C", "F")
GAPS = ("K", "D")


def _constant_family(errors, *, circuits=2, cells=1):
    """One family whose every circuit and cell carries the same error."""
    counts = [[1] * cells for _ in range(circuits)]
    tables = {method: [[error] * cells for _ in range(circuits)]
              for method, error in errors.items()}
    return {"counts": counts, "errors": tables}


def _all_draws_of_two_circuits():
    return [[0, 0], [0, 1], [1, 0], [1, 1]]


def _check_constant_three_rung_ladder():
    """A = 1/2, C = 1/8, F = 1/16 on every row, so every draw agrees.

    T = 1/2 - 1/16 = 7/16.  K = 1/2 - 1/8 = 3/8.  D = 1/8 - 1/16 = 1/16.
    K + D = 6/16 + 1/16 = 7/16 = T.  S = (3/8) / (7/16) = 6/7.
    Constant errors leave no resampling spread, so both interval endpoints sit on
    the point value.
    """
    family = _constant_family({"feat-only": 0.5, "liao-feat-only": 0.125,
                               "liao": 0.0625}, circuits=2, cells=1)
    out = decompose({"tfi": family}, {"tfi": _all_draws_of_two_circuits()},
                    ladder=LADDER, rung_labels=RUNGS, gap_labels=GAPS)["tfi"]

    assert out["point"]["A"] == Fraction(1, 2)
    assert out["point"]["C"] == Fraction(1, 8)
    assert out["point"]["F"] == Fraction(1, 16)
    assert out["point"]["total"] == Fraction(7, 16)
    assert out["point"]["K"] == Fraction(3, 8)
    assert out["point"]["D"] == Fraction(1, 16)
    assert out["point"]["share"] == Fraction(6, 7)
    assert out["interval"]["share"] == (Fraction(6, 7), Fraction(6, 7))
    assert out["interval"]["total"] == (Fraction(7, 16), Fraction(7, 16))
    assert all(value == Fraction(6, 7) for value in out["draws"]["share"])


def _check_unequal_counts_pool_within_a_cell():
    """Counts differ across cells and circuits, so pooling is what is under test.

    Two circuits and two cells.  Counts are ``[[1, 2], [3, 1]]``.  The A errors
    are ``[[1/2, 1], [3/2, 1/4]]``, and C is exactly half of A entry by entry
    while F is exactly a quarter of it.

    Point draw, both circuits.  Pooled counts are [4, 3] and pooled A errors are
    [2, 5/4].  Cell means are 1/2 and 5/12, so macro A = (6/12 + 5/12) / 2
    = 11/24.  Then C = 11/48 and F = 11/96.
    T = 11/24 - 11/96 = 44/96 - 11/96 = 33/96 = 11/32.
    K = 22/48 - 11/48 = 11/48.  D = 22/96 - 11/96 = 11/96.
    K + D = 22/96 + 11/96 = 33/96 = T.  S = (11/48) / (11/32) = 2/3.

    Averaging the four circuit-and-cell means instead of pooling inside each cell
    would give (1/2 + 1/2 + 1/2 + 1/4) / 4 = 7/16, which differs from 11/24, so
    this example separates the two aggregations.

    The two single-circuit draws give macro A = 1/2 and macro A = 3/8, worked out
    the same way, and the share is 2/3 in every draw because C and F are fixed
    multiples of A in every cell.
    """
    family = {
        "counts": [[1, 2], [3, 1]],
        "errors": {
            "feat-only": [[0.5, 1.0], [1.5, 0.25]],
            "liao-feat-only": [[0.25, 0.5], [0.75, 0.125]],
            "liao": [[0.125, 0.25], [0.375, 0.0625]],
        },
    }
    out = decompose({"tfi": family}, {"tfi": _all_draws_of_two_circuits()},
                    ladder=LADDER, rung_labels=RUNGS, gap_labels=GAPS)["tfi"]

    assert out["point"]["A"] == Fraction(11, 24)
    assert out["point"]["C"] == Fraction(11, 48)
    assert out["point"]["F"] == Fraction(11, 96)
    assert out["point"]["total"] == Fraction(11, 32)
    assert out["point"]["K"] == Fraction(11, 48)
    assert out["point"]["D"] == Fraction(11, 96)
    assert out["point"]["share"] == Fraction(2, 3)
    assert out["point"]["A"] != Fraction(7, 16)

    assert sorted(out["draws"]["A"]) == [
        Fraction(3, 8), Fraction(11, 24), Fraction(11, 24), Fraction(1, 2)]
    assert all(value == Fraction(2, 3) for value in out["draws"]["share"])
    assert out["interval"]["share"] == (Fraction(2, 3), Fraction(2, 3))
    assert out["interval"]["A"][0] < out["interval"]["A"][1]


def _check_a_negative_residual_gap_is_reported_as_computed():
    """A = 1/2, C = 1/16, F = 1/8, so the residual gap is negative.

    T = 1/2 - 1/8 = 3/8.  K = 1/2 - 1/16 = 7/16.  D = 1/16 - 1/8 = -1/16.
    K + D = 7/16 - 1/16 = 6/16 = 3/8 = T.  S = (7/16) / (3/8) = 7/6, above one.
    """
    family = _constant_family({"feat-only": 0.5, "liao-feat-only": 0.0625,
                               "liao": 0.125}, circuits=2, cells=1)
    out = decompose({"tfi": family}, {"tfi": _all_draws_of_two_circuits()},
                    ladder=LADDER, rung_labels=RUNGS, gap_labels=GAPS)["tfi"]

    assert out["point"]["D"] == Fraction(-1, 16)
    assert out["point"]["K"] == Fraction(7, 16)
    assert out["point"]["total"] == Fraction(3, 8)
    assert out["point"]["share"] == Fraction(7, 6)
    assert out["interval"]["D"] == (Fraction(-1, 16), Fraction(-1, 16))


def _check_a_zero_total_leaves_the_share_undefined():
    """A = F = 1/4 and C = 1/8, so T = 0 while K and D stay defined.

    K = 1/4 - 1/8 = 1/8 and D = 1/8 - 1/4 = -1/8, which sum to zero.  The share
    has no value, here or in any draw, so its interval has no value either.
    """
    family = _constant_family({"feat-only": 0.25, "liao-feat-only": 0.125,
                               "liao": 0.25}, circuits=2, cells=1)
    out = decompose({"tfi": family}, {"tfi": _all_draws_of_two_circuits()},
                    ladder=LADDER, rung_labels=RUNGS, gap_labels=GAPS)["tfi"]

    assert out["point"]["total"] == Fraction(0)
    assert out["point"]["K"] == Fraction(1, 8)
    assert out["point"]["D"] == Fraction(-1, 8)
    assert out["point"]["share"] is None
    assert out["draws"]["share"] == [None, None, None, None]
    assert out["interval"]["share"] is None
    assert out["interval"]["K"] == (Fraction(1, 8), Fraction(1, 8))


def _check_a_draw_that_empties_a_cell_has_no_values():
    """Cell one is populated by circuit one alone, so the draw [0, 0] empties it.

    Counts are ``[[2, 0], [1, 1]]``.  The A errors are ``[[1, 0], [1/2, 1/4]]``.
    Draw [0, 1] pools counts [3, 1] and A errors [3/2, 1/4], giving cell means
    1/2 and 1/4 and macro A = 3/8.  Draw [0, 0] leaves cell one with no rows, so
    that draw carries no value at all and the intervals fall away.
    """
    family = {
        "counts": [[2, 0], [1, 1]],
        "errors": {
            "feat-only": [[1.0, 0.0], [0.5, 0.25]],
            "liao-feat-only": [[0.5, 0.0], [0.25, 0.125]],
            "liao": [[0.25, 0.0], [0.125, 0.0625]],
        },
    }
    out = decompose({"tfi": family}, {"tfi": _all_draws_of_two_circuits()},
                    ladder=LADDER, rung_labels=RUNGS, gap_labels=GAPS)["tfi"]

    assert out["draws"]["A"][0] is None
    assert out["draws"]["share"][0] is None
    assert out["draws"]["A"][1] == Fraction(3, 8)
    assert out["draws"]["A"][3] == Fraction(3, 8)
    assert out["interval"]["A"] is None
    assert out["point"]["A"] == Fraction(3, 8)


def _check_a_four_rung_ladder_with_a_span_and_a_reference():
    """A = 1, C = 1/2, O = 3/8, F = 1/4, with an unmitigated reference at 3/2.

    T = 1 - 1/4 = 3/4.  K = 1/2, M = 1/8, D = 1/8, and those three sum to 3/4.
    The span from C to F is 1/2 - 1/4 = 1/4, which is M + D.
    S = K / T = (1/2) / (3/4) = 2/3.  The reference sits beside the ladder and
    changes neither the total nor any gap.
    """
    four_ladder = ("feat-only", "liao-feat-only", "liao-observed", "liao")
    family = _constant_family({
        "feat-only": 1.0, "liao-feat-only": 0.5, "liao-observed": 0.375,
        "liao": 0.25, "unmitigated": 1.5}, circuits=2, cells=1)
    out = decompose(
        {"tfi": family}, {"tfi": _all_draws_of_two_circuits()},
        ladder=four_ladder, rung_labels=("A", "C", "O", "F"),
        gap_labels=("K", "M", "D"), spans={"C_minus_F": ("C", "F")},
        reference_methods=("unmitigated",))["tfi"]

    assert out["point"]["total"] == Fraction(3, 4)
    assert out["point"]["K"] == Fraction(1, 2)
    assert out["point"]["M"] == Fraction(1, 8)
    assert out["point"]["D"] == Fraction(1, 8)
    assert out["point"]["C_minus_F"] == Fraction(1, 4)
    assert out["point"]["C_minus_F"] == out["point"]["M"] + out["point"]["D"]
    assert out["point"]["share"] == Fraction(2, 3)
    assert out["point"]["unmitigated"] == Fraction(3, 2)
    assert out["labels"] == ("A", "C", "O", "F", "unmitigated", "total",
                             "K", "M", "D", "C_minus_F", "share")


def _check_the_percentile_rule():
    """Four values, so h = 3q and the interpolation is checkable in the head.

    On [1, 2, 4, 8]: q = 0 gives 1; q = 1/4 gives h = 3/4, so 1 + (3/4)(2 - 1)
    = 7/4; q = 1/3 gives h = 1, so exactly 2; q = 1/2 gives h = 3/2, so
    2 + (1/2)(4 - 2) = 3; q = 1 gives 8.
    """
    values = [Fraction(1), Fraction(2), Fraction(4), Fraction(8)]
    assert percentile(values, Fraction(0)) == Fraction(1)
    assert percentile(values, Fraction(1, 4)) == Fraction(7, 4)
    assert percentile(values, Fraction(1, 3)) == Fraction(2)
    assert percentile(values, Fraction(1, 2)) == Fraction(3)
    assert percentile(values, Fraction(1)) == Fraction(8)
    # The sequence is sorted on the way in, so the input order cannot matter.
    assert percentile([Fraction(8), Fraction(1), Fraction(4), Fraction(2)],
                      Fraction(1, 2)) == Fraction(3)


def _check_a_common_circuit_scale_cancels_out_of_the_share():
    """Each circuit carries its own error scale, shared by every rung.

    Circuit ``c`` has errors ``s_c`` times 1, 1/2 and 1/4 for A, C and F, with
    the scales 1, 4, 16 and 64.  Every draw scales all three macro errors by the
    same drawn amount, so the share is (1 - 1/2) / (1 - 1/4) = 2/3 in every draw
    while the total moves over a wide range.  A rung resampled on its own draw
    would leave a different scale in the numerator and the share would spread.
    """
    scales = (1.0, 4.0, 16.0, 64.0)
    family = {
        "counts": [[1] for _ in scales],
        "errors": {
            "feat-only": [[scale] for scale in scales],
            "liao-feat-only": [[scale * 0.5] for scale in scales],
            "liao": [[scale * 0.25] for scale in scales],
        },
    }
    drawn = [[0, 0, 0, 0], [0, 1, 2, 3], [3, 3, 3, 3], [1, 1, 2, 3]]
    out = decompose({"tfi": family}, {"tfi": drawn}, ladder=LADDER,
                    rung_labels=RUNGS, gap_labels=GAPS)["tfi"]

    assert all(value == Fraction(2, 3) for value in out["draws"]["share"])
    assert out["interval"]["share"] == (Fraction(2, 3), Fraction(2, 3))
    assert out["draws"]["total"][0] == Fraction(3, 4)
    assert out["draws"]["total"][2] == Fraction(48)
    assert out["interval"]["total"][0] < out["interval"]["total"][1]


def self_check():
    """Run every hand-computed example and raise on the first disagreement."""
    _check_constant_three_rung_ladder()
    _check_unequal_counts_pool_within_a_cell()
    _check_a_negative_residual_gap_is_reported_as_computed()
    _check_a_zero_total_leaves_the_share_undefined()
    _check_a_draw_that_empties_a_cell_has_no_values()
    _check_a_four_rung_ladder_with_a_span_and_a_reference()
    _check_the_percentile_rule()
    _check_a_common_circuit_scale_cancels_out_of_the_share()


if __name__ == "__main__":
    self_check()
    print("share_oracle: every hand-computed example agrees")
