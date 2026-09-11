"""Offline, descriptive correlations of observed anonymous + file RSS excess.

The caller supplies past-only window features and labels, including labels evaluated
at the future one-second horizon. It estimates the exploratory, fixed NEVER median
reference from 30 NEVER runs at phase/elapsed_s with support >=15, and reports its
IQR separately. This module neither estimates that reference nor validates its
support. Excess is not full RSS, measured waste, or a causal THP effect.

Only ALWAYS rows contribute. A collection is the independent unit, separately
within each phase; phases are never counted as independent replicates together.
Adjusted correlations fit OLS on complete cases across the WHOLE collection phase,
including future labels. They are descriptive partial correlations, NOT deployable
feature transforms, test-trained estimates, models, or predictive validation.
"""

import numpy as np
import polars as pl
from scipy import stats

TARGETS = ("excess_now_mib", "excess_1s_mib", "excess_change_1s_mib")
KEY_SCHEMA = dict.fromkeys(
    ("phase", "feature", "target", "method", "adjustment"), pl.String
)
CATALOG_SCHEMA = dict.fromkeys(
    ("group", "scope", "unit", "description", "availability", "limitations"), pl.String
) | {"target_linked": pl.Boolean}
NOTES = {
    "target_definition": (
        "Observed anonymous + file RSS excess; not full RSS, measured waste, "
        "or a causal THP effect."
    ),
    "reference_notes": (
        "Exploratory fixed reference: caller-estimated median of 30 NEVER runs "
        "at phase/elapsed_s, minimum support 15. CI and p-values are conditional "
        "on this estimate and exclude reference uncertainty. Caller must report "
        "NEVER-reference IQR separately; no test-trained claim."
    ),
    "adjustment_notes": (
        "Adjusted variants fit OLS with intercept and controls on finite complete "
        "cases across the whole collection phase. Partial Spearman ranks X, Y "
        "and each control first (average ties). Descriptive partial correlations "
        "only, not deployable feature transforms or predictive validation. "
        "Input window features are unchanged."
    ),
    "ci_method": (
        "95% percentile bootstrap of median_r, resampling whole independent-run "
        "correlations with equal weight; no window pooling or event weighting"
    ),
    "p_value_method": "two-sided independent-run sign test",
    "p_value_null": (
        "Directional consistency: P(positive run correlation | nonzero) = 0.5; "
        "exact zero ties excluded. Not pooled Pearson significance or a Welch test."
    ),
    "holm_family": (
        "All valid feature x target x method x adjustment x phase tests in this "
        "call, including target-linked diagnostics; no top-feature preselection"
    ),
}
PER_RUN_SCHEMA = {
    "collection_id": pl.String,
    **KEY_SCHEMA,
    "n_windows": pl.Int64,
    "n_points": pl.Int64,
    "r": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
SUMMARY_SCHEMA = {
    **KEY_SCHEMA,
    **dict.fromkeys(
        ("median_r", "mean_r", "ci_low", "ci_high", "p_value", "p_value_holm"),
        pl.Float64,
    ),
    **dict.fromkeys(
        (
            "n_runs",
            "n_positive",
            "n_negative",
            "n_tied",
            "n_total_windows",
            "n_candidate_runs",
            "n_excluded_runs",
            "n_input_windows",
            "n_complete_windows",
            "holm_family_size",
        ),
        pl.Int64,
    ),
    "median_windows_per_run": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
    **CATALOG_SCHEMA,
    "predictor_role": pl.String,
    "controls": pl.String,
    **dict.fromkeys(NOTES, pl.String),
}


def _correlate(values, method, adjusted, min_points):
    values = values[np.isfinite(values).all(axis=1)]
    n = len(values)
    if n < min_points:
        return n, None, "insufficient_points"
    if method == "spearman":
        values = stats.rankdata(values, method="average", axis=0)

    # Scaling before demeaning avoids overflow for otherwise finite inputs.
    scales = np.max(np.abs(values), axis=0)
    values = values / np.where(scales == 0, 1, scales)
    values -= values.mean(axis=0)
    norms = np.linalg.norm(values, axis=0)
    for i, name in enumerate(("x", "y")):
        if norms[i] == 0:
            return n, None, f"constant_{name}"
    if adjusted and n <= values.shape[1]:
        # Partial-correlation residual df = n - number_of_controls - 2.
        return n, None, "insufficient_residual_dof"
    if np.any(norms[2:] == 0):
        return n, None, "degenerate_controls"
    values /= norms
    xy = values[:, :2]
    if adjusted:
        design = np.column_stack((np.ones(n) / np.sqrt(n), values[:, 2:]))
        try:
            coefficients, _, rank, _ = np.linalg.lstsq(design, xy, rcond=None)
        except np.linalg.LinAlgError:
            return n, None, "degenerate_controls"
        if rank != design.shape[1]:
            return n, None, "degenerate_controls"
        xy = xy - design @ coefficients
        xy -= xy.mean(axis=0)
        norms = np.linalg.norm(xy, axis=0)
        tolerance = 10 * np.finfo(float).eps * max(design.shape)
        for i, name in enumerate(("x", "y")):
            if norms[i] <= tolerance:
                return n, None, f"constant_residual_{name}"
        xy /= norms
    r = float(np.sum(xy[:, 0] * xy[:, 1]))
    if not np.isfinite(r):
        return n, None, "nonfinite_correlation"
    return n, float(np.clip(r, -1, 1)), "ok"


def summarize_correlations(
    windows: pl.DataFrame,
    catalog: list[dict],
    *,
    seed=17,
    bootstrap_samples=2000,
    min_points=8,
    min_runs=10,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (summary, per_run), without modifying windows or catalog.

    Screen every f__ column and f__ catalog name, including unsupported features.
    Missing features/labels/controls and rejected correlations retain QC rows;
    missing grouping keys on nonempty input are errors. Missing/null phase means
    "run". Empty input, no features, or no ALWAYS rows gives typed empty tables.

    Raw correlations use finite X/Y pairs. Adjusted variants additionally require
    elapsed_s, and future level/change also require excess_now_mib. OLS includes
    an intercept; rank-deficient controls, constant residuals, and nonpositive
    residual degrees of freedom are rejected, not silently reduced to raw fits.

    Summary rows are phase-specific. n_runs counts valid collection correlations;
    n_total_windows and median_windows_per_run describe only those contributors.
    n_input_windows/n_complete_windows also include rejected runs, whose n_points
    and status/reason are in per_run. Missing r is null, never a zero/tied result.

    ci_low/ci_high bound median_r. Inference requires min_runs (at least 10),
    otherwise status is insufficient_runs and CI/p-values are null. Exact-zero
    ties count toward n_runs but not the binomial sign test; an all-tied test has
    null p-value. Holm includes every non-null p-value in this entire call. All
    inference is conditional on the caller's fixed estimated NEVER reference.
    Sorted feature/phase/collection order and a local RNG make catalog/column
    order irrelevant. Bootstrap storage is bounded per test, not per family.
    """
    for name, value, minimum in (
        ("bootstrap_samples", bootstrap_samples, 1),
        ("min_points", min_points, 2),
        ("min_runs", min_runs, 10),
    ):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    empty = (pl.DataFrame(schema=SUMMARY_SCHEMA), pl.DataFrame(schema=PER_RUN_SCHEMA))
    if windows.is_empty():
        return empty
    if not {"collection_id", "thp"} <= set(windows.columns):
        raise ValueError("nonempty windows require collection_id and thp")
    always = windows.filter(pl.col("thp") == "always")
    if always.is_empty():
        return empty
    if always["collection_id"].null_count():
        raise ValueError("ALWAYS rows require non-null collection_id")
    entries = {entry["name"]: entry for entry in catalog}
    if len(entries) != len(catalog):
        raise ValueError("catalog feature names must be unique")
    features = sorted(
        name for name in set(windows.columns) | set(entries) if name.startswith("f__")
    )
    if not features:
        return empty
    numeric = sorted(set(features) | set(TARGETS) | {"elapsed_s"})
    missing = set(numeric) - set(windows.columns)
    indices = {name: index for index, name in enumerate(numeric)}
    data = always.select(
        pl.col("collection_id").cast(pl.String),
        (pl.col("phase") if "phase" in always.columns else pl.lit("run"))
        .cast(pl.String)
        .fill_null("run")
        .alias("phase"),
        *[
            (pl.lit(None) if name in missing else pl.col(name))
            .cast(pl.Float64, strict=False)
            .alias(name)
            for name in numeric
        ],
    )
    runs = {}
    for (cid, phase), frame in sorted(
        data.partition_by(["collection_id", "phase"], as_dict=True).items()
    ):
        runs.setdefault(phase, []).append((cid, frame.select(numeric).to_numpy()))

    rng = np.random.default_rng(seed)
    summary_rows, run_rows = [], []
    for feature in features:
        entry = entries.get(feature, {})
        metadata = {name: entry.get(name) for name in CATALOG_SCHEMA}
        linked = metadata["target_linked"]
        metadata["predictor_role"] = (
            "Target-linked diagnostic; not an independent predictor"
            if linked is True
            else "Not marked target-linked; predictive validity not established"
            if linked is False
            else "Unknown target linkage; independence not established"
        )
        for phase, phase_runs in sorted(runs.items()):
            for target in TARGETS:
                for method in ("pearson", "spearman"):
                    for adjustment in ("raw", "adjusted"):
                        controls = []
                        if adjustment == "adjusted":
                            controls = ["elapsed_s"]
                            if target != "excess_now_mib":
                                controls.append("excess_now_mib")
                        columns = [feature, target, *controls]
                        absent = sorted(set(columns) & missing)
                        key = dict(
                            phase=phase,
                            feature=feature,
                            target=target,
                            method=method,
                            adjustment=adjustment,
                        )
                        valid_r, valid_points = [], []
                        complete_points = 0
                        for cid, values in phase_runs:
                            if absent:
                                n, r, status = 0, None, "missing_columns"
                            else:
                                n, r, status = _correlate(
                                    values[:, [indices[name] for name in columns]],
                                    method,
                                    adjustment == "adjusted",
                                    min_points,
                                )
                            run_rows.append(
                                {
                                    "collection_id": cid,
                                    **key,
                                    "n_windows": len(values),
                                    "n_points": n,
                                    "r": r,
                                    "status": status,
                                    "reason": "Missing columns: " + ", ".join(absent)
                                    if absent
                                    else status.replace("_", " "),
                                }
                            )
                            complete_points += n
                            if r is not None:
                                valid_r.append(r)
                                valid_points.append(n)
                        rs = np.asarray(valid_r, dtype=float)
                        count = len(rs)
                        positive, negative = int(np.sum(rs > 0)), int(np.sum(rs < 0))
                        row = {
                            **key,
                            **metadata,
                            **NOTES,
                            "controls": ",".join(controls),
                            "median_r": float(np.median(rs)) if count else None,
                            "mean_r": float(np.mean(rs)) if count else None,
                            "n_runs": count,
                            "n_positive": positive,
                            "n_negative": negative,
                            "n_tied": count - positive - negative,
                            "n_total_windows": sum(valid_points),
                            "median_windows_per_run": float(np.median(valid_points))
                            if count
                            else None,
                            "n_candidate_runs": len(phase_runs),
                            "n_excluded_runs": len(phase_runs) - count,
                            "n_input_windows": sum(
                                len(values) for _, values in phase_runs
                            ),
                            "n_complete_windows": complete_points,
                            "ci_low": None,
                            "ci_high": None,
                            "p_value": None,
                            "p_value_holm": None,
                            "status": "insufficient_runs",
                            "reason": f"Requires {min_runs} valid independent runs; got {count}. "
                            "See per_run QC for exclusions.",
                        }
                        if count >= min_runs:
                            medians = np.empty(bootstrap_samples)
                            batch = max(1, 65536 // count)
                            for start in range(0, bootstrap_samples, batch):
                                stop = min(start + batch, bootstrap_samples)
                                draws = rng.integers(
                                    0, count, size=(stop - start, count)
                                )
                                medians[start:stop] = np.median(rs[draws], axis=1)
                            row["ci_low"], row["ci_high"] = map(
                                float, np.quantile(medians, [0.025, 0.975])
                            )
                            if positive + negative:
                                row["p_value"] = float(
                                    stats.binomtest(
                                        positive,
                                        positive + negative,
                                        p=0.5,
                                        alternative="two-sided",
                                    ).pvalue
                                )
                                row.update(status="ok", reason="ok")
                            else:
                                row.update(
                                    status="no_nonzero_runs",
                                    reason="All valid correlations are exact-zero ties; sign test undefined.",
                                )
                        summary_rows.append(row)

    family = sorted(
        (row for row in summary_rows if row["p_value"] is not None),
        key=lambda row: row["p_value"],
    )
    previous = 0.0
    for index, row in enumerate(family):
        previous = min(1.0, max(previous, (len(family) - index) * row["p_value"]))
        row["p_value_holm"] = previous
    for row in summary_rows:
        row["holm_family_size"] = len(family)
    return (
        pl.DataFrame(summary_rows, schema=SUMMARY_SCHEMA),
        pl.DataFrame(run_rows, schema=PER_RUN_SCHEMA),
    )
