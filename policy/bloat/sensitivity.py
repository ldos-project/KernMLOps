"""Post-hoc, descriptive phase-profile sensitivity of observed RSS excess.

Use only completed ALWAYS r00-r14 to fit phase/elapsed medians, then correlate
residuals in ALWAYS r15-r29. This analysis was chosen after seeing the full screen:
the split is NOT a confirmatory holdout, trained predictor or predictive validation.
Only existing derived inputs are read. Importing this module performs no I/O.
"""

import argparse
import ctypes
import hashlib
import json
import os
import tempfile
from pathlib import Path

import correlate
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("data/derived/bloat/full-probes-20260908/correlations")
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "phase-sensitivity"
TARGETS = correlate.TARGETS
VARIANT = "calibration_phase_profile_residuals"
SPLIT = {
    "calibration_runs": "ALWAYS r00-r14",
    "evaluation_runs": "ALWAYS r15-r29",
}
CONTEXT = {
    "analysis_variant": VARIANT,
    **SPLIT,
    "feature_data_kind": "derived_phase_profile_residuals_not_raw",
    "analysis_notes": (
        "Post-hoc descriptive sensitivity chosen after seeing the full-cohort screen; "
        "NOT a confirmatory holdout, trained predictor or predictive validation. "
        "No classifier or constant-positive label is fitted."
    ),
    "profile_notes": (
        "Per-variable median at exact (phase, elapsed_s), using only completed "
        "ALWAYS r00-r14; minimum 8 distinct finite calibration runs. Completed "
        "calibration trajectories, including future-horizon labels, are available "
        "prior to evaluation. No evaluation or NEVER rows fit profiles. No time "
        "interpolation, extrapolation, phase mixing or missing-value filling. "
        "Each of the three targets is centered independently, including the current "
        "target control; residual change need not equal residual future minus now."
    ),
    "feature_data_notes": (
        "Output residual f__ columns are derived, NOT raw feature data. Original "
        "input windows and catalog are unmodified. Only f__ columns and the three "
        "targets are centered; elapsed_s and observed RSS/reference columns are "
        "unchanged. Unsupported or nonfinite residuals are null."
    ),
}
STATISTICS = {
    **correlate.NOTES,
    **CONTEXT,
    "target_definition": (
        "Calibration-phase-profile residual of observed anonymous + file RSS excess "
        "versus the fixed NEVER median; not full RSS, measured waste or a causal "
        "THP effect. Each current/future/change target has its own calibration median."
    ),
    "reference_notes": (
        "The input's fixed phase/elapsed NEVER reference (30 NEVER runs, minimum "
        "support 15) is unchanged across calibration/evaluation subsets, never "
        "refitted or mixed across phases. Reference IQR columns remain unchanged. "
        "CI and p-values are conditional on BOTH this reference and the fitted "
        "calibration profiles; they exclude calibration and reference uncertainty. "
        "NEVER is not an observed counterfactual."
    ),
    "adjustment_notes": (
        "raw means correlation of calibration-phase-profile residual X and Y, NOT "
        "raw-input correlation. adjusted additionally fits within-collection, "
        "within-phase OLS with intercept and unchanged elapsed_s; future level/change "
        "also controls for residual excess_now_mib. Fits use finite complete cases "
        "across the whole evaluation collection phase, including future labels. "
        "Partial Spearman ranks residual X/Y and each control first (average ties). "
        "Descriptive partial correlations, not deployable prediction transforms."
    ),
    "ci_method": correlate.NOTES["ci_method"]
    + "; conditional on fixed calibration profiles and NEVER reference; excludes "
    "calibration and reference uncertainty",
    "p_value_method": (
        "two-sided independent-evaluation-run sign test, conditional on fixed "
        "calibration profiles and NEVER reference"
    ),
    "p_value_null": correlate.NOTES["p_value_null"]
    + " Runs are treated as independent conditional on the shared fixed profiles "
    "and reference. Post-hoc choice is not corrected by Holm; not confirmatory inference.",
}


def phase_residuals(windows: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Return (long profiles, evaluation residual windows, QC), without mutation.

    Require exactly one ALWAYS collection per replicate 0..29 and one row per
    collection/phase/integer second. NEVER rows are ignored, not used to rebuild
    the fixed input reference. Unsupported profile points retain n with null value;
    evaluation-only phase/times have no profile and are retained with null residuals.
    """
    keys = ["phase", "elapsed_s"]
    required = {"collection_id", "thp", "replicate", *keys, *TARGETS}
    if missing := required - set(windows.columns):
        raise ValueError(f"missing window columns: {sorted(missing)}")
    if "analysis_variant" in windows.columns:
        raise ValueError("require original derived windows, not prior residual output")
    features = sorted(name for name in windows.columns if name.startswith("f__"))
    if not features:
        raise ValueError("require f__ feature columns")
    variables = sorted([*features, *TARGETS])
    always = windows.filter(pl.col("thp") == "always")
    for name in ("replicate", "elapsed_s"):
        if not always.schema[name].is_integer() or always[name].null_count():
            raise ValueError(f"ALWAYS {name} must be non-null integers")
    for name in ("collection_id", "phase"):
        if (
            always.schema[name] != pl.String
            or always[name].null_count()
            or (always[name] == "").any()
        ):
            raise ValueError(f"ALWAYS {name} must be nonempty strings")
    runs = always.select("replicate", "collection_id").unique().sort("replicate")
    if (
        runs.height != 30
        or runs["replicate"].to_list() != list(range(30))
        or runs["collection_id"].n_unique() != 30
    ):
        raise ValueError(
            "require exactly one unique ALWAYS collection per replicate 0..29"
        )
    if always.select("collection_id", *keys).is_duplicated().any():
        raise ValueError("duplicate ALWAYS collection/phase/time windows")

    calibration = always.filter(pl.col("replicate") < 15)
    evaluation = always.filter(pl.col("replicate") >= 15)
    long = calibration.select(
        "collection_id",
        *keys,
        *[
            pl.when(pl.col(name).cast(pl.Float64, strict=False).is_finite())
            .then(pl.col(name).cast(pl.Float64, strict=False))
            .otherwise(None)
            .alias(name)
            for name in variables
        ],
    ).unpivot(
        on=variables,
        index=["collection_id", *keys],
        variable_name="feature",
        value_name="value",
    )
    profiles = (
        long.group_by("feature", *keys)
        .agg(
            pl.col("collection_id")
            .filter(pl.col("value").is_not_null())
            .n_unique()
            .cast(pl.Int64)
            .alias("n"),
            pl.col("value").median(),
        )
        .with_columns(
            pl.when(pl.col("n") >= 8)
            .then(pl.col("value"))
            .otherwise(None)
            .alias("value")
        )
        .select("feature", *keys, "n", "value")
        .sort("feature", *keys)
    )
    lookup = evaluation.select(keys).join(
        profiles.pivot(on="feature", index=keys, values="value"),
        on=keys,
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    centered = []
    for name in variables:
        value = evaluation[name].cast(pl.Float64, strict=False) - lookup[name]
        centered.append(
            pl.when(value.is_finite()).then(value).otherwise(None).alias(name)
        )
    residuals = evaluation.with_columns(
        *centered, *[pl.lit(value).alias(name) for name, value in CONTEXT.items()]
    )

    support = []
    for phase in sorted(always["phase"].unique().to_list()):
        profile_phase = profiles.filter(pl.col("phase") == phase)
        input_phase = evaluation.filter(pl.col("phase") == phase)
        lookup_phase = lookup.filter(pl.col("phase") == phase)
        residual_phase = residuals.filter(pl.col("phase") == phase)
        for name in variables:
            points = profile_phase.filter(pl.col("feature") == name)
            finite = (
                input_phase[name]
                .cast(pl.Float64, strict=False)
                .is_finite()
                .fill_null(False)
            )
            covered = lookup_phase[name].is_not_null()
            support.append(
                {
                    "feature": name,
                    "phase": phase,
                    "profile_points": points.height,
                    "supported_profile_points": points["value"].is_not_null().sum(),
                    "evaluation_windows": input_phase.height,
                    "evaluation_finite_input": finite.sum(),
                    "evaluation_profile_covered": covered.sum(),
                    "evaluation_finite_input_without_profile": (
                        finite & ~covered
                    ).sum(),
                    "evaluation_finite_residuals": residual_phase[name]
                    .is_not_null()
                    .sum(),
                }
            )
    qc = {
        "expected_calibration_runs": 15,
        "expected_evaluation_runs": 15,
        "calibration_runs": calibration["collection_id"].n_unique(),
        "evaluation_runs": evaluation["collection_id"].n_unique(),
        "calibration_collections": runs.filter(pl.col("replicate") < 15).to_dicts(),
        "evaluation_collections": runs.filter(pl.col("replicate") >= 15).to_dicts(),
        "input_windows": windows.height,
        "ignored_non_always_windows": windows.height - always.height,
        "calibration_windows": calibration.height,
        "evaluation_windows": evaluation.height,
        "features": len(features),
        "profile_variables": variables,
        "min_profile_runs": 8,
        "profile_points": profiles.height,
        "supported_profile_points": profiles["value"].is_not_null().sum(),
        "profile_support": support,
    }
    return profiles, residuals, qc


def summarize_sensitivity(windows, catalog, *, seed=17, bootstrap_samples=2000):
    """Return (profiles, residuals, correlations, correlations_per_run, QC).

    All features enter the existing statistics engine, including target-linked
    diagnostics and unsupported catalog entries. No significance outcome is assumed.
    """
    for name, value, minimum in (
        ("seed", seed, 0),
        ("bootstrap_samples", bootstrap_samples, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    profiles, residuals, qc = phase_residuals(windows)
    summary, per_run = correlate.summarize_correlations(
        residuals,
        catalog,
        seed=seed,
        min_points=8,
        min_runs=10,
        bootstrap_samples=bootstrap_samples,
    )
    # Override the engine's generic raw-input/reference notes on the derived tables.
    summary = summary.with_columns(
        *[pl.lit(value).alias(name) for name, value in STATISTICS.items()]
    )
    per_run = per_run.join(
        residuals.select("collection_id", "replicate").unique(),
        on="collection_id",
        how="left",
        validate="m:1",
        maintain_order="left",
    ).with_columns(*[pl.lit(value).alias(name) for name, value in STATISTICS.items()])
    qc.update(
        correlation_tests=summary.height,
        eligible_tests=summary.filter(pl.col("status") == "ok").height,
        holm_below_005=summary.filter(pl.col("p_value_holm") < 0.05).height,
        non_target_linked_holm_below_005=summary.filter(
            (pl.col("p_value_holm") < 0.05) & ~pl.col("target_linked")
        ).height,
    )
    return profiles, residuals, summary, per_run, qc


def raw_diagnostics(windows: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    """Describe uncentered ALWAYS input, per run/phase and equal-run-weight macro.

    'Raw' here means original derived windows, NOT raw capture reads. Keep all
    input windows, independent of residual-profile coverage. No forecast is trained.
    """
    columns = (
        "observed_rss_mib",
        "observed_rss_1s_mib",
        "reference_mib",
        "reference_1s_mib",
        "excess_now_mib",
        "excess_1s_mib",
    )
    if missing := {"thp", "collection_id", "replicate", "phase", *columns} - set(
        windows.columns
    ):
        raise ValueError(f"missing diagnostic columns: {sorted(missing)}")
    if "analysis_variant" in windows.columns:
        raise ValueError("diagnostics require original derived windows, not residuals")
    rows = []
    for (cid, replicate, phase), frame in sorted(
        windows.filter(pl.col("thp") == "always")
        .partition_by(["collection_id", "replicate", "phase"], as_dict=True)
        .items()
    ):
        rss, future_rss, reference, future_reference, now, future = (
            frame[name].cast(pl.Float64, strict=False).to_numpy() for name in columns
        )
        rss_pairs = np.isfinite(rss) & np.isfinite(future_rss)
        unchanged = rss_pairs & (rss == future_rss)
        dynamic = (
            rss_pairs
            & np.isfinite(reference)
            & np.isfinite(future_reference)
            & np.isfinite(now)
            & np.isfinite(future)
        )
        changed = dynamic & (now != future)
        reference_only = changed & unchanged & (reference != future_reference)
        with np.errstate(invalid="ignore", over="ignore"):
            error = rss - future_reference - future
        error = error[np.isfinite(error)]
        row = {
            "collection_id": cid,
            "replicate": replicate,
            "phase": phase,
            "subset": "calibration" if replicate < 15 else "evaluation",
            "n_windows": frame.height,
            "n_observed_rss_windows": int(np.isfinite(rss).sum()),
            "n_current_target_windows": int(np.isfinite(now).sum()),
            "n_positive_excess_windows": int((np.isfinite(now) & (now > 0)).sum()),
            "n_rss_pairs": int(rss_pairs.sum()),
            "n_unchanged_rss_windows": int(unchanged.sum()),
            "n_dynamic_windows": int(dynamic.sum()),
            "n_changed_excess_windows": int(changed.sum()),
            "n_reference_only_change_windows": int(reference_only.sum()),
            "n_persistence_windows": len(error),
        }
        for name, values in (("observed_rss_mib", rss), ("excess_now_mib", now)):
            values = values[np.isfinite(values)]
            row[f"mean_{name}"] = float(np.mean(values)) if len(values) else None
            row[f"median_{name}"] = float(np.median(values)) if len(values) else None
        for name, numerator, denominator in (
            (
                "fraction_positive_excess",
                "n_positive_excess_windows",
                "n_current_target_windows",
            ),
            ("fraction_unchanged_rss", "n_unchanged_rss_windows", "n_rss_pairs"),
            (
                "fraction_changed_excess",
                "n_changed_excess_windows",
                "n_dynamic_windows",
            ),
            (
                "fraction_reference_only_change",
                "n_reference_only_change_windows",
                "n_dynamic_windows",
            ),
            (
                "fraction_reference_only_among_label_changes",
                "n_reference_only_change_windows",
                "n_changed_excess_windows",
            ),
        ):
            row[name] = row[numerator] / row[denominator] if row[denominator] else None
        row["persistence_mae_mib"] = (
            float(np.mean(np.abs(error))) if len(error) else None
        )
        scale = np.max(np.abs(error)) if len(error) else 0
        row["persistence_rmse_mib"] = (
            float(scale * np.sqrt(np.mean((error / scale) ** 2)))
            if scale
            else 0.0
            if len(error)
            else None
        )
        rows.append(row)
    if not rows:
        raise ValueError("diagnostics require ALWAYS windows")
    counts = [name for name in rows[0] if name.startswith("n_")]
    metrics = [
        name
        for name in rows[0]
        if name.startswith(("mean_", "median_", "fraction_", "persistence_"))
    ]
    per_run = pl.DataFrame(rows, schema_overrides=dict.fromkeys(metrics, pl.Float64))
    aggregates = []
    for phase in sorted(per_run["phase"].unique().to_list()):
        for subset in ("all_always", "calibration", "evaluation"):
            frame = per_run.filter(pl.col("phase") == phase)
            if subset != "all_always":
                frame = frame.filter(pl.col("subset") == subset)
            aggregates.append(
                {
                    "phase": phase,
                    "subset": subset,
                    "n_runs": frame.height,
                    "window_counts": {name: frame[name].sum() for name in counts},
                    "macro_means": {name: frame[name].mean() for name in metrics},
                    "metric_run_support": {
                        name: frame[name].count() for name in metrics
                    },
                }
            )
    report = {
        "input_kind": "original_uncentered_derived_windows",
        "scope": "ALWAYS only, all r00-r29 and each split separately; phases never pooled",
        "target_definition": correlate.NOTES["target_definition"],
        "aggregation": (
            "MACRO mean of per-run metrics, equal weight per independent collection "
            "within each phase/subset; NOT pooled-window MAE/RMSE. Null metrics are "
            "excluded, with metric_run_support reported. Window counts are descriptive totals."
        ),
        "fraction_denominators": {
            "fraction_positive_excess": "finite current excess windows",
            "fraction_unchanged_rss": "finite observed RSS now/future pairs; exact equality",
            "fraction_changed_excess": "finite RSS now/future, references now/future and excess now/future",
            "fraction_reference_only_change": "same complete dynamic windows as fraction_changed_excess",
            "fraction_reference_only_among_label_changes": "changed excess among complete dynamic windows",
        },
        "persistence_forecast": "predicted_future_excess = observed_rss_mib - reference_1s_mib",
        "persistence_error": "predicted_future_excess - actual excess_1s_mib; finite errors only",
        "limitations": [
            CONTEXT["analysis_notes"],
            "Persistence is an exploratory baseline, NOT validated predictive performance; it uses the fixed exploratory NEVER reference.",
            "When observed RSS is mostly unchanged, small errors reflect little actual dynamic variation, not learned predictive skill.",
            "Excess changes with unchanged observed RSS and changed NEVER reference are reference-curve artifacts, not actual RSS changes.",
            "Positive excess is not evidence of waste or a useful constant-positive classifier. Fractions and error outcomes are measured, not presumed.",
        ],
        "aggregates": aggregates,
    }
    return report, per_run


def _digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _publish(stage, output):
    # Linux RENAME_NOREPLACE also rejects destinations created after the path guard.
    rename = ctypes.CDLL(None, use_errno=True).renameat2
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(output), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def analyze(
    input_dir=DEFAULT_INPUT, output=DEFAULT_OUTPUT, *, seed=17, bootstrap_samples=2000
):
    """Read four derived inputs and atomically publish a new directory; return manifest.

    Relative paths are ROOT-relative, not cwd-relative. No raw input paths in the
    upstream manifest are followed. Existing paths (including empty directories and
    dangling symlinks) and overlapping input/output trees are never overwritten.
    """
    input_dir, requested_output = (ROOT / Path(path) for path in (input_dir, output))
    input_dir, output = input_dir.resolve(), requested_output.resolve()
    derived = ROOT / "data/derived/bloat"
    if (
        not output.is_relative_to(derived)
        or output == derived
        or output.exists()
        or requested_output.is_symlink()
        or output.is_relative_to(input_dir)
        or input_dir.is_relative_to(output)
    ):
        raise ValueError(
            "choose a new output directory under data/derived/bloat, separate from input"
        )
    if not input_dir.is_relative_to(derived) or not input_dir.is_dir():
        raise ValueError(
            "input must be an existing derived directory under data/derived/bloat"
        )
    inputs = [
        input_dir / name
        for name in (
            "windows.parquet",
            "feature_catalog.parquet",
            "correlations.parquet",
            "manifest.json",
        )
    ]
    for path in inputs:
        if not path.is_file() or not path.resolve().is_relative_to(input_dir):
            raise ValueError(
                f"require derived input file inside input directory: {path}"
            )
    hashes = {str(path.relative_to(ROOT)): _digest(path) for path in inputs}
    source_manifest = json.loads((input_dir / "manifest.json").read_text())
    if (
        source_manifest.get("kind") != "derived-bloat-correlations"
        or source_manifest.get("target_kind")
        != "observed_anonymous_plus_file_rss_excess"
        or source_manifest.get("always_collections") != 30
        or source_manifest.get("never_reference_collections") != 30
        or source_manifest.get("min_reference_runs") != 15
        or source_manifest.get("forecast_horizon_seconds") != 1
    ):
        raise ValueError(
            "require the completed 30 ALWAYS / 30 NEVER fixed-reference analysis"
        )
    windows = pl.read_parquet(input_dir / "windows.parquet")
    catalog = pl.read_parquet(input_dir / "feature_catalog.parquet").to_dicts()
    profiles, residuals, summary, per_run, qc = summarize_sensitivity(
        windows,
        catalog,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    diagnostics, per_run_diagnostics = raw_diagnostics(windows)
    source_root = Path(__file__).resolve().parents[2]
    report = {
        "kind": "derived-bloat-phase-sensitivity",
        "version": 1,
        **CONTEXT,
        "input_directory": str(input_dir.relative_to(ROOT)),
        "output_directory": str(output.relative_to(ROOT)),
        "seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples),
        "min_points_per_run": 8,
        "min_runs_for_inference": 10,
        "statistics": STATISTICS,
        "qc": qc,
        "input_files": hashes,
        "analysis_sources": {
            str(path.relative_to(source_root)): _digest(path)
            for path in (Path(__file__).resolve(), Path(correlate.__file__).resolve())
        },
        "source_limitations": source_manifest.get("limitations", []),
        "plots": "not generated; full correlation and per-run tables retained",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".phase-sensitivity-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        for name, frame in (
            ("phase_profiles", profiles),
            ("residual_windows", residuals),
            ("correlations", summary),
            ("correlations_per_run", per_run),
        ):
            frame.write_parquet(stage / f"{name}.parquet")
            frame.write_csv(stage / f"{name}.csv")
        per_run_diagnostics.write_csv(stage / "per_run_diagnostics.csv")
        for name, value in (("diagnostics", diagnostics), ("manifest", report)):
            (stage / f"{name}.json").write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n"
            )
        if hashes != {str(path.relative_to(ROOT)): _digest(path) for path in inputs}:
            raise ValueError("derived inputs changed during sensitivity analysis")
        _publish(stage, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    report = analyze(**vars(parser.parse_args(argv)))
    print(
        json.dumps(
            {
                "output_directory": report["output_directory"],
                "analysis_notes": report["analysis_notes"],
                "qc": {
                    name: value
                    for name, value in report["qc"].items()
                    if name != "profile_support"
                },
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
