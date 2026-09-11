"""Exploratory full-probe correlations with an observed-resident-memory proxy.

Outputs are derived data, never raw collection tables or model-training evidence.
The proxy is anonymous + file RSS excess; unobserved shmem is not set to zero.
"""

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from audit import ROOT, full_runs, read_table, saved_path, table_files
from correlate import NOTES as STATISTICS_NOTES
from correlate import summarize_correlations
from features import build_features

SECOND = 1_000_000_000
MIB_PER_PAGE = 4096 / 2**20
FEATURE_TABLES = (
    "quanta_runtime",
    "quanta_queued_time",
    "file_data",
    "unmap_range",
    "zswap_runtime",
    "trace_mm_khugepaged_scan_pmd",
    "dtlb_misses",
    "itlb_misses",
    "tlb_flushes",
    "dtlb_walk_duration",
    "memory_usage",
)
TARGET_KIND = "observed_anonymous_plus_file_rss_excess"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def rss_component(rss, member, ticks):
    """Causal gauge lookup; conflicting/invalid observations invalidate the level."""
    values, ages = np.full(len(ticks), np.nan), np.full(len(ticks), np.nan)
    if not {"ts_ns", "member", "count"} <= set(rss.columns):
        return values, ages
    if not rss.schema["ts_ns"].is_integer() or not rss.schema["count"].is_integer():
        raise ValueError("RSS timestamps and component levels must be integers")
    rows = (
        rss.filter((pl.col("member") == member) & (pl.col("ts_ns") >= 0))
        .group_by("ts_ns")
        .agg(
            pl.col("count").min().alias("lo"),
            pl.col("count").max().alias("hi"),
            pl.col("count").null_count().alias("nulls"),
        )
        .sort("ts_ns")
    )
    if rows.is_empty():
        return values, ages
    valid = (
        (pl.col("lo") >= 0) & (pl.col("lo") == pl.col("hi")) & (pl.col("nulls") == 0)
    )
    levels = (
        rows.select(pl.when(valid).then(pl.col("lo") * MIB_PER_PAGE).otherwise(None))
        .to_series()
        .to_numpy()
    )
    times = rows["ts_ns"].to_numpy()
    indices = np.searchsorted(times, ticks, side="right") - 1
    present = indices >= 0
    values[present] = levels[indices[present]]
    ages[present] = (ticks[present] - times[indices[present]]) / SECOND
    ages[~np.isfinite(values)] = np.nan
    return values, ages


def rss_windows(rss, start_ns, end_ns):
    """One-second nonoverlapping grids; future values are labels, never features."""
    elapsed = np.arange(1, (end_ns - start_ns) // SECOND + 1, dtype=np.int64)
    ticks = start_ns + elapsed * SECOND
    anon, anon_age = rss_component(rss, "MM_ANONPAGES", ticks)
    file, file_age = rss_component(rss, "MM_FILEPAGES", ticks)
    before_anon, _ = rss_component(rss, "MM_ANONPAGES", ticks - SECOND)
    before_file, _ = rss_component(rss, "MM_FILEPAGES", ticks - SECOND)
    future_anon, _ = rss_component(rss, "MM_ANONPAGES", ticks + SECOND)
    future_file, _ = rss_component(rss, "MM_FILEPAGES", ticks + SECOND)
    future = future_anon + future_file
    future[ticks + SECOND > end_ns] = np.nan
    values = {
        "observed_mib": anon + file,
        "anon_mib": anon,
        "file_mib": file,
        "observed_change_1s_mib": anon + file - before_anon - before_file,
        "anon_change_1s_mib": anon - before_anon,
        "file_change_1s_mib": file - before_file,
        "last_update_age_s": np.minimum(anon_age, file_age),
    }
    features = {f"f__rss__redis__{name}": value for name, value in values.items()}
    return ticks, elapsed, features, future


def add_reference(windows, min_reference_runs=15):
    """Use NEVER only for the fixed reference, without cross-run feature joins."""
    if not 1 <= min_reference_runs <= 30:
        raise ValueError("minimum NEVER support must be between 1 and 30")
    keys = ["phase", "elapsed_s"]
    unique = ["collection_id", *keys]
    if windows.select(unique).unique().height != windows.height:
        raise ValueError("duplicate collection/phase/time windows")
    never = windows.filter(
        (pl.col("thp") == "never") & pl.col("observed_rss_mib").is_finite()
    )
    reference = (
        never.group_by(keys)
        .agg(
            pl.col("collection_id").n_unique().alias("reference_n"),
            pl.col("observed_rss_mib").median().alias("reference_mib"),
            pl.col("observed_rss_mib")
            .quantile(0.25, interpolation="linear")
            .alias("reference_q25_mib"),
            pl.col("observed_rss_mib")
            .quantile(0.75, interpolation="linear")
            .alias("reference_q75_mib"),
        )
        .sort(keys)
    )
    reference = reference.with_columns(
        *[
            pl.when(pl.col("reference_n") >= min_reference_runs)
            .then(pl.col(name))
            .otherwise(None)
            .alias(name)
            for name in ("reference_mib", "reference_q25_mib", "reference_q75_mib")
        ]
    )
    future = reference.select(
        "phase",
        (pl.col("elapsed_s") - 1).alias("elapsed_s"),
        pl.col("reference_mib").alias("reference_1s_mib"),
        pl.col("reference_n").alias("reference_1s_n"),
    )
    result = windows.join(reference, on=keys, how="left").join(
        future, on=keys, how="left"
    )
    result = result.with_columns(
        pl.when(pl.col("thp") == "always")
        .then(pl.col("observed_rss_mib") - pl.col("reference_mib"))
        .otherwise(None)
        .alias("excess_now_mib"),
        pl.when(pl.col("thp") == "always")
        .then(pl.col("observed_rss_1s_mib") - pl.col("reference_1s_mib"))
        .otherwise(None)
        .alias("excess_1s_mib"),
    ).with_columns(
        (pl.col("excess_1s_mib") - pl.col("excess_now_mib")).alias(
            "excess_change_1s_mib"
        )
    )
    return result.sort(["collection_id", "phase", "elapsed_s"]), reference


def plot_results(windows, reference, summary, per_run, directory):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for (thp, _), frame in windows.partition_by(
        ["thp", "collection_id"], as_dict=True
    ).items():
        color = "#c66125" if thp == "always" else "#666666"
        axes[0].plot(
            frame["elapsed_s"], frame["observed_rss_mib"], color=color, alpha=0.25
        )
        if thp == "always":
            axes[1].plot(
                frame["elapsed_s"], frame["excess_now_mib"], color=color, alpha=0.3
            )
    x = reference["elapsed_s"].to_numpy()
    axes[0].plot(x, reference["reference_mib"], color="black", label="NEVER median")
    axes[0].fill_between(
        x,
        reference["reference_q25_mib"].to_numpy(),
        reference["reference_q75_mib"].to_numpy(),
        color="black",
        alpha=0.15,
        label="NEVER IQR (not a confidence interval)",
    )
    axes[0].set_ylabel("Observed anonymous + file RSS (MiB)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("ALWAYS minus NEVER median (MiB)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    for ax in axes:
        ax.set_xlabel("Seconds since run-client launch (includes preparation)")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Full-probe RSS proxy: orange ALWAYS / gray NEVER; one line per collection\nShared-memory RSS is unobserved; this is not verified total RSS or measured waste"
    )
    fig.savefig(directory / "rss_reference.png", dpi=140)
    plt.close(fig)

    candidates = summary.filter(
        (pl.col("method") == "spearman")
        & ~pl.col("target_linked")
        & (pl.col("status") == "ok")
    )
    ranking = candidates.group_by("feature").agg(
        pl.col("median_r").abs().max().alias("strength")
    )
    names = ranking.sort("strength", descending=True).head(20)["feature"].to_list()
    columns = [
        (target, adjustment)
        for target in ("excess_now_mib", "excess_1s_mib", "excess_change_1s_mib")
        for adjustment in ("raw", "adjusted")
    ]
    fig, ax = plt.subplots(
        figsize=(14, max(4, len(names) * 0.35 + 2)), constrained_layout=True
    )
    if names:
        matrix = np.full((len(names), len(columns)), np.nan)
        significant = np.zeros(matrix.shape, dtype=bool)
        for row in candidates.to_dicts():
            key = (row["target"], row["adjustment"])
            if row["feature"] in names and key in columns:
                i, j = names.index(row["feature"]), columns.index(key)
                matrix[i, j] = row["median_r"]
                significant[i, j] = row["p_value_holm"] < 0.05
        image = ax.imshow(
            np.ma.masked_invalid(matrix), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
        )
        fig.colorbar(image, ax=ax, label="Median within-run Spearman correlation")
        ax.set_yticks(
            range(len(names)),
            [name.removeprefix("f__").replace("__", " / ") for name in names],
            fontsize=8,
        )
        ax.set_xticks(
            range(6),
            [
                f"{target.replace('excess_', '').replace('_mib', '')}\n{adjustment}"
                for target, adjustment in columns
            ],
            fontsize=9,
        )
        for i, j in np.ndindex(matrix.shape):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}" + ("*" if significant[i, j] else ""),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    else:
        ax.text(
            0.5,
            0.5,
            "No features have sufficient independent-run support",
            ha="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    ax.set_title(
        "Exploratory non-target-linked features (top 20 by observed |correlation|)\n* Holm-adjusted independent-run sign-test p < 0.05; no predictive validation"
    )
    fig.savefig(directory / "correlation_heatmap.png", dpi=140)
    plt.close(fig)

    leading = (
        candidates.filter(
            (pl.col("target") == "excess_change_1s_mib")
            & (pl.col("adjustment") == "adjusted")
        )
        .sort(pl.col("median_r").abs(), descending=True)
        .head(3)
    )
    fig, axes = plt.subplots(
        max(1, leading.height),
        1,
        figsize=(12, max(4, leading.height * 3.5)),
        squeeze=False,
        constrained_layout=True,
    )
    if not leading.height:
        axes[0, 0].text(
            0.5,
            0.5,
            "No eligible adjusted future-change associations",
            ha="center",
            transform=axes[0, 0].transAxes,
        )
    for i, row in enumerate(leading.to_dicts()):
        examples = per_run.filter(
            (pl.col("feature") == row["feature"])
            & (pl.col("target") == row["target"])
            & (pl.col("method") == "spearman")
            & (pl.col("adjustment") == "adjusted")
            & (pl.col("status") == "ok")
        ).sort((pl.col("r") - row["median_r"]).abs())
        cid = examples["collection_id"][0]
        frame = windows.filter(pl.col("collection_id") == cid).sort("elapsed_s")
        ax = axes[i, 0]
        other = ax.twinx()
        ax.plot(frame["elapsed_s"], frame[row["feature"]], color="#2675a6", marker=".")
        other.plot(
            frame["elapsed_s"],
            frame["excess_change_1s_mib"],
            color="#c66125",
            marker=".",
        )
        ax.set_ylabel(f"Feature ({row['unit']})", color="#2675a6")
        other.set_ylabel("Observed next-second excess change (MiB)", color="#c66125")
        ax.set_xlabel("Seconds since run-client launch")
        ax.set_title(
            f"{row['feature'].removeprefix('f__')}\nRepresentative run nearest median association: {cid}",
            fontsize=8,
        )
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Descriptive example traces, not held-out forecasts; axes have different units"
    )
    fig.savefig(directory / "example_traces.png", dpi=140)
    plt.close(fig)


def analyze(
    experiment, audit_directory, output, *, phase="run", seed=17, bootstrap_samples=2000
):
    experiment, audit_directory, output = map(
        saved_path, (experiment, audit_directory, output)
    )
    if not output.is_relative_to(ROOT / "data/derived/bloat") or output.exists():
        raise ValueError("choose a new output directory under data/derived/bloat")
    if phase not in ("run", "load"):
        raise ValueError("phases must be analyzed separately")
    with (experiment / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        plan, events = full_runs(experiment)
        audit_report = json.loads((audit_directory / "audit.json").read_text())
        if (
            audit_report["source_plan_sha256"] != digest(experiment / "plan.json")
            or audit_report["audited_collections"] != 60
        ):
            raise ValueError("the source cohort needs a complete matching audit")
        runs = {
            row["collection_id"]: row
            for row in pl.read_parquet(audit_directory / "runs.parquet").to_dicts()
        }
        phases = pl.read_parquet(audit_directory / "phases.parquet").filter(
            pl.col("phase") == phase
        )
        catalog, frames, quality, consumed = {}, [], {}, set()
        for event in events:
            cid = event["collection_id"]
            run = runs[cid]
            directory = saved_path(event["collection_dir"])
            if (
                run["collection_directory"] != str(directory.relative_to(ROOT))
                or run["thp"] != event["thp"]
            ):
                raise ValueError("audit identity mismatch")
            invocations = phases.filter(pl.col("collection_id") == cid).to_dicts()
            if len(invocations) != 1:
                raise ValueError(
                    "this initial analysis requires one invocation per phase/collection"
                )
            invocation = invocations[0]
            rss = read_table(directory, "mm_rss_stat").filter(
                (pl.col("tgid") == run["redis_pid"])
                & pl.col("ts_ns").is_between(
                    run["process_start_ns"], run["process_end_ns"]
                )
            )
            if rss["collection_id"].unique().to_list() != [cid]:
                raise ValueError("RSS collection identity mismatch")
            ticks, elapsed, values, future = rss_windows(
                rss, invocation["start_ns"], invocation["end_ns"]
            )
            frame_values = {
                "collection_id": [cid] * len(ticks),
                "thp": [event["thp"]] * len(ticks),
                "cohort_id": [plan["experiment"]] * len(ticks),
                "target_kind": [TARGET_KIND] * len(ticks),
                "replicate": [event["replicate"]] * len(ticks),
                "phase": [phase] * len(ticks),
                "elapsed_s": elapsed,
                "time_ns": ticks,
                "redis_pid": [run["redis_pid"]] * len(ticks),
                "observed_rss_mib": values["f__rss__redis__observed_mib"],
                "observed_rss_1s_mib": future,
                **values,
            }
            for name in values:
                catalog[name] = {
                    "name": name,
                    "group": "rss",
                    "scope": "redis",
                    "unit": "s" if name.endswith("age_s") else "MiB",
                    "description": "Observed resident-component history; algebraically related to the target",
                    "target_linked": True,
                    "availability": "last component observations at/before prediction time; no backward fill",
                    "limitations": "Anonymous + file only; unknown shmem; not an independent predictor; losses unknown",
                }
            if event["thp"] == "always":
                tables = {}
                for name in FEATURE_TABLES:
                    paths = table_files(directory, name)
                    if paths:
                        tables[name] = read_table(directory, name)
                        consumed.update(paths)
                        if tables[name].height and (
                            "collection_id" not in tables[name].columns
                            or tables[name]["collection_id"].unique().to_list() != [cid]
                        ):
                            raise ValueError(f"mismatched feature collection: {name}")
                info = read_table(directory, "system_info").to_dicts()[0]
                probe_values, entries, qc = build_features(
                    tables,
                    run["redis_pid"],
                    ticks,
                    phase_start_ns=invocation["start_ns"],
                    logical_cpus=info["logical_cores"],
                )
                frame_values.update(probe_values)
                for entry in entries:
                    entry = entry | {
                        "limitations": "; ".join(entry["limitations"])
                        if isinstance(entry["limitations"], list)
                        else entry["limitations"]
                    }
                    if entry["name"] in catalog and catalog[entry["name"]] != entry:
                        raise ValueError(
                            "inconsistent feature definitions across collections"
                        )
                    catalog[entry["name"]] = entry
                quality[cid] = qc
            frames.append(pl.DataFrame(frame_values))
            for name in (
                "mm_rss_stat",
                "system_info",
                "ycsb_invocations",
                "ycsb_metrics",
                "process_metadata",
                "process_trace",
            ):
                consumed.update(table_files(directory, name))
            consumed.update((saved_path(event["stdout"]), saved_path(event["stderr"])))
            print(
                f"Prepared {len(frames)}/60: {event['thp']} r{event['replicate']:02d}",
                flush=True,
            )
        windows = pl.concat(frames, how="diagonal_relaxed").with_columns(
            pl.col(pl.Float64).fill_nan(None)
        )
        windows, reference = add_reference(windows)
        windows = windows.with_columns(pl.col(pl.Float64).fill_nan(None))
        entries = [catalog[name] for name in sorted(catalog)]
        summary, per_run = summarize_correlations(
            windows, entries, seed=seed, bootstrap_samples=bootstrap_samples
        )
        consumed.update(
            experiment / name
            for name in ("plan.json", "manifest.json", "results.jsonl")
        )
        consumed.update(audit_directory.glob("*.parquet"))
        consumed.add(audit_directory / "audit.json")
        input_hashes = {
            str(path.relative_to(ROOT)): digest(path) for path in sorted(consumed)
        }
        always = windows.filter(pl.col("thp") == "always")
        report = {
            "kind": "derived-bloat-correlations",
            "version": 1,
            "target_kind": TARGET_KIND,
            "phase": phase,
            "collections": 60,
            "always_collections": 30,
            "never_reference_collections": 30,
            "window_seconds": 1,
            "grid_step_seconds": 1,
            "forecast_horizon_seconds": 1,
            "min_reference_runs": 15,
            "min_points_per_run": 8,
            "min_runs_for_inference": 10,
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "source_experiment": str(experiment.relative_to(ROOT)),
            "windows": windows.height,
            "always_windows": always.height,
            "valid_current_target_windows": always["excess_now_mib"]
            .is_not_null()
            .sum(),
            "valid_future_target_windows": always["excess_1s_mib"].is_not_null().sum(),
            "features": len(entries),
            "correlation_tests": summary.height,
            "eligible_tests": summary.filter(pl.col("status") == "ok").height,
            "holm_below_005": summary.filter(pl.col("p_value_holm") < 0.05).height,
            "non_target_linked_holm_below_005": summary.filter(
                (pl.col("p_value_holm") < 0.05) & ~pl.col("target_linked")
            ).height,
            "statistics": STATISTICS_NOTES,
            "limitations": [
                "Target is only observed anonymous + file resident memory, not verified total RSS or hugepage waste.",
                "Missing shmem and swap observations are not imputed to zero.",
                "Only ALWAYS windows enter correlations; NEVER builds the fixed exploratory phase/elapsed median reference.",
                "No identical-operation-sequence claim; elapsed time is relative to launcher start and includes Maven preparation.",
                "Never reference is not an observed counterfactual. Its IQR is descriptive; correlation CI/p-values condition on the reference.",
                "No training/held-out prediction or causal intervention is evaluated.",
                "Unsafe completion-selected/misjoined tables are excluded; capture losses and unknown counter gaps limit conclusions.",
                "Memory context uses a conservative BOOTTIME upper bound and prior completed read; exact clock lag is uncalibrated.",
            ],
            "input_files": input_hashes,
            "analysis_sources": {
                str(path.relative_to(ROOT)): digest(path)
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".bloat-", dir=output.parent
        ) as temporary:
            stage = Path(temporary)
            for name, frame in (
                ("windows", windows),
                ("never_reference", reference),
                ("correlations", summary),
                ("correlations_per_run", per_run),
                ("feature_catalog", pl.DataFrame(entries)),
            ):
                frame.write_parquet(stage / f"{name}.parquet")
                frame.write_csv(stage / f"{name}.csv")
            (stage / "feature_quality.json").write_text(
                json.dumps(quality, indent=2, allow_nan=False) + "\n"
            )
            (stage / "manifest.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            plot_results(windows, reference, summary, per_run, stage)
            os.replace(stage, output)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in ("input_files", "analysis_sources", "statistics")
                },
                indent=2,
            )
        )
        print(output)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("data/experiments/redis-rss-dtlb-20260908"),
    )
    parser.add_argument(
        "--audit-directory",
        type=Path,
        default=Path("data/derived/bloat/full-probes-20260908/audit"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/bloat/full-probes-20260908/correlations"),
    )
    parser.add_argument("--phase", choices=("run", "load"), default="run")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    analyze(**vars(args))


if __name__ == "__main__":
    main()
