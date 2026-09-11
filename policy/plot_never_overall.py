"""THP=never slowdown and throughput loss with unadjusted run-bootstrap intervals."""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = (
    ("all", "All", "#c66125"),
    ("selected-control", "Selected", "#8b4f9e"),
    ("original-reduced", "Reduced", "#2675a6"),
    ("original-mpmo", "Reduced+\nMPMO", "#548b41"),
    ("none", "None", "#666666"),
)
METRICS = (
    ("RunTime(ms)", "ms", "Runtime slowdown (%)", "never_runtime_slowdown"),
    ("Throughput(ops/sec)", "ops/sec", "Throughput loss (%)", "never_throughput_loss"),
)


def bootstrap_effects(samples, *, seed=17, bootstrap_samples=10000):
    """Resample independent collections, preserving the two metrics within each run."""
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    modes = [item[0] for item in CONFIGURATIONS]
    if set(samples) != set(modes):
        raise ValueError("expected all five configurations")
    arrays = {mode: np.asarray(samples[mode], dtype=float) for mode in modes}
    if any(
        values.shape != (30, 2) or not np.isfinite(values).all() or np.any(values <= 0)
        for values in arrays.values()
    ):
        raise ValueError(
            "expected 30 positive, finite runtime/throughput pairs per mode"
        )

    rng = np.random.default_rng(seed)
    baseline = arrays["none"]
    baseline_mean = baseline.mean(axis=0)
    # The same baseline draw is shared across contrasts; metrics remain paired
    # within a collection, but configuration and baseline collections are unpaired.
    baseline_draws = rng.integers(0, 30, size=(bootstrap_samples, 30))
    baseline_boot = baseline[baseline_draws].mean(axis=1)
    directions = np.array([100.0, -100.0])
    result = {}
    for mode in modes:
        if mode == "none":
            result[mode] = (np.zeros(2), np.zeros(2), np.zeros(2))
            continue
        values = arrays[mode]
        draws = rng.integers(0, 30, size=(bootstrap_samples, 30))
        estimates = (values[draws].mean(axis=1) / baseline_boot - 1) * directions
        low, high = np.quantile(estimates, [0.025, 0.975], axis=0)
        point = (values.mean(axis=0) / baseline_mean - 1) * directions
        result[mode] = (point, low, high)
    return result


def plot_never_overall(directory, *, seed=17, bootstrap_samples=10000):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    directory = (ROOT / directory).resolve()
    raw = pl.read_csv(directory / "raw_metrics.csv").filter(
        (pl.col("thp") == "never")
        & (pl.col("phase") == "run")
        & (pl.col("section") == "OVERALL")
        & pl.col("metric").is_in([item[0] for item in METRICS])
        & pl.col("mode").is_in([item[0] for item in CONFIGURATIONS])
    )
    if raw.select("outer_iteration", "iteration").unique().rows() != [(0, 0)]:
        raise ValueError("expected one run-phase repeat position (outer=0, inner=0)")
    rows, identities, values_by_metric = [], {}, {}
    for metric, unit, _, _ in METRICS:
        for mode, label, _ in CONFIGURATIONS:
            samples = raw.filter(
                (pl.col("mode") == mode) & (pl.col("metric") == metric)
            ).sort("collection_id")
            ids = set(samples["collection_id"].to_list())
            values = samples["value"].cast(pl.Float64)
            if (
                samples.height != 30
                or len(ids) != 30
                or None in ids
                or values.null_count()
                or not values.is_finite().all()
                or (values <= 0).any()
                or samples["unit"].unique().to_list() != [unit]
            ):
                raise ValueError(
                    f"expected 30 distinct, positive, finite {mode}/{metric} samples in {unit}"
                )
            if mode in identities and identities[mode] != ids:
                raise ValueError(f"runtime/throughput collection mismatch: {mode}")
            identities[mode] = ids
            values_by_metric[mode, metric] = values.to_numpy()
            n = samples.height
            mean, std = values.mean(), values.std(ddof=1)
            sem = std / np.sqrt(n)
            rows.append(
                {
                    "thp": "never",
                    "phase": "run",
                    "mode": mode,
                    "label": label.replace("\n", ""),
                    "metric": metric,
                    "unit": unit,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                }
            )
    if len(set.union(*identities.values())) != 150:
        raise ValueError("a collection is reused across configurations")
    samples = {
        mode: np.column_stack([values_by_metric[mode, item[0]] for item in METRICS])
        for mode, _, _ in CONFIGURATIONS
    }
    effects = bootstrap_effects(samples, seed=seed, bootstrap_samples=bootstrap_samples)
    for row in rows:
        metric_index = [item[0] for item in METRICS].index(row["metric"])
        point, low, high = effects[row["mode"]]
        row.update(
            baseline="None",
            baseline_n=30,
            baseline_mean=float(samples["none"][:, metric_index].mean()),
            effect=METRICS[metric_index][3].removeprefix("never_"),
            effect_percent=float(point[metric_index]),
            ci95_low_percent=float(low[metric_index]),
            ci95_high_percent=float(high[metric_index]),
            ci95_method="reference_identity"
            if row["mode"] == "none"
            else "independent_run_bootstrap_percentile_unadjusted",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=seed,
        )
    summary = pl.DataFrame(rows)
    names = [
        f"{stem}.{suffix}" for _, _, _, stem in METRICS for suffix in ("png", "svg")
    ] + ["never_slowdown_throughput_loss.csv"]
    for name in names:
        path = directory / name
        if (
            path.is_symlink()
            or path.exists()
            and (not path.is_file() or path.stat().st_nlink != 1)
        ):
            raise ValueError(f"unsafe plot output: {path}")

    with tempfile.TemporaryDirectory(prefix=".never-bars-", dir=directory) as temporary:
        staged = Path(temporary)
        positions = np.arange(len(CONFIGURATIONS))
        for metric, _, label, stem in METRICS:
            fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
            try:
                selected = summary.filter(pl.col("metric") == metric)
                points = selected["effect_percent"].to_numpy()
                lows = selected["ci95_low_percent"].to_numpy()
                highs = selected["ci95_high_percent"].to_numpy()
                lower = min(0.0, float(np.min(lows)), float(np.min(points)))
                upper = max(0.0, float(np.max(highs)), float(np.max(points)))
                span = upper - lower or 1.0
                ax.bar(
                    positions,
                    points,
                    width=0.64,
                    color=[item[2] for item in CONFIGURATIONS],
                    zorder=3,
                )
                # Draw percentile endpoints directly: intervals need not be
                # symmetric about, or even contain, the original point estimate.
                ax.errorbar(
                    positions,
                    (lows + highs) / 2,
                    yerr=(highs - lows) / 2,
                    fmt="none",
                    ecolor="#20252b",
                    elinewidth=1.3,
                    capsize=5,
                    capthick=1.3,
                    zorder=4,
                )
                ax.set_ylabel(label, fontsize=11)
                ax.set_xticks(
                    positions, [item[1] for item in CONFIGURATIONS], fontsize=10
                )
                ax.set_ylim(
                    lower - 0.08 * span if lower < 0 else 0,
                    upper + 0.08 * span if upper > 0 else 0 if lower < 0 else 1,
                )
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                ax.set_axisbelow(True)
                ax.axhline(0, color="#20252b", linewidth=0.8, zorder=2)
                ax.grid(axis="y", alpha=0.2)
                ax.spines[["top", "right"]].set_visible(False)
                for suffix in ("png", "svg"):
                    fig.savefig(
                        staged / f"{stem}.{suffix}",
                        dpi=180,
                        facecolor="white",
                    )
            finally:
                plt.close(fig)
        summary.write_csv(staged / names[-1])
        for name in names:
            os.replace(staged / name, directory / name)
    return summary, [directory / name for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Existing matrix analysis directory",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    summary, paths = plot_never_overall(**vars(args))
    print(summary)
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
