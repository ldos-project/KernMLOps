"""Offline, per-metric comparisons of saved YCSB collections.

One collection contributes at most one observation for a given phase, repeat
position, operation, and metric. A single collection on each side provides
descriptive differences, not enough samples for a t-test.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from scipy.stats import ttest_ind_from_stats

KEY_COLUMNS = (
    "benchmark_name",
    "phase",
    "outer_iteration",
    "iteration",
    "section",
    "metric",
    "unit",
)
SCHEMA = {
    "benchmark_name": pl.String,
    "phase": pl.String,
    "outer_iteration": pl.Int64,
    "iteration": pl.Int64,
    "section": pl.String,
    "metric": pl.String,
    "unit": pl.String,
    "left_collections": pl.String,
    "right_collections": pl.String,
    "left_n": pl.UInt64,
    "right_n": pl.UInt64,
    "left_missing": pl.UInt64,
    "right_missing": pl.UInt64,
    "left_invalid": pl.UInt64,
    "right_invalid": pl.UInt64,
    "left_mean": pl.Float64,
    "right_mean": pl.Float64,
    "left_std": pl.Float64,
    "right_std": pl.Float64,
    "difference": pl.Float64,
    "change_percent": pl.Float64,
    "t_statistic": pl.Float64,
    "p_value": pl.Float64,
    "p_value_holm": pl.Float64,
    "status": pl.String,
    "notes": pl.String,
}


def welch_ttest(left: Iterable[float], right: Iterable[float]) -> dict[str, Any]:
    """Two-sided Welch test of independent samples, with right-minus-left effects.

    Non-finite/missing inputs are errors, not silently discarded observations.
    Insufficient or zero-variance samples retain their descriptive statistics
    but have null inferential statistics. No significance threshold is applied.
    """
    try:
        a, b = [float(value) for value in left], [float(value) for value in right]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("samples must contain finite numbers") from exc
    if not all(math.isfinite(value) for sample in (a, b) for value in sample):
        raise ValueError("samples must contain finite numbers")
    mean_a = statistics.mean(a) if a else None
    mean_b = statistics.mean(b) if b else None
    result = {
        "left_n": len(a),
        "right_n": len(b),
        "left_mean": mean_a,
        "right_mean": mean_b,
        "left_std": None,
        "right_std": None,
        "difference": None,
        "change_percent": None,
        "t_statistic": None,
        "p_value": None,
        "status": "insufficient_samples",
    }
    if mean_a is not None and mean_b is not None:
        difference = mean_b - mean_a
        if math.isfinite(difference):
            result["difference"] = difference
        if mean_a != 0:
            change = 100 * (mean_b / mean_a - 1)
            if math.isfinite(change):
                result["change_percent"] = change
    try:
        std_a = statistics.stdev(a) if len(a) >= 2 else None
        std_b = statistics.stdev(b) if len(b) >= 2 else None
    except (OverflowError, ValueError):
        result["status"] = "non_finite_result"
        return result
    result["left_std"], result["right_std"] = std_a, std_b
    if std_a is None or std_b is None:
        return result
    if not math.isfinite(std_a) or not math.isfinite(std_b):
        result.update(status="non_finite_result", left_std=None, right_std=None)
        return result
    if std_a == 0 and std_b == 0:
        result["status"] = "zero_variance"
        return result
    test = ttest_ind_from_stats(
        mean1=mean_b,
        std1=std_b,
        nobs1=len(b),
        mean2=mean_a,
        std2=std_a,
        nobs2=len(a),
        equal_var=False,
    )
    statistic, p_value = float(test.statistic), float(test.pvalue)
    if not math.isfinite(statistic) or not math.isfinite(p_value):
        result["status"] = "non_finite_result"
        return result
    result.update(t_statistic=statistic, p_value=p_value, status="ok")
    return result


@dataclass(frozen=True)
class _Observation:
    value: float | None
    problem: str | None
    argv: tuple[str, ...]


@dataclass
class _Run:
    collection_id: str
    observations: dict[tuple, _Observation]


def _read_run(path: Path, phase: str) -> _Run:
    try:
        invocations = pl.read_parquet(path / "ycsb_invocations.end.parquet")
        metrics = pl.read_parquet(path / "ycsb_metrics.end.parquet")
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"{path}: cannot read saved YCSB tables: {exc}") from exc
    required_invocations = {
        "collection_id",
        "invocation_id",
        "benchmark_name",
        "phase",
        "iteration",
        "outer_iteration",
        "argv",
        "exit_code",
        "execution_state",
        "parse_state",
    }
    required_metrics = {
        "collection_id",
        "invocation_id",
        "section",
        "metric",
        "unit",
        "value",
    }
    for table, required in (
        (invocations, required_invocations),
        (metrics, required_metrics),
    ):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
    ids = invocations["collection_id"].unique().to_list()
    if len(ids) != 1 or not isinstance(ids[0], str) or not ids[0].strip():
        raise ValueError(f"{path}: expected one nonempty collection ID")
    metadata = {}
    for invocation in invocations.iter_rows(named=True):
        if (
            not isinstance(invocation["benchmark_name"], str)
            or not invocation["benchmark_name"].strip()
        ):
            raise ValueError(f"{path}: missing benchmark identity")
        for column in ("invocation_id", "iteration", "outer_iteration"):
            value = invocation[column]
            if column == "outer_iteration" and value is None:
                continue
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{path}: invalid {column}")
        identifier = (invocation["collection_id"], invocation["invocation_id"])
        if identifier in metadata:
            raise ValueError(f"{path}: duplicate invocation ID {identifier}")
        if invocation["phase"] not in {"load", "run"}:
            raise ValueError(f"{path}: invalid invocation phase")
        metadata[identifier] = invocation
    observations = {}
    for metric in metrics.iter_rows(named=True):
        if type(metric["invocation_id"]) is not int:
            raise ValueError(f"{path}: invalid metric invocation ID")
        for column in ("section", "metric"):
            if not isinstance(metric[column], str) or not metric[column].strip():
                raise ValueError(f"{path}: missing {column} identity")
        if metric["unit"] is not None and (
            not isinstance(metric["unit"], str) or not metric["unit"].strip()
        ):
            raise ValueError(f"{path}: invalid metric unit")
        identifier = (metric["collection_id"], metric["invocation_id"])
        if identifier not in metadata:
            raise ValueError(
                f"{path}: metric references unknown invocation {identifier}"
            )
        invocation = metadata[identifier]
        if phase != "all" and invocation["phase"] != phase:
            continue
        key = (
            invocation["benchmark_name"],
            invocation["phase"],
            invocation["outer_iteration"],
            invocation["iteration"],
            metric["section"],
            metric["metric"],
            metric["unit"],
        )
        argv = invocation["argv"]
        if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError(f"{path}: argv must be a list of strings")
        try:
            # Launcher locations do not define the workload; retain the command,
            # binding, and every recorded option from the load/run token onward.
            arguments = tuple(argv[argv.index(invocation["phase"]) :])
        except ValueError as exc:
            raise ValueError(f"{path}: argv does not identify its YCSB phase") from exc
        problem = None
        if (
            invocation["exit_code"] != 0
            or invocation["execution_state"] != "completed"
            or invocation["parse_state"] != "ok"
        ):
            problem = "invalid_invocation"
        try:
            value = float(metric["value"])
            if not math.isfinite(value):
                value = None
        except (TypeError, ValueError, OverflowError):
            value = None
        if value is None and problem is None:
            problem = "invalid_value"
        if key in observations:
            problem = "duplicate_metric"
        observations[key] = _Observation(value, problem, arguments)
    return _Run(ids[0], observations)


def compare_ycsb_runs(
    left: str | Path | Sequence[str | Path],
    right: str | Path | Sequence[str | Path],
    *,
    phase: str = "run",
) -> pl.DataFrame:
    """Match saved metrics; multiple independent collections enable t-tests.

    Repeat positions are matching keys, never extra replications. Missing,
    ambiguous, invalid, or configuration-mismatched observations block inference
    for that metric. Valid descriptive values are still reported.
    """
    if phase not in {"load", "run", "all"}:
        raise ValueError("phase must be load, run, or all")
    groups = []
    seen = set()
    for side in (left, right):
        paths = [side] if isinstance(side, (str, Path)) else list(side)
        if not paths:
            raise ValueError("provide at least one collection on each side")
        runs = []
        for path in paths:
            run = _read_run(Path(path), phase)
            if run.collection_id in seen:
                raise ValueError(
                    f"collection {run.collection_id} occurs more than once; samples must be independent"
                )
            seen.add(run.collection_id)
            runs.append(run)
        groups.append(runs)
    left_runs, right_runs = groups
    keys = set().union(*(run.observations.keys() for group in groups for run in group))
    rows = []
    for key in keys:
        observations = [
            [run.observations.get(key) for run in group] for group in groups
        ]
        samples = [
            [
                entry.value
                for entry in entries
                if entry is not None
                and entry.problem is None
                and entry.value is not None
            ]
            for entries in observations
        ]
        result = welch_ttest(*samples)
        row = dict(zip(KEY_COLUMNS, key, strict=True))
        row.update(result)
        row.update(
            left_collections=json.dumps([run.collection_id for run in left_runs]),
            right_collections=json.dumps([run.collection_id for run in right_runs]),
            p_value_holm=None,
            notes="",
        )
        problems = []
        arguments = set()
        for label, entries in zip(("left", "right"), observations, strict=True):
            row[f"{label}_missing"] = sum(entry is None for entry in entries)
            row[f"{label}_invalid"] = sum(
                entry is not None and entry.problem is not None for entry in entries
            )
            for run, entry in zip(
                left_runs if label == "left" else right_runs, entries, strict=True
            ):
                if entry is None:
                    problems.append("missing_metric")
                    row["notes"] += f"{label} {run.collection_id}: missing metric; "
                elif entry.problem:
                    problems.append(entry.problem)
                    row["notes"] += f"{label} {run.collection_id}: {entry.problem}; "
                else:
                    arguments.add(entry.argv)
        if len(arguments) > 1:
            problems.append("configuration_mismatch")
            row["notes"] += "recorded YCSB workload arguments differ; "
        if problems:
            for status in (
                "duplicate_metric",
                "invalid_invocation",
                "invalid_value",
                "missing_metric",
                "configuration_mismatch",
            ):
                if status in problems:
                    row.update(status=status, t_statistic=None, p_value=None)
                    break
        row["notes"] = row["notes"].rstrip("; ")
        rows.append(row)

    # One correction family contains every valid per-metric test in this report.
    tested = sorted(
        (row for row in rows if row["status"] == "ok"), key=lambda row: row["p_value"]
    )
    adjusted = 0.0
    for index, row in enumerate(tested):
        adjusted = min(1.0, max(adjusted, (len(tested) - index) * row["p_value"]))
        row["p_value_holm"] = adjusted
    return pl.DataFrame(rows, schema=SCHEMA).sort(list(KEY_COLUMNS), nulls_last=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left",
        nargs="+",
        type=Path,
        required=True,
        help="Left collection directory/directories",
    )
    parser.add_argument(
        "--right",
        nargs="+",
        type=Path,
        required=True,
        help="Right collection directory/directories",
    )
    parser.add_argument("--phase", choices=("load", "run", "all"), default="run")
    parser.add_argument(
        "--output", type=Path, help="New CSV report; existing files are not overwritten"
    )
    args = parser.parse_args()
    try:
        report = compare_ycsb_runs(args.left, args.right, phase=args.phase)
        if args.output is None:
            sys.stdout.write(report.write_csv())
        else:
            if args.output.suffix.lower() != ".csv":
                raise ValueError("--output must name a CSV file")
            with args.output.open("x", encoding="utf-8", newline="") as output:
                report.write_csv(output)
        tested = report.filter(pl.col("status") == "ok").height
        print(
            f"Compared {report.height} metric rows; computed {tested} Welch t-tests.",
            file=sys.stderr,
        )
        if report.filter(pl.col("status") == "insufficient_samples").height:
            print(
                "A t-test needs at least two independent collections per side for the same metric; single runs only provide differences.",
                file=sys.stderr,
            )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
