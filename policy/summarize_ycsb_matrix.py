"""Offline Redis/YCSB matrix summaries; only analysis/ is written.

summarize_matrix(directory) returns coverage, including the five artifact paths.
Raw exports retain all metric rows and phases; plots are descriptive, run-only.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
THPS, MODES = ("always", "never"), ("none", "all", "reduced")
DIMENSIONS = ("phase", "outer_iteration", "iteration", "section", "metric", "unit")
PAIR = ("thp", "replicate", *DIMENSIONS)
FIELDS = (
    "thp",
    "mode",
    "replicate",
    "collection_id",
    "invocation_id",
    *DIMENSIONS,
    "value",
)
EXTRA = ("baseline_collection_id", "baseline_value", "overhead_percent")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def family(row):
    if row["section"] == "OVERALL":
        return (
            "overall"
            if row["metric"] in ("Throughput(ops/sec)", "RunTime(ms)")
            else None
        )
    if row["metric"].split("(")[0] in (
        "AverageLatency",
        "95thPercentileLatency",
        "99thPercentileLatency",
    ):
        return "latency"
    return None


def read_collection(event, cell):
    cid, tables = event["collection_id"], event["tables"]
    directory = Path(event["collection_dir"]).resolve()
    require(
        event["return_code"] == 0 and directory.is_dir() and directory.name == cid,
        "missing/mismatched collection directory or failed collector",
    )
    required = {"system_info", "ycsb_invocations", "ycsb_metrics"}
    if cell["mode"] == "reduced":
        required |= {"policy_rss", "policy_dtlb"}
    require(
        required <= tables.keys() and all(tables[name] for name in required),
        "missing required tables",
    )
    for paths in tables.values():
        for path in paths if isinstance(paths, list) else [paths]:
            path = Path(path)
            require(
                path.is_file() and path.resolve().is_relative_to(directory),
                f"missing/outside artifact: {path}",
            )
    system = pl.read_parquet(tables["system_info"]).to_dicts()
    require(
        len(system) == 1
        and system[0]["collection_id"] == cid
        and system[0]["benchmark_name"] == "redis"
        and system[0]["return_code"] == 0
        and Counter(system[0]["hooks"]) == Counter(cell["expected_hooks"]),
        "invalid SystemInfo",
    )
    invocations = pl.read_parquet(tables["ycsb_invocations"]).to_dicts()
    metadata = {(i["collection_id"], i["invocation_id"]): i for i in invocations}
    redis = cell["config"]["benchmark_config"]["redis"]
    expected = {("load", outer, 0) for outer in range(redis["outer_repeat"])}
    expected |= {
        ("run", outer, inner)
        for outer in range(redis["outer_repeat"])
        for inner in range(redis["repeat"])
    }
    require(
        len(metadata) == len(invocations) == len(expected)
        and {(i["phase"], i["outer_iteration"], i["iteration"]) for i in invocations}
        == expected,
        "duplicate invocation or missing repeat position",
    )
    require(
        all(
            i["collection_id"] == cid
            and i["benchmark_name"] == "redis"
            and i["execution_state"] == "completed"
            and i["exit_code"] == 0
            and i["parse_state"] == "ok"
            for i in invocations
        ),
        "invalid YCSB invocation",
    )
    rows, seen = [], set()
    for metric in pl.read_parquet(tables["ycsb_metrics"]).to_dicts():
        identifier = (metric["collection_id"], metric["invocation_id"])
        require(
            identifier in metadata, "metric references unknown collection/invocation"
        )
        require(
            all(
                isinstance(metric[k], str) and metric[k] for k in ("section", "metric")
            ),
            "invalid metric identity",
        )
        combined = {**event, **metadata[identifier], **metric}
        row = {name: combined[name] for name in FIELDS}
        key = tuple(row[name] for name in PAIR)
        require(key not in seen, "duplicate metric at repeat position")
        seen.add(key)
        rows.append(row)
    return rows


def plot_metrics(raw, overheads, coverage, category, path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    keys = sorted(
        {
            tuple(r[k] for k in DIMENSIONS)
            for r in raw
            if r["phase"] == "run" and family(r) == category
        },
        key=str,
    )
    fig, axes = plt.subplots(
        max(1, len(keys)),
        2,
        squeeze=False,
        figsize=(15, max(4, 3.1 * len(keys) + 1.4)),
        constrained_layout=True,
    )
    counts = "; ".join(
        f"{name}: {c['completed']}/{c['expected']}"
        for name, c in coverage["cells"].items()
    )
    status = "PARTIAL" if coverage["partial"] else "COMPLETE"
    fig.suptitle(
        f"Redis/YCSB {category}, run phase: {status}, {coverage['completed']}/{coverage['expected']} collections\n"
        f"{counts}\nAll excludes cbmm; reduced = extracted RSS components and DTLB READ MISS / READ ACCESS\n"
        "Descriptive boxes and collection points; n counts valid collections or matched pairs",
        fontsize=10,
    )
    for index, key in enumerate(keys):
        for column, source in enumerate((raw, overheads)):
            ax, modes = axes[index, column], MODES[1:] if column else MODES
            field, labels, positions = "overhead_percent" if column else "value", [], []
            for thp_index, thp in enumerate(THPS):
                for mode_index, mode in enumerate(modes):
                    position = thp_index * (len(modes) + 1) + mode_index
                    samples = [
                        r
                        for r in source
                        if r["thp"] == thp
                        and r["mode"] == mode
                        and tuple(r[k] for k in DIMENSIONS) == key
                        and finite(r[field])
                    ]
                    positions.append(position)
                    n = len({r["collection_id"] for r in samples})
                    labels.append(
                        f"{thp}\n{mode}\nn={n}/{coverage['cells'][f'{thp}-{mode}']['expected']}"
                    )
                    if samples:
                        color = {
                            "none": "#666666",
                            "all": "#c66125",
                            "reduced": "#2675a6",
                        }[mode]
                        values = [r[field] for r in samples]
                        ax.boxplot(
                            [values],
                            positions=[position],
                            widths=0.55,
                            showfliers=False,
                            patch_artist=True,
                            boxprops={"facecolor": color, "alpha": 0.25},
                        )
                        xs = [
                            position
                            + 0.3
                            * (r["replicate"] / max(1, coverage["repeats"] - 1) - 0.5)
                            for r in samples
                        ]
                        ax.scatter(xs, values, color=color, s=14, alpha=0.75)
            if column:
                ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(positions, labels)
            ax.set_xlim(-0.7, positions[-1] + 0.7)
            ax.set_title(
                f"{key[3]} {key[4]} | outer={key[1]}, inner={key[2]} | "
                + ("paired vs none" if column else "raw"),
                fontsize=10,
            )
            ax.set_ylabel(
                (
                    "Throughput loss (%)"
                    if key[4] == "Throughput(ops/sec)"
                    else "Increase (%)"
                )
                if column
                else (key[5] or "value")
            )
            ax.grid(axis="y", alpha=0.2)
    if not keys:
        for ax in axes[0]:
            ax.set_axis_off()
            ax.text(
                0.5, 0.5, "No valid run metrics", ha="center", transform=ax.transAxes
            )
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize_matrix(directory, *, ttests=False):
    directory = (ROOT / directory).resolve()
    # A read-only shared lock also excludes a runner resuming a terminal manifest.
    with (directory / ".lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("matrix runner is active; summarize only offline") from exc
        manifest = json.loads((directory / "manifest.json").read_text())
        require(
            manifest["status"]
            in {"complete", "paused", "failed", "incomplete", "blocked"},
            f"matrix must be stopped before analysis (status={manifest['status']})",
        )
        plan = json.loads((directory / "plan.json").read_text())
        cells, repeats = plan["cells"], plan["repeats"]
        require(type(repeats) is int and repeats > 0, "invalid planned repeats")
        require(
            set(cells) == {f"{thp}-{mode}" for thp in THPS for mode in MODES},
            "expected six THP/probe cells",
        )
        require("cbmm" not in plan["all_hooks"], "cbmm is forbidden")
        for name, cell in cells.items():
            thp, mode = cell["thp"], cell["mode"]
            generic = cell["config"]["collector_config"]["generic"]
            benchmark = cell["config"]["benchmark_config"]["generic"]
            require(
                "cbmm" not in generic["hooks"] and "cbmm" not in cell["expected_hooks"],
                "cbmm is forbidden",
            )
            require(
                name == f"{thp}-{mode}"
                and benchmark["benchmark"] == "redis"
                and benchmark["transparent_hugepages"] == thp,
                "wrong Redis/THP scenario",
            )
            hooks = plan["all_hooks"] if mode == "all" else []
            require(
                generic["hooks"] == hooks
                and cell["expected_hooks"]
                == (["extracted_rss_dtlb"] if mode == "reduced" else hooks)
                and bool(generic.get("extracted_rss_tlb")) == (mode == "reduced"),
                "probe configuration mismatch",
            )
        jobs = {j["key"]: j for j in plan["jobs"]}
        require(
            len(jobs) == len(plan["jobs"])
            and Counter((j["cell"], j["replicate"]) for j in jobs.values())
            == Counter((cell, r) for cell in cells for r in range(repeats)),
            "invalid planned jobs/rounds",
        )
        require(
            all(
                j["thp"] == cells[j["cell"]]["thp"]
                and j["mode"] == cells[j["cell"]]["mode"]
                for j in jobs.values()
            ),
            "job/cell mismatch",
        )
        latest, progress = {}, directory / "results.jsonl"
        for line in progress.read_text().splitlines() if progress.exists() else []:
            event = json.loads(line)
            require(
                event["key"] in jobs
                and all(event[k] == v for k, v in jobs[event["key"]].items()),
                "result/plan mismatch",
            )
            latest[event["key"]] = event
        latest = {
            key: event for key, event in latest.items() if event["status"] == "success"
        }
        coverage = {
            "manifest_status": manifest["status"],
            "repeats": repeats,
            "expected": len(jobs),
            "cells": {name: {"expected": repeats, "completed": 0} for name in cells},
            "missing_jobs": [key for key in jobs if key not in latest],
            "invalid_jobs": {},
        }
        raw, seen = [], set()
        collections = {name: [] for name in cells}
        for key, event in latest.items():
            try:
                require(
                    event["collection_id"] not in seen,
                    "collection reused by multiple jobs",
                )
                rows = read_collection(event, cells[event["cell"]])
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                pl.exceptions.PolarsError,
            ) as exc:
                coverage["invalid_jobs"][key] = str(exc)
                continue
            raw.extend(rows)
            seen.add(event["collection_id"])
            coverage["cells"][event["cell"]]["completed"] += 1
            collections[event["cell"]].append(event["collection_dir"])
        baselines = {tuple(r[k] for k in PAIR): r for r in raw if r["mode"] == "none"}
        candidates = [r for r in raw if r["mode"] != "none" and family(r)]
        overheads = []
        for row in candidates:
            base = baselines.get(tuple(row[k] for k in PAIR))
            if (
                base is None
                or not finite(base["value"])
                or base["value"] <= 0
                or not finite(row["value"])
            ):
                continue
            ratio = row["value"] / base["value"]
            change = 100 * (
                1 - ratio if row["metric"] == "Throughput(ops/sec)" else ratio - 1
            )
            if finite(change):
                overheads.append(
                    {
                        **row,
                        "baseline_collection_id": base["collection_id"],
                        "baseline_value": base["value"],
                        "overhead_percent": change,
                    }
                )
        coverage.update(
            completed=len(seen),
            partial=len(seen) != len(jobs),
            raw_rows=len(raw),
            overhead_rows=len(overheads),
            excluded_overhead_rows=len(candidates) - len(overheads),
            notes="Descriptive only; plots use run phase. Overheads match THP, round, phase, outer/inner position and metric/unit. No CIs or thresholds.",
        )
        analysis = directory / "analysis"
        analysis.mkdir(exist_ok=True)
        names = (
            "raw_metrics.csv",
            "overheads.csv",
            "coverage.json",
            "overall_overhead.png",
            "latency_overhead.png",
        )
        coverage["artifacts"] = [str(analysis / name) for name in names]
        for name, rows, fields in (
            (names[0], raw, FIELDS),
            (names[1], overheads, FIELDS + EXTRA),
        ):
            with (analysis / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        for category in ("overall", "latency"):
            plot_metrics(
                raw,
                overheads,
                coverage,
                category,
                analysis / f"{category}_overhead.png",
            )
        if ttests:
            from ycsb_ttest import compare_ycsb_runs

            contrasts = [(f"always-{mode}", f"never-{mode}") for mode in MODES]
            contrasts += [
                (f"{thp}-{left}", f"{thp}-{right}")
                for thp in THPS
                for left, right in combinations(MODES, 2)
            ]
            for left, right in contrasts:
                if not collections[left] or not collections[right]:
                    continue
                report = analysis / f"ttest-{left}-vs-{right}.csv"
                compare_ycsb_runs(collections[left], collections[right]).write_csv(
                    report
                )
                coverage["artifacts"].append(str(report))
        (analysis / "coverage.json").write_text(
            json.dumps(coverage, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Repository-relative or absolute experiment directory",
    )
    parser.add_argument(
        "--ttests",
        action="store_true",
        help="Also write nine offline Welch comparison reports",
    )
    args = parser.parse_args(argv)
    try:
        coverage = summarize_matrix(args.directory, ttests=args.ttests)
    except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as exc:
        parser.error(str(exc))
    print(
        f"{'PARTIAL' if coverage['partial'] else 'COMPLETE'}: {coverage['completed']}/{coverage['expected']} valid completed collections"
    )
    print("\n".join(coverage["artifacts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
