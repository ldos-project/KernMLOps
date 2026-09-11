"""Offline original capture/MPMO versus historical NONE/ALL, without rewriting history.

Replicate numbers belong to their own cohorts, not matched pairs. Raw exports keep
load/run and repeat positions separate; plots and Welch reports default to run.
An optional third cohort adds unreduced, selected-only stock capture controls.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import polars as pl
import run_ycsb_matrix as runner
from summarize_ycsb_matrix import DIMENSIONS, FIELDS, family, finite, read_collection
from ycsb_ttest import compare_ycsb_runs

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = Path("data/experiments/redis-rss-dtlb-20260908")
THPS = runner.THPS
MODES = ("none", "all", *runner.ORIGINAL_MODES)
RAW_FIELDS = ("cohort", "cohort_id", *FIELDS)
OVERHEAD_FIELDS = (
    "cohort",
    "cohort_id",
    "thp",
    "mode",
    *DIMENSIONS,
    "n",
    "mean",
    "baseline_cohort_id",
    "baseline_n",
    "baseline_mean",
    "overhead_percent",
    "status",
)
NOTES = (
    "Historical NONE/ALL and new original collectors are different time cohorts; "
    "cohort/time confounding prevents attributing differences solely to instrumentation. "
    "Historical ALL excludes cbmm and lacks the DTLB READ ACCESS counter; both new "
    "original modes explicitly enable ACCESS. The 60 historical replacement-reduced "
    "jobs remain archived and are never pooled. Overhead is descriptive ratio of "
    "independent collection means versus historical NONE within THP and metric/repeat "
    "position, not paired replicates or median paired overhead. Positive overhead "
    "means throughput loss or runtime/latency increase. Welch tests use independent "
    "collections, with Holm correction within each report only, not across contrasts. "
    "Load and run phases are never pooled. Original producer/bridge/read failures "
    "are not observable; missing error counters are not evidence of zero errors."
)
CONTROL_NOTES = (
    "The selected-only control is a third time cohort; cohort/time confounding remains. "
    "Unreduced selected uses plain BCC CAPTURE with full RSS and perf records, retaining "
    "stock capture processing without LLVM reduction or a private bundle. Its exact "
    "selected probe/counter set matches the reduced original modes: mm_rss_stat and "
    "perf with DTLB READ MISS and READ ACCESS only, no ITLB, flush, walk, or cbmm. "
    "Userspace decoders and backends differ, so these comparisons do not isolate an "
    "LLVM reduction effect."
)
require = runner.require


def read_cohort(directory, cohort):
    """Read frozen cells and latest attempts only; never regenerate a historical plan."""
    manifest = json.loads((directory / "manifest.json").read_text())
    require(
        manifest["status"] in {"complete", "paused", "failed", "incomplete", "blocked"},
        f"{cohort} matrix must be stopped (status={manifest['status']})",
    )
    plan = json.loads((directory / "plan.json").read_text())
    require(
        plan["version"] == 1 and plan["repeats"] == 30,
        "expected version 1, thirty repeats per cell",
    )
    require(
        Path(plan["directory"]).resolve() == directory,
        "frozen matrix directory mismatch",
    )
    if cohort == "control":
        modes = selected = ("selected-control",)
        require(plan["artifact"] is None, "control must not use a reduction manifest")
        require(plan["seed"] == 17, "expected control seed 17")
    else:
        modes = runner.ORIGINAL_MODES if cohort == "candidate" else runner.MODES
        selected = runner.ORIGINAL_MODES if cohort == "candidate" else ("none", "all")
    require(
        set(plan["cells"]) == {f"{thp}-{mode}" for thp in THPS for mode in modes},
        "unexpected matrix cells",
    )
    require(
        len(plan["all_hooks"]) == len(set(plan["all_hooks"])) == 12
        and "cbmm" not in plan["all_hooks"],
        "expected twelve ALL hooks without cbmm",
    )
    cells = {
        f"{thp}-{mode}": plan["cells"][f"{thp}-{mode}"]
        for thp in THPS
        for mode in selected
    }
    for name, cell in cells.items():
        require(name == f"{cell['thp']}-{cell['mode']}", "cell identity mismatch")
        path = Path(cell["config_path"])
        require(
            path == directory / "configs" / f"{name}.yaml"
            and path.resolve().is_relative_to(directory / "configs"),
            "frozen config outside matrix configs",
        )
        content = path.read_bytes()
        require(
            runner.digest(content) == cell["config_sha256"],
            "frozen config hash mismatch",
        )
        require(json.loads(content) == cell["config"], "saved config differs from plan")
        runner.check_hooks(
            cell["config"],
            cell["mode"],
            plan["all_hooks"],
            None if cohort == "control" else plan["artifact"]["path"],
        )
        generic = cell["config"]["collector_config"]["generic"]
        expected_hooks = runner.ORIGINAL_HOOKS.get(cell["mode"])
        require(
            cell["expected_hooks"]
            == ([expected_hooks] if expected_hooks else generic["hooks"]),
            "frozen expected hooks mismatch",
        )
        if cohort == "control":
            require(
                cell["factory_metadata"]
                == {name: generic[name] for name in ("hooks", "perf_events")},
                "frozen control factory metadata mismatch",
            )
        benchmark = cell["config"]["benchmark_config"]["generic"]
        require(
            benchmark["benchmark"] == "redis"
            and benchmark["transparent_hugepages"] == cell["thp"],
            "wrong Redis/THP scenario",
        )
    if cohort == "candidate":
        artifact = Path(plan["artifact"]["path"]).read_bytes()
        require(
            runner.digest(artifact) == plan["artifact"]["sha256"],
            "candidate manifest hash mismatch",
        )
        bundle = json.loads(artifact)
        require(
            bundle.get("version") == 1 and bundle.get("kind") == "original-bcc-llvm",
            "not an original bundle",
        )
        require(
            all(
                cell["factory_metadata"]["manifest"] == bundle
                for cell in cells.values()
            ),
            "frozen factory manifest mismatch",
        )
    jobs = {job["key"]: job for job in plan["jobs"]}
    require(
        len(jobs) == len(plan["jobs"])
        and Counter((job["cell"], job["replicate"]) for job in jobs.values())
        == Counter((cell, r) for cell in plan["cells"] for r in range(30)),
        "invalid frozen jobs/replicates",
    )
    require(
        all(
            job["thp"] == plan["cells"][job["cell"]]["thp"]
            and job["mode"] == plan["cells"][job["cell"]]["mode"]
            for job in jobs.values()
        ),
        "job/cell mismatch",
    )
    latest = {}
    progress = directory / "results.jsonl"
    for line in progress.read_text().splitlines() if progress.exists() else []:
        event = json.loads(line)
        require(
            event["key"] in jobs
            and all(event[k] == v for k, v in jobs[event["key"]].items()),
            "result/plan mismatch",
        )
        if event["cell"] not in cells:
            continue  # Archived replacement probes are not comparison inputs.
        require(
            type(event["attempt"]) is int and event["attempt"] > 0, "invalid attempt"
        )
        require(
            event["attempt"] >= latest.get(event["key"], {}).get("attempt", 0),
            "attempt order regressed",
        )
        prefix = f"{plan['experiment']}-{event['key']}-a{event['attempt']:02d}"
        require(
            event["prefix"] == prefix
            and event["config_path"] == cells[event["cell"]]["config_path"]
            and all(
                event[source] == str(directory / "runs" / f"{prefix}.{source}.log")
                for source in ("stdout", "stderr")
            ),
            "saved attempt paths/label mismatch",
        )
        latest[event["key"]] = event
    coverage = {
        "cohort_id": plan["experiment"],
        "directory": str(directory),
        "manifest_status": manifest["status"],
        "expected": len(cells) * plan["repeats"],
        "completed": 0,
        "missing_jobs": [],
        "invalid_jobs": {},
        "excluded_modes": ["reduced"] if cohort == "historical" else [],
    }
    raw, valid = [], []
    for key, job in jobs.items():
        if job["cell"] not in cells:
            continue
        event = latest.get(key)
        if event is None or event["status"] != "success":
            coverage["missing_jobs"].append(key)
            continue
        try:
            cell = cells[event["cell"]]
            rows = read_collection(event, cell)
            verified = deepcopy(event)
            runner.validate_result(verified, cell)
            require(
                verified["collection_dir"] == event["collection_dir"]
                and verified["tables"] == event["tables"],
                "saved table paths differ from validated files",
            )
            if cohort == "candidate":
                require(
                    verified["compiler_evidence_dir"] == event["compiler_evidence_dir"]
                    and verified["telemetry_status"] == event["telemetry_status"],
                    "saved compiler evidence/status mismatch",
                )
            info = runner.read_rows(event["tables"]["system_info"])[0]
            require(
                info["transparent_hugepages"] == cell["thp"], "SystemInfo THP mismatch"
            )
            system = {
                name: info[name]
                for name in (
                    "hostname",
                    "os",
                    "kernel_version",
                    "architecture",
                    "sockets",
                    "cores",
                    "logical_cores",
                    "ram_bytes",
                )
            }
            commands = {}
            for invocation in runner.read_rows(event["tables"]["ycsb_invocations"]):
                argv = invocation["argv"]
                require(
                    isinstance(argv, list)
                    and all(isinstance(arg, str) for arg in argv),
                    "invalid YCSB argv",
                )
                position = tuple(invocation[name] for name in DIMENSIONS[:3])
                commands[position] = tuple(argv[argv.index(invocation["phase"]) :])
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            pl.exceptions.PolarsError,
        ) as exc:
            coverage["invalid_jobs"][key] = str(exc)
            continue
        raw.extend(
            {"cohort": cohort, "cohort_id": plan["experiment"], **row} for row in rows
        )
        valid.append((event, system, commands))
    coverage["completed"] = len(valid)
    return plan, cells, raw, valid, coverage


def mean_overheads(raw):
    """Ratio of means by metric/repeat position, deliberately without replicate keys."""
    groups = defaultdict(list)
    for row in raw:
        if family(row):
            groups[(row["thp"], row["mode"], *(row[k] for k in DIMENSIONS))].append(row)
    result = []
    for (thp, mode, *dimensions), rows in sorted(
        groups.items(), key=lambda item: str(item[0])
    ):
        if mode == "none":
            continue
        base = groups.get((thp, "none", *dimensions), [])
        values = [r["value"] for r in rows if finite(r["value"])]
        baseline = [r["value"] for r in base if finite(r["value"])]
        mean = statistics.fmean(values) if values else None
        baseline_mean = statistics.fmean(baseline) if baseline else None
        overhead = None
        if finite(mean) and finite(baseline_mean) and baseline_mean > 0:
            ratio = mean / baseline_mean
            change = 100 * (
                1 - ratio if dimensions[4] == "Throughput(ops/sec)" else ratio - 1
            )
            overhead = change if finite(change) else None
        result.append(
            {
                "cohort": rows[0]["cohort"],
                "cohort_id": rows[0]["cohort_id"],
                "thp": thp,
                "mode": mode,
                **dict(zip(DIMENSIONS, dimensions)),
                "n": len(values),
                "mean": mean if finite(mean) else None,
                "baseline_cohort_id": base[0]["cohort_id"] if base else None,
                "baseline_n": len(baseline),
                "baseline_mean": baseline_mean if finite(baseline_mean) else None,
                "overhead_percent": overhead,
                "status": "descriptive" if overhead is not None else "unavailable",
            }
        )
    return result


def plot_metrics(raw, overheads, coverage, category, path, phase):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    control = "control" in coverage["cohorts"]
    modes = MODES + (("selected-control",) if control else ())
    keys = sorted(
        {
            tuple(row[k] for k in DIMENSIONS)
            for row in raw
            if family(row) == category and (phase == "all" or row["phase"] == phase)
        },
        key=str,
    )
    fig, axes = plt.subplots(
        max(1, len(keys)),
        2,
        squeeze=False,
        figsize=(16, max(4, 3.2 * len(keys) + 1.5)),
        constrained_layout=True,
    )
    fig.suptitle(
        f"Redis/YCSB {category}: {'PARTIAL' if coverage['partial'] else 'COMPLETE'} {coverage['completed']}/{coverage['expected']} collections; phase={phase}\n"
        "Historical NONE/ALL vs new original capture/MPMO: time-cohort confounding; historical ALL lacks ACCESS\n"
        + (
            "Unreduced selected control: stock full-record RSS + DTLB MISS/ACCESS only; no ITLB/flush/walk or CBMM\n"
            if control
            else ""
        )
        + "Raw boxes/collection points; descriptive ratio of means vs historical NONE, never replicate-paired",
        fontsize=10,
    )
    colors = ("#666666", "#c66125", "#2675a6", "#548b41", "#8b4f9e")
    for index, key in enumerate(keys):
        for column in range(2):
            ax = axes[index, column]
            positions, labels = [], []
            for thp_index, thp in enumerate(THPS):
                for mode_index, mode in enumerate(modes):
                    if column and mode == "none":
                        continue
                    position = thp_index * (len(modes) + 1) + mode_index
                    samples = [
                        r
                        for r in raw
                        if r["thp"] == thp
                        and r["mode"] == mode
                        and tuple(r[k] for k in DIMENSIONS) == key
                        and finite(r["value"])
                    ]
                    positions.append(position)
                    label = {
                        "none": "hist NONE",
                        "all": "hist ALL",
                        "original-reduced": "capture",
                        "original-mpmo": "MPMO",
                        "selected-control": "unreduced\nselected",
                    }[mode]
                    labels.append(f"{thp}\n{label}\nn={len(samples)}/30")
                    if not column and samples:
                        values = [r["value"] for r in samples]
                        ax.boxplot(
                            [values],
                            positions=[position],
                            widths=0.55,
                            showfliers=False,
                            patch_artist=True,
                            boxprops={"facecolor": colors[mode_index], "alpha": 0.25},
                        )
                        ax.scatter(
                            [
                                position + 0.3 * (i / max(1, len(values) - 1) - 0.5)
                                for i in range(len(values))
                            ],
                            values,
                            s=12,
                            color=colors[mode_index],
                        )
                    if column:
                        changes = [
                            r["overhead_percent"]
                            for r in overheads
                            if r["thp"] == thp
                            and r["mode"] == mode
                            and tuple(r[k] for k in DIMENSIONS) == key
                            and finite(r["overhead_percent"])
                        ]
                        if changes:
                            ax.bar(
                                position,
                                changes[0],
                                width=0.55,
                                color=colors[mode_index],
                                alpha=0.7,
                            )
            ax.set_xticks(positions, labels, fontsize=8)
            ax.set_title(
                f"{key[0]} {key[3]} {key[4]} | outer={key[1]}, inner={key[2]} | {'ratio of means' if column else 'raw'}",
                fontsize=9,
            )
            ax.set_ylabel(
                (
                    "Throughput loss (%)"
                    if key[4] == "Throughput(ops/sec)"
                    else "Increase (%)"
                )
                if column
                else key[5] or "value"
            )
            ax.grid(axis="y", alpha=0.2)
            if column:
                ax.axhline(0, color="black", linewidth=0.8)
    if not keys:
        for ax in axes[0]:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No valid metrics", ha="center", transform=ax.transAxes)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def compare_original_matrix(
    directory,
    historical=HISTORICAL,
    *,
    control_directory=None,
    output_dir=None,
    phase="run",
    require_complete=False,
):
    directory, historical = (ROOT / directory).resolve(), (ROOT / historical).resolve()
    require(
        not directory.is_relative_to(historical)
        and not historical.is_relative_to(directory),
        "candidate and historical matrices must be distinct, nonnested directories",
    )
    cohorts = [("historical", historical), ("candidate", directory)]
    if control_directory is not None:
        control_directory = (ROOT / control_directory).resolve()
        require(
            all(
                not control_directory.is_relative_to(path)
                and not path.is_relative_to(control_directory)
                for _, path in cohorts
            ),
            "control, candidate and historical matrices must be distinct, nonnested directories",
        )
        cohorts.append(("control", control_directory))
    require(phase in {"load", "run", "all"}, "invalid phase")
    output_cohort, output_root = cohorts[-1]
    analysis = (output_root / (output_dir or "analysis")).resolve()
    require(
        analysis != output_root and analysis.is_relative_to(output_root),
        f"output must be inside the {output_cohort} matrix",
    )
    require(
        not any(
            analysis.is_relative_to(output_root / name) for name in ("configs", "runs")
        ),
        "output cannot overwrite matrix inputs",
    )
    with ExitStack() as stack:
        for path in sorted(path for _, path in cohorts):
            lock = stack.enter_context((path / ".lock").open("rb"))
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    "matrix runner is active; compare only offline"
                ) from exc
        coverage = {
            "expected": 0,
            "repeats": 30,
            "cells": {},
            "cohorts": {},
            "notes": NOTES
            + (" " + CONTROL_NOTES if control_directory is not None else ""),
            "phase": phase,
        }
        raw, collections, seen = [], {}, set()
        reference_workload = reference_system = reference_commands = None
        cohort_ids = set()
        for cohort, path in cohorts:
            plan, cells, rows, valid, details = read_cohort(path, cohort)
            require(plan["experiment"] not in cohort_ids, "cohort IDs must be distinct")
            cohort_ids.add(plan["experiment"])
            for name, cell in cells.items():
                # Failed/missing attempts are still input storage, not output space.
                require(
                    not analysis.is_relative_to(
                        (
                            Path(cell["output_dir"])
                            / ("baseline" if cell["mode"] == "none" else "curated")
                            / "redis"
                        ).resolve()
                    ),
                    "output cannot overwrite collection inputs",
                )
                generic = cell["config"]["benchmark_config"]["generic"].copy()
                generic.pop("transparent_hugepages")
                workload = (generic, cell["config"]["benchmark_config"]["redis"])
                if reference_workload is None:
                    reference_workload = workload
                require(
                    workload == reference_workload,
                    "frozen Redis workload/VM configuration differs",
                )
                coverage["cells"][name] = {
                    "cohort_id": plan["experiment"],
                    "expected": 30,
                    "completed": 0,
                }
                collections[name] = []
            for event, system, commands in valid:
                require(
                    event["collection_id"] not in seen,
                    "collection reused across jobs/cohorts",
                )
                seen.add(event["collection_id"])
                if reference_system is None:
                    reference_system, reference_commands = system, commands
                require(
                    system == reference_system,
                    "SystemInfo machine configuration differs across collections",
                )
                require(
                    commands == reference_commands,
                    "recorded YCSB commands differ across collections",
                )
                require(
                    not analysis.is_relative_to(
                        Path(event["collection_dir"]).resolve()
                    ),
                    "output cannot overwrite collection inputs",
                )
                collections[event["cell"]].append(event["collection_dir"])
                coverage["cells"][event["cell"]]["completed"] += 1
            coverage["cohorts"][cohort] = details
            coverage["expected"] += details["expected"]
            raw.extend(rows)
        coverage.update(
            completed=len(seen),
            partial=len(seen) != coverage["expected"],
            raw_rows=len(raw),
        )
        require(
            not require_complete or not coverage["partial"],
            f"comparison is partial; require {coverage['expected']} valid collections",
        )
        overheads = mean_overheads(raw)
        contrasts = []
        for thp in THPS:
            contrasts.append((f"{thp}-original-reduced", f"{thp}-original-mpmo"))
            contrasts.extend(
                (f"{thp}-{control}", f"{thp}-{mode}")
                for mode in runner.ORIGINAL_MODES
                for control in ("none", "all")
            )
        if control_directory is not None:
            contrasts.extend(
                (f"{thp}-selected-control", f"{thp}-{mode}")
                for thp in THPS
                for mode in (*runner.ORIGINAL_MODES, "all", "none")
            )
        names = [
            "raw_metrics.csv",
            "overheads.csv",
            "coverage.json",
            "overall_comparison.png",
            "latency_comparison.png",
        ]
        reports = {}
        coverage["contrasts"] = []
        for left, right in contrasts:
            name = f"ttest-{left}-vs-{right}.csv"
            available = bool(collections[left] and collections[right])
            coverage["contrasts"].append(
                {
                    "left": left,
                    "right": right,
                    "status": "reported" if available else "missing_collections",
                    "path": str(analysis / name) if available else None,
                }
            )
            if available:
                reports[name] = compare_ycsb_runs(
                    collections[left], collections[right], phase=phase
                )
            names.append(name)
        # Check every potential output before writing anything, including stale reports.
        for name in names:
            path = analysis / name
            require(
                not path.is_symlink()
                and (not path.exists() or path.is_file() and path.stat().st_nlink == 1),
                f"unsafe analysis output: {path}",
            )
        analysis.mkdir(parents=True, exist_ok=True)
        coverage["artifacts"] = [
            str(analysis / name) for name in names[:5] + list(reports)
        ]
        # Replace completed files rather than following an output link while writing.
        with tempfile.TemporaryDirectory(
            prefix=".original-", dir=analysis
        ) as temporary:
            staged = Path(temporary)
            for name, rows, fields in (
                ("raw_metrics.csv", raw, RAW_FIELDS),
                ("overheads.csv", overheads, OVERHEAD_FIELDS),
            ):
                with (staged / name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            for category in ("overall", "latency"):
                plot_metrics(
                    raw,
                    overheads,
                    coverage,
                    category,
                    staged / f"{category}_comparison.png",
                    phase,
                )
            for name, report in reports.items():
                report.write_csv(staged / name)
            (staged / "coverage.json").write_text(
                json.dumps(coverage, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            for name in names:
                if (staged / name).exists():
                    os.replace(staged / name, analysis / name)
                else:
                    (analysis / name).unlink(missing_ok=True)
        return coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="New original-only candidate matrix",
    )
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument(
        "--control-directory",
        type=Path,
        default=argparse.SUPPRESS,
        help="Optional selected-control matrix; writes analysis only inside this experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Analysis subdirectory inside the control, if supplied, otherwise the candidate (default: analysis)",
    )
    parser.add_argument("--phase", choices=("load", "run", "all"), default="run")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        coverage = compare_original_matrix(**vars(args))
    except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as exc:
        parser.error(str(exc))
    print(
        f"{'PARTIAL' if coverage['partial'] else 'COMPLETE'}: {coverage['completed']}/{coverage['expected']} valid collections"
    )
    print(NOTES)
    if hasattr(args, "control_directory"):
        print(CONTROL_NOTES)
    print("\n".join(coverage["artifacts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
