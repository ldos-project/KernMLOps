"""Audit archived full-probe captures; write only a separate derived-data directory."""

import argparse
import fcntl
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RESIDENT = ("MM_ANONPAGES", "MM_FILEPAGES", "MM_SHMEMPAGES")


def saved_path(value):
    """Resolve the archive's container mount without rewriting the archive."""
    path = Path(value)
    if path.is_relative_to("/KernMLOps"):
        path = ROOT / path.relative_to("/KernMLOps")
    elif not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"input outside repository: {value}")
    return path


def table_files(directory, name):
    return sorted(directory.glob(f"{name}.*.parquet"))


def read_table(directory, name):
    paths = table_files(directory, name)
    if not paths:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed")


def full_runs(experiment):
    manifest = json.loads((experiment / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("audit requires a completed, stopped source experiment")
    plan = json.loads((experiment / "plan.json").read_text())
    if plan["repeats"] != 30 or "cbmm" in plan["all_hooks"]:
        raise ValueError("expected thirty full-probe rounds without cbmm")
    jobs = {job["key"]: job for job in plan["jobs"]}
    latest = {}
    for line in (experiment / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["key"] not in jobs or any(
            row[k] != v for k, v in jobs[row["key"]].items()
        ):
            raise ValueError("result differs from frozen plan")
        if row["mode"] == "all":
            latest[row["key"]] = row
    if len(latest) != 60 or any(row["status"] != "success" for row in latest.values()):
        raise ValueError("expected sixty successful latest full-probe collections")
    if Counter(row["thp"] for row in latest.values()) != {"always": 30, "never": 30}:
        raise ValueError("unbalanced full-probe cohort")
    for cell in plan["cells"].values():
        if cell["mode"] != "all":
            continue
        content = saved_path(cell["config_path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != cell["config_sha256"]:
            raise ValueError("frozen source configuration changed")
    return plan, sorted(latest.values(), key=lambda row: (row["replicate"], row["thp"]))


def audit_run(event, expected_hooks):
    cid = event["collection_id"]
    directory = saved_path(event["collection_dir"])
    stdout = saved_path(event["stdout"]).read_text(errors="replace")
    pids = set(re.findall(r"\bpid=(\d+), just started", stdout))
    if len(pids) != 1:
        raise ValueError("expected one explicitly logged Redis server PID")
    pid = int(pids.pop())
    info = read_table(directory, "system_info").to_dicts()
    if (
        len(info) != 1
        or info[0]["collection_id"] != cid
        or info[0]["return_code"] != 0
        or info[0]["transparent_hugepages"] != event["thp"]
        or Counter(info[0]["hooks"]) != Counter(expected_hooks)
    ):
        raise ValueError("invalid source SystemInfo")
    metadata = read_table(directory, "process_metadata").filter(pl.col("pid") == pid)
    if not metadata.height or not any(
        row["name"] == "redis-server"
        and isinstance(row["cmdline"], str)
        and "redis-server" in row["cmdline"]
        for row in metadata.to_dicts()
    ):
        raise ValueError("Redis log PID lacks matching process metadata")
    trace = read_table(directory, "process_trace").filter(
        (pl.col("pid") == pid) & (pl.col("tgid") == pid)
    )
    starts = trace.filter(pl.col("cap_type") == "start")["ts_ns"].to_list()
    ends = trace.filter(pl.col("cap_type") == "end")["ts_ns"].to_list()
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise ValueError("missing or ambiguous Redis leader lifetime")
    rss = read_table(directory, "mm_rss_stat").filter(pl.col("tgid") == pid)
    if not rss.height or rss["collection_id"].unique().to_list() != [cid]:
        raise ValueError("missing or mismatched Redis RSS records")
    components = rss["member"].unique().sort().to_list()
    common = {
        "collection_id": cid,
        "thp": event["thp"],
        "replicate": event["replicate"],
        "redis_pid": pid,
    }
    member_rows = [
        common | row
        for row in rss.group_by("member")
        .agg(
            pl.len().alias("rows"),
            pl.col("count").min().alias("min_pages"),
            pl.col("count").max().alias("max_pages"),
            pl.col("count").n_unique().alias("distinct_levels"),
            pl.col("ts_ns").min().alias("first_ns"),
            pl.col("ts_ns").max().alias("last_ns"),
        )
        .sort("member")
        .to_dicts()
    ]
    phase_rows = []
    metrics = read_table(directory, "ycsb_metrics")
    for invocation in read_table(directory, "ycsb_invocations").to_dicts():
        if (
            invocation["collection_id"] != cid
            or invocation["execution_state"] != "completed"
            or invocation["exit_code"] != 0
            or invocation["parse_state"] != "ok"
        ):
            raise ValueError("invalid YCSB invocation")
        start, end = invocation["start_monotonic_ns"], invocation["end_monotonic_ns"]
        if not starts[0] <= start < end <= ends[0]:
            raise ValueError("YCSB invocation outside the Redis process lifetime")
        phase_rss = rss.filter(pl.col("ts_ns").is_between(start, end))
        runtime = metrics.filter(
            (pl.col("invocation_id") == invocation["invocation_id"])
            & (pl.col("section") == "OVERALL")
            & (pl.col("metric") == "RunTime(ms)")
        )["value"].to_list()
        progress_path = Path(invocation["stderr_path"])
        progress_path = saved_path(
            progress_path if progress_path.is_absolute() else directory / progress_path
        )
        if not progress_path.is_relative_to(directory):
            raise ValueError("YCSB log outside its collection directory")
        progress = progress_path.read_text(errors="replace")
        phase_rows.append(
            common
            | {
                "phase": invocation["phase"],
                "outer_iteration": invocation["outer_iteration"],
                "iteration": invocation["iteration"],
                "start_ns": start,
                "end_ns": end,
                "launcher_duration_s": (end - start) / 1e9,
                "diagnostic_reported_runtime_s": runtime[0] / 1000
                if len(runtime) == 1
                else None,
                "rss_rows": phase_rss.height,
                "status_progress_lines": len(
                    re.findall(r"\d+ sec: \d+ operations", progress)
                ),
                "maven_launcher": "invoke Maven" in progress,
            }
        )
    inventory = []
    for name in sorted(
        {path.name.split(".")[0] for path in directory.glob("*.parquet")}
    ):
        paths = table_files(directory, name)
        schemas = [pl.read_parquet_schema(path) for path in paths]
        count = sum(
            pl.scan_parquet(path).select(pl.len()).collect().item() for path in paths
        )
        inventory.append(
            common
            | {
                "table": name,
                "rows": count,
                "files": json.dumps([str(path.relative_to(ROOT)) for path in paths]),
                "schemas": json.dumps(
                    [
                        {key: str(value) for key, value in schema.items()}
                        for schema in schemas
                    ]
                ),
            }
        )
    losses = re.findall(
        r"\b(?:lost|dropped)\s+(\d+)\s+(?:events|samples)",
        saved_path(event["stderr"]).read_text(errors="replace"),
        flags=re.IGNORECASE,
    )
    run = common | {
        "collection_directory": str(directory.relative_to(ROOT)),
        "process_start_ns": starts[0],
        "process_end_ns": ends[0],
        "rss_rows": rss.height,
        "rss_components": ";".join(components),
        "strict_resident_components_observed": all(
            name in components for name in RESIDENT
        ),
        "reported_lost_events": sum(map(int, losses)),
        "loss_completeness_known": False,
    }
    return run, member_rows, phase_rows, inventory


def audit(experiment, output):
    experiment, output = saved_path(experiment), saved_path(output)
    derived = ROOT / "data/derived/bloat"
    if not output.is_relative_to(derived) or output == derived:
        raise ValueError("output must be a new directory under data/derived/bloat")
    with (experiment / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        plan, events = full_runs(experiment)
        runs, members, phases, tables, excluded = [], [], [], [], {}
        for event in events:
            try:
                run, member_rows, phase_rows, inventory = audit_run(
                    event, plan["all_hooks"]
                )
                runs.append(run)
                members.extend(member_rows)
                phases.extend(phase_rows)
                tables.extend(inventory)
            except (ValueError, OSError, KeyError, pl.exceptions.PolarsError) as exc:
                excluded[event["collection_id"]] = str(exc)
        report = {
            "kind": "derived-bloat-audit",
            "source_experiment": str(experiment.relative_to(ROOT)),
            "source_plan_sha256": hashlib.sha256(
                (experiment / "plan.json").read_bytes()
            ).hexdigest(),
            "expected_collections": 60,
            "audited_collections": len(runs),
            "excluded": excluded,
            "by_thp": dict(Counter(row["thp"] for row in runs)),
            "strict_resident_components_observed": sum(
                row["strict_resident_components_observed"] for row in runs
            ),
            "component_coverage": dict(Counter(row["member"] for row in members)),
            "notes": [
                "Resident RSS requires anonymous + file + shmem; swap is separate.",
                "Unobserved components remain unknown, not zero. No backward fill.",
                "Process starts establish lifetime, not initial component values after fork/exec.",
                "Invocation times bracket launcher/Maven preparation as well as YCSB; reported runtime is diagnostic only.",
                "No observed loss messages does not establish lossless telemetry.",
                "This is a data-quality audit; no bloat labels, correlations or predictions have been computed.",
            ],
        }
        if output.exists() and any(output.iterdir()):
            raise ValueError("derived audit output already exists")
        output.mkdir(parents=True, exist_ok=True)
        for name, rows in (
            ("runs", runs),
            ("rss_components", members),
            ("phases", phases),
            ("tables", tables),
        ):
            frame = pl.DataFrame(rows)
            frame.write_parquet(output / f"{name}.parquet")
            frame.write_csv(output / f"{name}.csv")
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("data/experiments/redis-rss-dtlb-20260908"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/bloat/full-probes-20260908/audit"),
    )
    args = parser.parse_args()
    audit(args.experiment, args.output)


if __name__ == "__main__":
    main()
