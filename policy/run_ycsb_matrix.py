"""Serial, balanced Redis/YCSB matrix. Native execution requires separate approval.

Run with PYTHONPATH=python/kernmlops in the collector environment. The extracted
collector owns manifest/artifact validation; factory metadata remains declarative.
No hook instances, VM-setting operations, or workload processes are created here
except through a fresh collector subprocess for each job.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("policy/build/container/rss_tlb/manifest.json")
MODES = ("none", "all", "reduced")
THPS = ("always", "never")
EXTRACTED_HOOK = "extracted_rss_dtlb"
MPMO_HOOK = "extracted_rss_dtlb_mpmo"
ORIGINAL_MODES = ("original-reduced", "original-mpmo")
ORIGINAL_HOOKS = dict(
    zip(ORIGINAL_MODES, ("original_rss_dtlb_capture", "original_rss_dtlb_mpmo"))
)
SELECTED_CONTROL_MODE = "selected-control"
SELECTED_CONTROL_HOOKS = ("perf", "mm_rss_stat")
SELECTED_CONTROL_EVENTS = ("dtlb_misses", "dtlb_accesses")
TIMEOUT = 300
INT_GRACE = 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def encoded(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def registry_names():
    from data_collection.bpf_instrumentation import all_hooks

    return [name for name in all_hooks if name != "cbmm"]


def configuration(raw):
    from cli.config import KernmlopsConfig

    config = KernmlopsConfig().merge(raw)
    return asdict(config), config.collector_config.generic.get_factory_metadata()


def check_hooks(config, mode, names, manifest=None):
    require(
        mode in (*MODES, "reduced-mpmo", *ORIGINAL_MODES, SELECTED_CONTROL_MODE),
        "unknown collector mode",
    )
    generic = config["collector_config"]["generic"]
    require("hooks" in generic and "cbmm" not in generic["hooks"], "cbmm is forbidden")
    require(
        generic["hooks"]
        == (
            list(SELECTED_CONTROL_HOOKS)
            if mode == SELECTED_CONTROL_MODE
            else names
            if mode == "all"
            else []
        ),
        "hook list mismatch",
    )
    if mode == SELECTED_CONTROL_MODE:
        require(
            generic["hooks"]
            == [name for name in names if name in SELECTED_CONTROL_HOOKS],
            "selected-control registry order mismatch",
        )
    require(
        generic.get("perf_events")
        == (list(SELECTED_CONTROL_EVENTS) if mode == SELECTED_CONTROL_MODE else None),
        "perf_events mismatch",
    )
    for selector, selected_mode in (
        ("extracted_rss_tlb", "reduced"),
        ("extracted_rss_tlb_mpmo", "reduced-mpmo"),
    ):
        expected = str(manifest) if mode == selected_mode else None
        require(generic.get(selector) == expected, f"{selector} manifest mismatch")
    require(
        generic.get("original_rss_tlb")
        == (str(manifest) if mode in ORIGINAL_MODES else None),
        "original_rss_tlb manifest mismatch",
    )
    require(
        generic.get("original_rss_tlb_transport", "capture")
        == ("mpmo" if mode == "original-mpmo" else "capture"),
        "original_rss_tlb_transport mismatch",
    )


def make_plan(
    directory,
    repeats,
    seed,
    manifest,
    mpmo_only=False,
    original_only=False,
    selected_control_only=False,
):
    import yaml

    require(repeats > 0, "--repeats must be positive")
    require(
        sum((mpmo_only, original_only, selected_control_only)) <= 1,
        "--original-only, --mpmo-only and --selected-control-only are incompatible",
    )
    directory = (ROOT / directory).resolve()
    artifact = None
    if selected_control_only:
        manifest = None
    else:
        manifest = (ROOT / manifest).resolve()
        content = manifest.read_bytes()
        require(isinstance(json.loads(content), dict), "manifest must be a JSON object")
        artifact = {"path": str(manifest), "sha256": digest(content)}
    original = None
    if original_only:
        from data_collection.bpf_instrumentation.original_rss_tlb import (
            original_manifest,
        )

        # Passive verification only; the child selects private BCC before any BCC import.
        original = original_manifest(manifest)
        require(
            original.get("version") == 1
            and original.get("kind") == "original-bcc-llvm",
            "unsupported original manifest",
        )
        for name in ("compiler_prefix", "kernel_source", "kernel_build"):
            require(
                isinstance(original.get(name), str)
                and Path(original[name]).is_absolute(),
                f"original manifest {name} must be an absolute path",
            )
    names = registry_names()
    require(
        len(names) == len(set(names)) == 12 and "cbmm" not in names,
        "expected exactly 12 registered hooks excluding cbmm",
    )
    plan = {
        "version": 1,
        "directory": str(directory),
        "root": str(ROOT),
        "experiment": re.sub(r"[^A-Za-z0-9_.-]", "_", directory.name)
        + "-"
        + digest(str(directory).encode())[:10],
        "repeats": repeats,
        "seed": seed,
        "all_hooks": names,
        "artifact": artifact,
        "base_configs": {},
        "cells": {},
        "jobs": [],
        "collector_sources": {},
    }
    for folder in (
        "python/kernmlops/data_collection",
        "python/kernmlops/kernmlops_benchmark",
        "python/kernmlops/cli",
    ) + (("python/kernmlops/data_schema",) if selected_control_only else ()):
        for source in sorted((ROOT / folder).rglob("*")):
            if source.is_file() and source.suffix in {".py", ".c", ".h"}:
                plan["collector_sources"][str(source.relative_to(ROOT))] = digest(
                    source.read_bytes()
                )
    for thp in THPS:
        base_path = ROOT / "config" / f"redis_{thp}.yaml"
        base_bytes = base_path.read_bytes()
        base = yaml.safe_load(base_bytes)
        plan["base_configs"][str(base_path)] = digest(base_bytes)
        for mode in (
            (SELECTED_CONTROL_MODE,)
            if selected_control_only
            else ORIGINAL_MODES
            if original_only
            else ("reduced-mpmo",)
            if mpmo_only
            else MODES
        ):
            raw = deepcopy(base)
            generic = raw["collector_config"]["generic"]
            generic["hooks"] = names.copy() if mode == "all" else []
            generic.pop("extracted_rss_tlb", None)
            generic.pop("extracted_rss_tlb_mpmo", None)
            generic.pop("original_rss_tlb", None)
            generic.pop("original_rss_tlb_transport", None)
            generic.pop("perf_events", None)
            if mode == SELECTED_CONTROL_MODE:
                generic["hooks"] = list(SELECTED_CONTROL_HOOKS)
                generic["perf_events"] = list(SELECTED_CONTROL_EVENTS)
            elif mode == "reduced":
                generic["extracted_rss_tlb"] = str(manifest)
            elif mode == "reduced-mpmo":
                generic["extracted_rss_tlb_mpmo"] = str(manifest)
            elif mode in ORIGINAL_MODES:
                generic["original_rss_tlb"] = str(manifest)
                generic["original_rss_tlb_transport"] = (
                    "mpmo" if mode == "original-mpmo" else "capture"
                )
            check_hooks(raw, mode, names, manifest)
            config, metadata = configuration(raw)
            check_hooks(config, mode, names, manifest)
            if mode == SELECTED_CONTROL_MODE:
                require(
                    metadata
                    == {
                        "hooks": generic["hooks"],
                        "perf_events": generic["perf_events"],
                    },
                    "selected-control factory metadata mismatch",
                )
            elif mode in ORIGINAL_MODES:
                require(
                    metadata.get("manifest") == original,
                    "original factory manifest mismatch",
                )
            setup = config["benchmark_config"]["generic"]
            require(
                setup["benchmark"] == "redis"
                and setup["transparent_hugepages"] == thp
                and setup["skip_clear_page_cache"] is False
                and setup["overcommit_memory"] == "never_check"
                and setup.get("khugepaged_scan_sleep_millis") is None,
                "base config must specify only the approved collector VM setup",
            )
            redis = config["benchmark_config"]["redis"]
            require(
                redis["repeat"] > 0 and redis["outer_repeat"] > 0,
                "Redis repeat counts must be positive",
            )
            key = f"{thp}-{mode}"
            plan["cells"][key] = {
                "thp": thp,
                "mode": mode,
                "config": config,
                "config_path": str(directory / "configs" / f"{key}.yaml"),
                "config_sha256": digest(encoded(config)),
                "factory_metadata": metadata,
                "expected_hooks": [EXTRACTED_HOOK]
                if mode == "reduced"
                else [MPMO_HOOK]
                if mode == "reduced-mpmo"
                else [ORIGINAL_HOOKS[mode]]
                if mode in ORIGINAL_MODES
                else generic["hooks"],
                "output_dir": str(
                    (
                        ROOT / config["collector_config"]["generic"]["output_dir"]
                    ).resolve()
                ),
            }
    rng = random.Random(seed)
    for replicate in range(repeats):
        order = list(plan["cells"])
        rng.shuffle(order)
        for cell in order:
            plan["jobs"].append(
                {
                    "key": f"{cell}-r{replicate:02d}",
                    "cell": cell,
                    "replicate": replicate,
                    "thp": plan["cells"][cell]["thp"],
                    "mode": plan["cells"][cell]["mode"],
                }
            )
    return plan


def validate_plan(plan, current):
    require(
        plan == current,
        "frozen plan differs: config, artifact, registry, repeats, or seed changed",
    )
    for cell in plan["cells"].values():
        content = Path(cell["config_path"]).read_bytes()
        require(digest(content) == cell["config_sha256"], "frozen config hash mismatch")
        # Generated JSON is also YAML, and freezes all effective dataclass defaults.
        check_hooks(
            json.loads(content),
            cell["mode"],
            plan["all_hooks"],
            None if cell["mode"] == SELECTED_CONTROL_MODE else plan["artifact"]["path"],
        )


def parse_cid(stdout, prefix):
    uuid = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    pattern = re.compile(re.escape(prefix) + "-" + uuid)
    with Path(stdout).open(encoding="utf-8", errors="replace") as stream:
        matches = [
            line.rstrip("\r\n")
            for line in stream
            if pattern.fullmatch(line.rstrip("\r\n"))
        ]
    require(
        len(matches) == 1,
        "stdout must contain exactly one standalone prefix-UUID collection ID",
    )
    return matches[0]


def read_rows(path, **kwargs):
    import polars as pl

    return pl.read_parquet(path, **kwargs).to_dicts()


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_mpmo_status(rows, cid):
    """Validate the final slot-0-only MPMO status; return True or raise ValueError."""
    require(
        len(rows) == 1 and isinstance(rows[0], dict),
        "expected exactly one MPMO policy_status row",
    )
    status = rows[0]
    require(status.get("collection_id") == cid, "MPMO collection ID mismatch")
    require(status.get("transport") == "mpmo", "MPMO transport mismatch")
    fixed = {
        "data_slots": 32,
        "read_slot": 0,
        "rss_schema_bytes": 24,
        "dtlb_schema_bytes": 64,
    }
    for name in (
        *fixed,
        "polls",
        "rss_slot0_reads",
        "dtlb_slot0_reads",
        "rss_empty",
        "dtlb_empty",
        "rss_busy",
        "dtlb_busy",
        "rss_checksum",
        "dtlb_checksum",
        "reader_errors",
        "rss_emit_failures",
        "rss_owner_misses",
        "dtlb_emit_failures",
        "dtlb_read_errors",
        "rss_unattributed",
    ):
        require(
            type(status.get(name)) is int and 0 <= status[name] < (1 << 64),
            f"MPMO {name} must be a nonnegative uint64",
        )
    for name, expected in fixed.items():
        require(status[name] == expected, f"MPMO {name} must be {expected}")
    require(status["polls"] > 0, "MPMO requires positive polls")
    for stream in ("rss", "dtlb"):
        require(
            status[f"{stream}_slot0_reads"] > 0,
            f"MPMO requires positive {stream}_slot0_reads",
        )
        require(
            status[f"{stream}_slot0_reads"]
            + status[f"{stream}_empty"]
            + status[f"{stream}_busy"]
            == status["polls"],
            f"MPMO {stream} reads + empty + busy must equal polls",
        )
    for name in ("reader_errors", "rss_owner_misses", "dtlb_read_errors"):
        require(status[name] == 0, f"MPMO {name} must be zero")
    return True


def validate_result(row, cell):
    cid = parse_cid(row["stdout"], row["prefix"])
    require(
        row.get("collection_id", cid) == cid, "saved collection ID differs from stdout"
    )
    directory = (
        Path(cell["output_dir"])
        / ("baseline" if cell["mode"] == "none" else "curated")
        / "redis"
        / cid
    )
    row.update(collection_id=cid, collection_dir=str(directory))
    paths = {
        name: str(directory / f"{name}.end.parquet")
        for name in ("system_info", "ycsb_invocations", "ycsb_metrics")
    }
    row["tables"] = paths
    system = read_rows(paths["system_info"])
    require(len(system) == 1, "expected one SystemInfo row")
    info = system[0]
    require(
        info["collection_id"] == cid
        and info["benchmark_name"] == "redis"
        and info["return_code"] == 0,
        "SystemInfo reports a failed or mismatched benchmark",
    )
    require(
        isinstance(info["hooks"], list)
        and Counter(info["hooks"]) == Counter(cell["expected_hooks"])
        and "cbmm" not in info["hooks"],
        "SystemInfo hook list mismatch",
    )
    invocations = read_rows(paths["ycsb_invocations"])
    redis = cell["config"]["benchmark_config"]["redis"]
    expected = [("load", outer, 0) for outer in range(redis["outer_repeat"])]
    expected += [
        ("run", outer, inner)
        for outer in range(redis["outer_repeat"])
        for inner in range(redis["repeat"])
    ]
    require(
        Counter((r["phase"], r["outer_iteration"], r["iteration"]) for r in invocations)
        == Counter(expected),
        "YCSB phase/repeat counts mismatch",
    )
    ids = set()
    for invocation in invocations:
        identifier = invocation["invocation_id"]
        require(
            type(identifier) is int and identifier not in ids,
            "duplicate/invalid invocation ID",
        )
        ids.add(identifier)
        require(
            invocation["collection_id"] == cid
            and invocation["benchmark_name"] == "redis"
            and invocation["run_label"] == row["prefix"]
            and invocation["execution_state"] == "completed"
            and invocation["exit_code"] == 0
            and invocation["parse_state"] == "ok",
            "YCSB invocation failed or did not parse",
        )
    overall = defaultdict(set)
    required = {"RunTime(ms)", "Throughput(ops/sec)"}
    for metric in read_rows(paths["ycsb_metrics"]):
        require(
            metric["collection_id"] == cid and metric["invocation_id"] in ids,
            "metric references an unknown invocation",
        )
        if metric["section"] == "OVERALL" and metric["metric"] in required:
            require(finite(metric["value"]), "non-finite OVERALL metric")
            overall[metric["invocation_id"]].add(metric["metric"])
    require(
        all(overall[identifier] == required for identifier in ids),
        "missing OVERALL metrics",
    )
    # Return=ERROR operation metrics are workload evidence, not invocation failures.
    if cell["mode"] == SELECTED_CONTROL_MODE:
        check_hooks(cell["config"], cell["mode"], SELECTED_CONTROL_HOOKS)
        telemetry = ("mm_rss_stat", *SELECTED_CONTROL_EVENTS)
        for path in directory.glob("*.parquet"):
            name = path.name.split(".")[0]
            require(
                name in (*paths, *telemetry),
                f"unexpected {name} output in selected-control mode",
            )
        for name in telemetry:
            paths[name] = [
                str(path) for path in sorted(directory.glob(f"{name}.*.parquet"))
            ]
            # RSS retains ns timestamps/member names; perf retains full us records.
            widths = {"pid": 32, "tgid": 32}
            widths.update(
                {"ts_ns": 64, "count": 64}
                if name == "mm_rss_stat"
                else {
                    "cpu": 64,
                    "ts_uptime_us": 64,
                    f"cumulative_{name}": 32,
                    "pmu_enabled_time_us": 64,
                    "pmu_running_time_us": 64,
                }
            )
            samples = 0
            for path in paths[name]:
                for sample in read_rows(path):
                    require(
                        sample.get("collection_id") == cid,
                        f"{name} collection ID mismatch",
                    )
                    for field, width in widths.items():
                        value = sample.get(field)
                        require(
                            type(value) is int and 0 <= value < (1 << width),
                            f"invalid selected-control {name} {field} width",
                        )
                    if name == "mm_rss_stat":
                        require(
                            sample.get("member")
                            in (
                                "MM_FILEPAGES",
                                "MM_ANONPAGES",
                                "MM_SWAPENTS",
                                "MM_SHMEMPAGES",
                            ),
                            "invalid selected-control mm_rss_stat member",
                        )
                    samples += 1
            require(samples > 0, f"no {name} samples")
    elif cell["mode"] == "reduced":
        for name in ("policy_rss", "policy_dtlb"):
            files = sorted(directory.glob(f"{name}.*.parquet"))
            paths[name] = [str(path) for path in files]
            require(
                any(read_rows(path, n_rows=1) for path in files), f"no {name} samples"
            )
        require(
            any(
                finite(r["dtlb_missrate"]) and r["dtlb_missrate"] >= 0
                for path in paths["policy_dtlb"]
                for r in read_rows(path, columns=["dtlb_missrate"])
            ),
            "no valid DTLB miss ratio",
        )
        paths["policy_status"] = [
            str(path) for path in sorted(directory.glob("policy_status.*.parquet"))
        ]
        status_rows = [
            status for path in paths["policy_status"] for status in read_rows(path)
        ]
        require(bool(status_rows), "missing extracted collector status")
        require(
            all(
                r["rss_owner_misses"] == 0 and r["dtlb_read_errors"] == 0
                for r in status_rows
            ),
            "extracted RSS/DTLB reported attribution or counter-read failures",
        )
        row["telemetry_status"] = status_rows
    elif cell["mode"] == "reduced-mpmo":
        for name in ("policy_rss", "policy_dtlb"):
            require(
                not any(directory.glob(f"{name}.*.parquet")),
                f"unexpected {name} output in reduced-mpmo mode",
            )
        paths["policy_status"] = [
            str(path) for path in sorted(directory.glob("policy_status.*.parquet"))
        ]
        require(
            paths["policy_status"] == [str(directory / "policy_status.end.parquet")],
            "expected only final MPMO policy_status output",
        )
        status_rows = read_rows(paths["policy_status"][0])
        row["telemetry_status"] = status_rows
        validate_mpmo_status(status_rows, cid)
    elif cell["mode"] in ORIGINAL_MODES:
        from data_collection.bpf_instrumentation.original_rss_tlb import (
            validate_original_status,
        )

        transport = "mpmo" if cell["mode"] == "original-mpmo" else "capture"
        paths["policy_status"] = [
            str(path) for path in sorted(directory.glob("policy_status.*.parquet"))
        ]
        require(
            paths["policy_status"] == [str(directory / "policy_status.end.parquet")],
            "expected only final original policy_status output",
        )
        status_rows = read_rows(paths["policy_status"][0])
        row["telemetry_status"] = status_rows
        validate_original_status(status_rows, cid, transport)
        evidence = status_rows[0]["compiler_evidence"]
        row["compiler_evidence_dir"] = evidence
        require(
            isinstance(evidence, str)
            and Path(evidence).is_absolute()
            and Path(evidence).is_dir(),
            "missing original compiler evidence directory",
        )
        compiled = json.loads(status_rows[0]["compiled_programs_json"])
        bundle = cell["factory_metadata"]["manifest"]
        require(
            compiled["sources"] == bundle["sources"]
            and compiled["bcc_library_sha256"]
            == bundle["files"]["bcc_library"]["sha256"],
            "original compiler provenance differs from frozen factory metadata",
        )
        require(
            set(compiled["programs"])
            == {
                "raw_tracepoint__rss_stat",
                "tracepoint__kmem__rss_stat",
                "dtlb_misses_on",
                "dtlb_accesses_on",
            },
            "unexpected original compiled programs",
        )
        for program, item in compiled["programs"].items():
            module = "rss" if program.endswith("rss_stat") else "perf"
            require(
                item["module"] == module
                and item["name"] == f"{module}.{program}.bpf.bin",
                "original bytecode path mismatch",
            )
            for name, sha in (
                (item["name"], item["sha256"]),
                (f"{module}.after.json", item["report_sha256"]),
                (f"{module}.generated.c", item["generated_source_sha256"]),
            ):
                path = Path(evidence) / name
                require(
                    path.resolve().is_relative_to(Path(evidence)),
                    "original evidence path escapes directory",
                )
                content = path.read_bytes()
                require(
                    bool(content) and digest(content) == sha,
                    "original compiler evidence hash mismatch",
                )
                if name == item["name"]:
                    require(
                        len(content) == item["size"] and len(content) % 8 == 0,
                        "invalid original BPF bytecode size",
                    )
        for name in (
            ("policy_rss", "original_perf", "policy_dtlb")
            if transport == "mpmo"
            else ("policy_dtlb",)
        ):
            require(
                not any(directory.glob(f"{name}.*.parquet")),
                f"unexpected {name} output in {cell['mode']} mode",
            )
        if transport == "capture":
            for name, widths in (
                ("policy_rss", {"tgid": 32, "ts_ns": 64, "member": -32, "count": 64}),
                (
                    "original_perf",
                    {
                        "cpu": 64,
                        "ts_uptime_us": 64,
                        "count": 32,
                        "enabled_time_us": 64,
                        "running_time_us": 64,
                    },
                ),
            ):
                paths[name] = [
                    str(path) for path in sorted(directory.glob(f"{name}.*.parquet"))
                ]
                streams = Counter()
                for path in paths[name]:
                    for sample in read_rows(path):
                        require(
                            sample.get("collection_id") == cid,
                            f"{name} collection ID mismatch",
                        )
                        for field, width in widths.items():
                            value = sample.get(field)
                            lower, upper = (
                                (0, 1 << width)
                                if width > 0
                                else (-(1 << (-width - 1)), 1 << (-width - 1))
                            )
                            require(
                                type(value) is int and lower <= value < upper,
                                f"invalid original {name} {field} width",
                            )
                        stream = (
                            sample.get("counter") if name == "original_perf" else "rss"
                        )
                        require(
                            stream
                            in (
                                ("dtlb_misses", "dtlb_accesses")
                                if name == "original_perf"
                                else ("rss",)
                            ),
                            "unexpected original counter",
                        )
                        streams[stream] += 1
                require(
                    all(
                        streams[stream] > 0
                        for stream in (
                            ("dtlb_misses", "dtlb_accesses")
                            if name == "original_perf"
                            else ("rss",)
                        )
                    ),
                    f"no {name} samples for every original stream",
                )


def child_environment(cell):
    """Select private BCC only for original collector subprocesses, never this process."""
    if cell["mode"] == SELECTED_CONTROL_MODE:
        for name in (
            "KERNMLOPS_BCC_REDUCE",
            "KERNMLOPS_BCC_EXPORT_ONLY",
            "KERNMLOPS_BCC_EVIDENCE",
        ):
            require(
                not os.environ.get(name),
                f"selected-control requires normal capture; unset {name}",
            )
    if cell["mode"] not in ORIGINAL_MODES:
        return None
    manifest = cell["factory_metadata"]["manifest"]
    prefix = Path(manifest["compiler_prefix"])
    environment = os.environ.copy()
    for name, path in (
        ("LD_LIBRARY_PATH", prefix / "lib"),
        ("PYTHONPATH", prefix / "lib/python3/site-packages"),
    ):
        existing = environment.get(name, "")
        environment[name] = str(path) + (os.pathsep + existing if existing else "")
    environment["BCC_KERNEL_SOURCE"] = manifest["kernel_source"]
    environment["BCC_KERNEL_BUILD"] = manifest["kernel_build"]
    return environment


def owned_directory(path, owner):
    if not path.parent.exists():
        owned_directory(path.parent, owner)
    path.mkdir(exist_ok=True)
    os.chown(path, *owner)


def write_json(path, value, owner, *, append=False):
    target = path if append else path.with_suffix(path.suffix + ".tmp")
    with target.open("ab" if append else "wb") as stream:
        os.fchown(stream.fileno(), *owner)
        stream.write(
            (json.dumps(value, allow_nan=False) + "\n").encode()
            if append
            else encoded(value)
        )
        stream.flush()
        os.fsync(stream.fileno())
    if not append:
        target.replace(path)


def stop_child(process):
    # Keep the leader unreaped through escalation so its process-group ID is not reused.
    previous = {
        sig: signal.signal(sig, signal.SIG_IGN)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for sig in (signal.SIGINT, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            if sig == signal.SIGINT:
                time.sleep(INT_GRACE)
        process.wait(timeout=5)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_matrix(
    directory,
    repeats=30,
    seed=0,
    manifest=MANIFEST,
    resume=False,
    limit=None,
    mpmo_only=False,
    original_only=False,
    selected_control_only=False,
):
    from data_schema import get_user_group_ids

    require(limit is None or limit >= 0, "--limit must be nonnegative")
    directory = (ROOT / directory).resolve()
    owner = get_user_group_ids()
    owned_directory(directory, owner)
    with (directory / ".lock").open("a") as lock:
        os.fchown(lock.fileno(), *owner)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path, progress = directory / "plan.json", directory / "results.jsonl"
        current = make_plan(
            directory,
            repeats,
            seed,
            manifest,
            mpmo_only=mpmo_only,
            original_only=original_only,
            selected_control_only=selected_control_only,
        )
        if resume:
            plan = json.loads(plan_path.read_bytes())
            validate_plan(plan, current)
        else:
            require(
                not plan_path.exists() and not progress.exists(),
                "experiment exists; use --resume",
            )
            plan = current
            for name in ("configs", "runs"):
                owned_directory(directory / name, owner)
            for cell in plan["cells"].values():
                require(not Path(cell["config_path"]).exists(), "config already exists")
                write_json(Path(cell["config_path"]), cell["config"], owner)
            write_json(plan_path, plan, owner)
        latest = {}
        jobs = {job["key"]: job for job in plan["jobs"]}
        if progress.exists():
            for line in progress.read_text().splitlines():
                event = json.loads(line)
                require(
                    event["key"] in jobs
                    and type(event["attempt"]) is int
                    and event["attempt"] > 0,
                    "invalid saved job/attempt identity",
                )
                require(
                    all(
                        event[name] == value
                        for name, value in jobs[event["key"]].items()
                    ),
                    "saved job does not match the frozen plan",
                )
                prefix = f"{plan['experiment']}-{event['key']}-a{event['attempt']:02d}"
                require(
                    event["prefix"] == prefix
                    and all(
                        event[source]
                        == str(directory / "runs" / f"{prefix}.{source}.log")
                        for source in ("stdout", "stderr")
                    ),
                    "saved attempt paths/label mismatch",
                )
                latest[event["key"]] = event
        completed, total = set(), len(plan["jobs"])

        def state(status, row=None):
            write_json(
                directory / "manifest.json",
                {
                    "status": status,
                    "completed": len(completed),
                    "total": total,
                    "plan": str(plan_path),
                    "results": str(progress),
                    "last_attempt": row,
                },
                owner,
            )

        def record(row):
            row["updated_at"] = time.time()
            write_json(progress, row, owner, append=True)
            latest[row["key"]] = row.copy()
            state("running" if row["status"] == "success" else row["status"], row)
            print(
                f"[{len(completed)}/{total}] {row['status']} {row['prefix']}",
                flush=True,
            )

        for job in plan["jobs"]:
            old = latest.get(job["key"])
            if old and old["status"] == "success":
                try:
                    require(
                        old["return_code"] == 0,
                        "saved collector exit status is not zero",
                    )
                    validate_result(old, plan["cells"][job["cell"]])
                    completed.add(job["key"])
                except Exception as exc:
                    old.update(status="invalidated", error=str(exc))
                    record(old)
            elif old and old["status"] == "running":
                old.update(
                    status="incomplete",
                    error="previous runner exited without a terminal record",
                )
                record(old)
        state("ready")
        print(
            f"Plan: {plan_path}; results: {progress}; status: {directory / 'manifest.json'}",
            flush=True,
        )
        launched = 0
        for job in plan["jobs"]:
            if job["key"] in completed:
                continue
            if limit is not None and launched >= limit:
                break
            # Recheck every frozen config, including explicit cbmm exclusion, before each run.
            try:
                validate_plan(
                    plan,
                    make_plan(
                        directory,
                        repeats,
                        seed,
                        manifest,
                        mpmo_only=mpmo_only,
                        original_only=original_only,
                        selected_control_only=selected_control_only,
                    ),
                )
            except Exception as exc:
                state("blocked", {**job, "error": str(exc)})
                raise
            cell = plan["cells"][job["cell"]]
            attempt = latest.get(job["key"], {}).get("attempt", 0) + 1
            prefix = f"{plan['experiment']}-{job['key']}-a{attempt:02d}"
            row = {
                **job,
                "attempt": attempt,
                "prefix": prefix,
                "status": "running",
                "config_path": cell["config_path"],
                "return_code": None,
                "stdout": str(directory / "runs" / f"{prefix}.stdout.log"),
                "stderr": str(directory / "runs" / f"{prefix}.stderr.log"),
            }
            record(row)
            process = None
            try:
                with (
                    Path(row["stdout"]).open("xb") as out,
                    Path(row["stderr"]).open("xb") as err,
                ):
                    os.fchown(out.fileno(), *owner)
                    os.fchown(err.fileno(), *owner)
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(ROOT / "python/kernmlops"),
                            "collect",
                            "data",
                            "--config-file",
                            cell["config_path"],
                            "--collection-prefix",
                            prefix,
                        ],
                        cwd=ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                        pass_fds=(lock.fileno(),),
                        env=child_environment(cell),
                    )
                    row["return_code"] = process.wait(timeout=TIMEOUT)
                require(
                    row["return_code"] == 0, f"collector exited {row['return_code']}"
                )
                validate_result(row, cell)
                row["status"] = "success"
                completed.add(job["key"])
            except BaseException as exc:
                row.update(
                    status="incomplete"
                    if isinstance(exc, (KeyboardInterrupt, SystemExit))
                    else "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                if process is not None and process.returncode is None:
                    try:
                        stop_child(process)
                        row["return_code"] = process.returncode
                    except Exception as cleanup:
                        row["error"] += f"; cleanup failed: {cleanup}"
                        if process.returncode is None:
                            row["status"] = "running"
                            state("running", row)
                            print(row["error"], file=sys.stderr, flush=True)
                            return 1
                # Even a nonzero collector exit can have useful partial output.
                try:
                    cid = parse_cid(row["stdout"], prefix)
                    row.update(
                        collection_id=cid,
                        collection_dir=str(
                            Path(cell["output_dir"])
                            / ("baseline" if cell["mode"] == "none" else "curated")
                            / "redis"
                            / cid
                        ),
                    )
                except (OSError, ValueError):
                    pass
            record(row)
            launched += 1
            if row["status"] != "success":
                return 130 if row["status"] == "incomplete" else 1
        state("complete" if len(completed) == total else "paused")
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Repository-relative or absolute experiment directory",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--mpmo-only",
        action="store_true",
        help="Run only reduced-mpmo for both THP settings; requires an MPMO --manifest",
    )
    modes.add_argument(
        "--original-only",
        action="store_true",
        help="Run original-reduced/capture and original-mpmo for both THP settings; requires an original --manifest",
    )
    modes.add_argument(
        "--selected-control-only",
        action="store_true",
        help="Capture full original RSS and DTLB READ MISS/ACCESS only for both THP settings; no reduction manifest",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum new jobs; 0 freezes/validates without launching a collector",
    )
    args = parser.parse_args(argv)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupt)
    vm_before = None
    try:
        if args.limit != 0:
            # Snapshot only the two host settings this matrix intentionally changes.
            thp = Path("/sys/kernel/mm/transparent_hugepage/enabled")
            overcommit = Path("/proc/sys/vm/overcommit_memory")
            current = re.search(r"\[([^\]]+)\]", thp.read_text())
            require(current is not None, "cannot identify the active THP setting")
            value = overcommit.read_text().strip()
            require(value in {"0", "1", "2"}, "unexpected overcommit setting")
            vm_before = (thp, current.group(1), overcommit, value)
            print(
                f"Host settings will be restored after this session: THP={current.group(1)}, overcommit={value}",
                flush=True,
            )
        return run_matrix(**vars(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Matrix stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if vm_before is not None:
            thp, thp_value, overcommit, overcommit_value = vm_before
            thp.write_text(thp_value)
            overcommit.write_text(overcommit_value)
            print(
                "Restored host THP and overcommit settings; page-cache contents cannot be restored.",
                flush=True,
            )
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
