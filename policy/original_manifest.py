"""Freeze the original-source compiler, selections, and compiler-produced layouts."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = {
    "rss": {
        "rss_stat_output": {
            "program": "tracepoint__kmem__rss_stat",
            "record": "rss_stat_output",
            "fields": ["tgid", "ts", "member", "counter_value"],
        }
    },
    "perf": {
        name: {
            "program": name + "_on",
            "record": "perf_event_data",
            "fields": ["ts_uptime_us", "count", "enabled_time_us", "running_time_us"],
        }
        for name in ("dtlb_misses", "dtlb_accesses")
    },
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--reports", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error("bundle directory must already exist")
    if args.reports is None:
        for module, outputs in SELECTIONS.items():
            (directory / f"{module}.selection.json").write_text(
                json.dumps({"version": 1, "outputs": outputs}, indent=2) + "\n"
            )
        return
    schemas = {}
    for transport in ("capture", "mpmo"):
        schemas[transport] = {}
        for module in SELECTIONS:
            report = json.loads(
                (args.reports / f"{module}-{transport}.after.json").read_text()
            )
            if report["transport"] != transport or report["version"] != 1:
                parser.error("unexpected compiler report")
            for output, schema in report["schemas"].items():
                if transport == "capture" and schema["slots"] is None:
                    schema["slots"] = 0
                schemas[transport][output] = schema
    sources = {}
    for folder in (
        ROOT / "python/kernmlops/data_collection",
        ROOT / "python/kernmlops/data_schema",
    ):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in (".c", ".h", ".py"):
                sources[str(path.relative_to(ROOT))] = digest(path)
    for name in (
        "bcc_frontend.patch",
        "bcc_reduce.hpp",
        "original_reduce.cpp",
        "original_reduce.hpp",
        "original_queue.c",
        "original_manifest.py",
        "original_bcc.mk",
        "original_reduce.mk",
        "inspect_original.py",
    ):
        path = ROOT / "policy" / name
        sources[str(path.relative_to(ROOT))] = digest(path)
    files = {
        key: {"name": name, "sha256": digest(directory / name)}
        for key, name in {
            "bcc_library": "install/lib/libbcc.so.0",
            "queue_library": "liboriginal_queue.so",
            "rss_selection": "rss.selection.json",
            "perf_selection": "perf.selection.json",
        }.items()
    }
    manifest = {
        "version": 1,
        "kind": "original-bcc-llvm",
        "compiler_prefix": str(directory / "install"),
        "sample_frequency": 1000,
        "queue_slots": 32,
        "reader_slot": 0,
        "kernel_source": "/kernel-source",
        "kernel_build": "/kernel-build",
        "bcc_commit": "4752b9906825bfc3b86e4590293fc55d7a5db4f8",
        "files": files,
        "sources": sources,
        "schemas": schemas,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
