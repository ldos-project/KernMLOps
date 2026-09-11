"""Export original probe IR through the private BCC frontend without loading BPF."""

import argparse
import json
import os
from pathlib import Path

from data_collection.bpf_instrumentation.bpf_hook import compile_bpf
from data_collection.bpf_instrumentation.perf.perf_hook import PerfBPFHook
from data_schema.perf.tlb_perf import DTLBAccessPerfTable, DTLBPerfTable

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "python/kernmlops/data_collection/bpf_instrumentation/bpf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=("rss", "perf"), required=True)
    parser.add_argument("--transport", choices=("capture", "mpmo"), required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    if not prefix.parent.is_dir():
        parser.error("evidence parent directory must already exist")
    import bcc

    if "/original_bcc/install/" not in str(Path(bcc.__file__).resolve()):
        parser.error("use the isolated BCC install, not the system BCC")
    if args.stream == "rss":
        source = (SOURCES / "mm_trace_rss_stat.bpf.c").read_text()
    else:
        source = PerfBPFHook(events=[DTLBPerfTable, DTLBAccessPerfTable]).bpf_text
    with Path(f"{prefix}.generated.c").open("x") as output:
        output.write(source)
    os.environ["KERNMLOPS_BCC_EXPORT_ONLY"] = "1"
    os.environ["KERNMLOPS_BCC_EVIDENCE"] = str(prefix)
    if args.selection:
        os.environ["KERNMLOPS_BCC_REDUCE"] = str(args.selection.resolve())
    else:
        os.environ.pop("KERNMLOPS_BCC_REDUCE", None)
    flags = (
        ("-DKERMIT_CAPTURE",)
        if args.transport == "capture"
        else ("-DKERMIT_QUEUE_SIZE=32",)
    )
    if args.transport == "mpmo" and args.stream == "perf":
        flags += ("-DKERMIT_CPU_METADATA",)
    if kernel_build := os.environ.get("BCC_KERNEL_BUILD"):
        flags += tuple(
            f"-I{kernel_build}/{suffix}"
            for suffix in (
                "include",
                "arch/x86/include/generated",
                "include/generated/uapi",
                "arch/x86/include/generated/uapi",
            )
        )
    try:
        probe = compile_bpf(source, cflags=flags)
    except Exception:
        metadata = Path(f"{prefix}.before.json")
        complete = all(
            Path(f"{prefix}{suffix}").is_file()
            and Path(f"{prefix}{suffix}").stat().st_size > 0
            for suffix in (".before.bc", ".before.ll", ".before.json", ".rewritten.c")
        )
        if not complete or not json.loads(metadata.read_text())["export_only"]:
            raise
        if args.selection:
            report = json.loads(Path(f"{prefix}.after.json").read_text())
            if not report["export_only"] or report["transport"] != args.transport:
                raise RuntimeError("missing or invalid reduced export")
        print(f"Exported {args.stream}/{args.transport} before BPF loading: {prefix}")
    else:
        probe.cleanup()
        raise RuntimeError("private BCC did not stop before loading")


if __name__ == "__main__":
    main()
