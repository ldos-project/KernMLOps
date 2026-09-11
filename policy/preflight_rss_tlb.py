"""Short attachment/feature check. No Redis workload or VM-setting changes."""

import argparse
import json
import os
import time
import uuid

import polars as pl
from data_collection import GenericCollectorConfig
from data_collection.bpf_instrumentation import all_hooks
from run_ycsb_matrix import SELECTED_CONTROL_EVENTS, SELECTED_CONTROL_HOOKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all", action="store_true", help="Check all registered hooks except cbmm"
    )
    mode.add_argument("--original", choices=("capture", "mpmo"))
    mode.add_argument("--selected-control", action="store_true")
    parser.add_argument(
        "--manifest", default="policy/build/container/rss_tlb/manifest.json"
    )
    args = parser.parse_args()
    names = [name for name in all_hooks if name != "cbmm"] if args.all else []
    config = (
        GenericCollectorConfig(
            hooks=list(SELECTED_CONTROL_HOOKS),
            perf_events=list(SELECTED_CONTROL_EVENTS),
        )
        if args.selected_control
        else GenericCollectorConfig(
            hooks=[],
            original_rss_tlb=args.manifest,
            original_rss_tlb_transport=args.original,
        )
        if args.original
        else GenericCollectorConfig(
            hooks=names, extracted_rss_tlb=None if args.all else args.manifest
        )
    )
    hooks = config.get_hooks()
    collection_id = (
        f"preflight-original-{args.original}-{uuid.uuid4()}"
        if args.original
        else "preflight"
    )
    loaded = []
    try:
        for hook in hooks:
            assert hook.name() != "cbmm"
            print(f"Loading {hook.name()}", flush=True)
            hook.load(collection_id)
            loaded.append(hook)
        memory = bytearray(32 * 1024 * 1024)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            for offset in range(0, len(memory), 4096):
                memory[offset] = (memory[offset] + 1) % 256
            for hook in hooks:
                hook.poll()
        del memory
    finally:
        errors = []
        for hook in reversed(loaded):
            try:
                hook.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"preflight cleanup failed: {errors}")
    if args.selected_control:
        tables = {
            table.name(): table.table for hook in hooks for table in hook.pop_data()
        }
        assert set(tables) == {"mm_rss_stat", *SELECTED_CONTROL_EVENTS}
        for name, table in tables.items():
            assert table.height > 0, f"no {name} samples"
            assert {"pid", "tgid"} <= set(table.columns), "expected full records"
        assert tables["mm_rss_stat"].filter(pl.col("tgid") == os.getpid()).height > 0
        print(
            json.dumps({name: table.height for name, table in tables.items()}, indent=2)
        )
    elif args.original:
        from data_collection.bpf_instrumentation.original_rss_tlb import (
            validate_original_status,
        )

        tables = {table.name(): table.table for table in hooks[0].pop_data()}
        status = tables["policy_status"].to_dicts()
        validate_original_status(status, collection_id, args.original)
        if args.original == "capture":
            assert tables["policy_rss"].filter(pl.col("tgid") == os.getpid()).height > 0
            assert set(tables["original_perf"]["counter"]) == {
                "dtlb_misses",
                "dtlb_accesses",
            }
        else:
            assert set(tables) == {"policy_status"}
        print(
            json.dumps(
                {k: v for k, v in status[0].items() if k != "compiled_programs_json"},
                indent=2,
            )
        )
    elif not args.all:
        tables = {table.name(): table.table for table in hooks[0].pop_data()}
        rss = tables["policy_rss"]
        tlb = tables["policy_dtlb"]
        status = tables["policy_status"]
        own_rss = rss.filter(pl.col("tgid") == os.getpid())
        ratios = tlb.filter(pl.col("dtlb_missrate").is_not_null())
        print(f"RSS events={rss.height}; own-process RSS events={own_rss.height}")
        print(f"DTLB events={tlb.height}; valid ratios={ratios.height}")
        print(status)
        assert own_rss.height > 0, "no correctly attributed RSS updates"
        assert ratios.height > 0, "no DTLB misses/access ratio samples"
        assert status.select("rss_owner_misses", "dtlb_read_errors").row(0) == (0, 0)
        print(
            "Ring-output losses are recorded; cumulative DTLB snapshots span missed notifications."
        )
    print("Preflight passed; cbmm was not loaded", flush=True)


if __name__ == "__main__":
    main()
