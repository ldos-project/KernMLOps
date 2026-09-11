"""Unprivileged source tests; run with PYTHONPATH=python/kernmlops in make docker."""

import unittest
from pathlib import Path
from unittest.mock import call, patch

import polars as pl
from bcc import BPF, PerfType
from data_collection import GenericCollectorConfig
from data_collection.bpf_instrumentation.perf import perf_hook
from data_schema import table_types
from data_schema.perf import perf_table_types
from data_schema.perf.tlb_perf import (
    DTLBAccessPerfTable,
    DTLBPerfTable,
    DTLBWalkDurationPerfTable,
    ITLBPerfTable,
    TLBFlushPerfTable,
)

# Snapshots protect the existing record layout, handler body, and time units.
BASE_SOURCE = """#include <linux/ptrace.h>
#include <uapi/linux/bpf_perf_event.h>

#include "kermit.h"

typedef struct perf_event_data {
  u32 pid;
  u32 tgid;
  u64 ts_uptime_us;
  u32 count;
  u64 enabled_time_us;
  u64 running_time_us;
} tlb_perf_event_data_t;

// Handler code is generated in perf/perf_hook.py.
"""

ORIGINAL_HANDLER = """
KERMIT_OUTPUT(NAME, struct perf_event_data);
int NAME_on(struct bpf_perf_event_data* ctx) {
  struct bpf_perf_event_value value_buf;
  if (bpf_perf_prog_read_value(ctx, (void*)&value_buf, sizeof(struct bpf_perf_event_value))) {
    return 0;
  }
  struct perf_event_data *data = KERMIT_RESERVE(NAME, struct perf_event_data, 1);
  if (data == NULL) {
    return 0;
  }
  u32 pid = bpf_get_current_pid_tgid();
  u32 tgid = bpf_get_current_pid_tgid() >> 32;
  u64 ts = bpf_ktime_get_ns();
  data->pid = pid;
  data->tgid = tgid;
  data->ts_uptime_us = ts / 1000;
  data->count = value_buf.counter;
  data->enabled_time_us = value_buf.enabled / 1000;
  data->running_time_us = value_buf.running / 1000;
  KERMIT_SUBMIT(NAME, ctx, data);
  return 0;
}
"""


class PerfSourceTests(unittest.TestCase):
    def setUp(self):
        bpf_constructor = self.enterContext(
            patch.object(
                BPF,
                "__init__",
                side_effect=AssertionError("Source generation must not construct BPF"),
            )
        )
        capture = self.enterContext(
            patch.object(
                perf_hook,
                "capture_bpf",
                side_effect=AssertionError("Source generation must not compile BPF"),
            )
        )
        self.addCleanup(bpf_constructor.assert_not_called)
        self.addCleanup(capture.assert_not_called)
        self.hw_config = self.enterContext(
            patch.object(
                perf_hook.CustomHWConfigManager, "get_hw_config", return_value=None
            )
        )

    def test_base_source_and_handler_body_are_unchanged(self):
        source_path = Path(perf_hook.__file__).parent / "../bpf/perf.bpf.c"
        self.assertEqual(source_path.read_text(), BASE_SOURCE)
        self.assertEqual(perf_hook.PERF_HANDLER, ORIGINAL_HANDLER)

    def test_factory_filter_selects_only_rss_and_the_two_dtlb_counters(self):
        config = GenericCollectorConfig(
            hooks=["perf", "mm_rss_stat"],
            perf_events=["dtlb_misses", "dtlb_accesses"],
        )
        metadata = config.get_factory_metadata()
        self.assertEqual(
            metadata,
            {
                "hooks": ["perf", "mm_rss_stat"],
                "perf_events": ["dtlb_misses", "dtlb_accesses"],
            },
        )
        hooks = config.get_hooks()
        self.assertEqual([hook.name() for hook in hooks], ["perf", "mm_rss_stat"])
        self.assertEqual(
            hooks[0].loaded_hw_event_configs,
            {DTLBPerfTable: 0x10003, DTLBAccessPerfTable: 0x00003},
        )
        self.assertEqual(
            hooks[0].bpf_text,
            BASE_SOURCE
            + ORIGINAL_HANDLER.replace("NAME", "dtlb_misses")
            + ORIGINAL_HANDLER.replace("NAME", "dtlb_accesses"),
        )
        self.hw_config.assert_not_called()
        metadata["perf_events"].append("itlb_misses")
        self.assertEqual(config.perf_events, ["dtlb_misses", "dtlb_accesses"])

    def test_factory_filter_validation_and_default_metadata(self):
        self.assertEqual(
            GenericCollectorConfig(hooks=["perf"]).get_factory_metadata(),
            {"hooks": ["perf"]},
        )
        for events in ("dtlb_misses", [None], ["unknown"], ["dtlb_misses"] * 2):
            with self.subTest(events=events), self.assertRaises(ValueError):
                GenericCollectorConfig(hooks=["perf"], perf_events=events).get_hooks()
        with self.assertRaisesRegex(ValueError, "requires the perf hook"):
            GenericCollectorConfig(hooks=[], perf_events=["dtlb_misses"]).get_hooks()
        empty = GenericCollectorConfig(hooks=["perf"], perf_events=[]).get_hooks()[0]
        self.assertEqual(empty.loaded_hw_event_configs, {})
        self.assertEqual(empty.bpf_text, BASE_SOURCE)

    def test_default_and_none_add_access_without_reordering_existing_events(self):
        configs = {
            DTLBPerfTable: 0x10003,
            ITLBPerfTable: 0x10004,
            TLBFlushPerfTable: 0,
            DTLBWalkDurationPerfTable: 0x1234,
            DTLBAccessPerfTable: 0x00003,
        }
        self.assertEqual(perf_table_types, {event.name(): event for event in configs})
        self.hw_config.side_effect = configs.__getitem__
        for kwargs in ({}, {"events": None}):
            with self.subTest(kwargs=kwargs):
                self.hw_config.reset_mock()
                hook = perf_hook.PerfBPFHook(**kwargs)
                self.assertEqual(hook.loaded_hw_event_configs, configs)
                self.assertEqual(
                    hook._perf_data, {event.name(): [] for event in configs}
                )
                self.assertEqual(
                    hook.bpf_text,
                    BASE_SOURCE
                    + "".join(
                        perf_hook.PERF_HANDLER.replace("NAME", event.name())
                        for event in configs
                    ),
                )
                self.assertEqual(hook.bpf_text.count("int dtlb_accesses_on("), 1)
                self.assertEqual(
                    self.hw_config.call_args_list,
                    [call(TLBFlushPerfTable), call(DTLBWalkDurationPerfTable)],
                )

    def test_unavailable_raw_events_are_still_skipped(self):
        with patch("builtins.print") as info:
            hook = perf_hook.PerfBPFHook()
        self.assertEqual(
            hook.loaded_hw_event_configs,
            {
                DTLBPerfTable: 0x10003,
                ITLBPerfTable: 0x10004,
                DTLBAccessPerfTable: 0x00003,
            },
        )
        self.assertEqual(
            hook.bpf_text,
            BASE_SOURCE
            + perf_hook.PERF_HANDLER.replace("NAME", "dtlb_misses")
            + perf_hook.PERF_HANDLER.replace("NAME", "itlb_misses")
            + perf_hook.PERF_HANDLER.replace("NAME", "dtlb_accesses"),
        )
        self.assertEqual(
            info.call_args_list,
            [
                call("info: could not enable perf counter for tlb_flushes"),
                call("info: could not enable perf counter for dtlb_walk_duration"),
            ],
        )

    def test_explicit_dtlb_pair_generates_only_two_original_handlers_in_order(self):
        for events in (
            [DTLBPerfTable, DTLBAccessPerfTable],
            [DTLBAccessPerfTable, DTLBPerfTable],
        ):
            with self.subTest(events=events):
                hook = perf_hook.PerfBPFHook(events=events)
                self.assertEqual(list(hook.loaded_hw_event_configs), events)
                self.assertEqual(
                    hook.loaded_hw_event_configs,
                    {DTLBPerfTable: 0x10003, DTLBAccessPerfTable: 0x00003},
                )
                self.assertEqual(
                    hook._perf_data, {"dtlb_misses": [], "dtlb_accesses": []}
                )
                self.assertEqual(
                    hook.bpf_text,
                    BASE_SOURCE
                    + "".join(
                        perf_hook.PERF_HANDLER.replace("NAME", event.name())
                        for event in events
                    ),
                )
                self.assertIsNone(hook.group_fds)
                self.assertFalse(hasattr(hook, "bpf"))
        self.hw_config.assert_not_called()
        self.assertIs(perf_table_types["dtlb_accesses"], DTLBAccessPerfTable)

    def test_empty_selection_generates_only_base_declarations(self):
        hook = perf_hook.PerfBPFHook(events=[])
        self.assertEqual(hook.bpf_text, BASE_SOURCE)
        self.assertEqual(hook.loaded_hw_event_configs, {})
        self.assertEqual(hook._perf_data, {})
        self.assertEqual(hook.data(), [])
        self.hw_config.assert_not_called()

    def test_duplicate_names_are_rejected_even_for_unavailable_raw_events(self):
        class SameNameDTLBPerfTable(DTLBPerfTable):
            pass

        for events in (
            [DTLBPerfTable, DTLBPerfTable],
            [DTLBPerfTable, SameNameDTLBPerfTable],
            [TLBFlushPerfTable, TLBFlushPerfTable],
        ):
            with self.subTest(events=events):
                with self.assertRaisesRegex(ValueError, "names must be unique"):
                    perf_hook.PerfBPFHook(events=events)
        self.hw_config.assert_not_called()

    def test_access_table_metadata_and_default_data_are_registered(self):
        self.assertEqual(DTLBAccessPerfTable.name(), "dtlb_accesses")
        self.assertEqual(DTLBAccessPerfTable.ev_type(), PerfType.HW_CACHE)
        self.assertEqual(DTLBAccessPerfTable.hw_ids(), [])
        self.assertEqual(DTLBAccessPerfTable.component_name(), "dTLB")
        self.assertEqual(DTLBAccessPerfTable.measured_event_name(), "Read Accesses")
        self.assertEqual(
            DTLBAccessPerfTable.cumulative_column_name(), "cumulative_dtlb_accesses"
        )
        sample = perf_hook.PerfData(
            cpu=0,
            pid=7,
            tgid=7,
            ts_uptime_us=1234,
            cumulative_count=21,
            pmu_enabled_time_us=900,
            pmu_running_time_us=800,
        )
        events = [DTLBPerfTable, DTLBAccessPerfTable]
        with patch("builtins.print"):
            hook = perf_hook.PerfBPFHook()
        hook.collection_id = "test-id"
        for event in events:
            hook._perf_data[event.name()].append(sample)
        tables = hook.pop_data()
        self.assertEqual([type(table) for table in tables], events)
        for table, event in zip(tables, events, strict=True):
            self.assertEqual(dict(table.table.schema), dict(event.schema()))
            self.assertEqual(
                table.table[event.cumulative_column_name()].to_list(), [21]
            )
            self.assertEqual(table.table["collection_id"].to_list(), ["test-id"])
            self.assertEqual(table.table["ts_uptime_us"].to_list(), [1234])
            self.assertEqual(table.table["pmu_enabled_time_us"].to_list(), [900])
            self.assertEqual(table.table["pmu_running_time_us"].to_list(), [800])
            self.assertIs(table.filtered_table(), table.table)
        self.assertEqual(tables[1].graphs(), [])
        self.assertEqual(hook.pop_data(), [])
        empty = DTLBAccessPerfTable.from_df(
            pl.DataFrame(schema=DTLBAccessPerfTable.schema())
        )
        self.assertIsInstance(empty, DTLBAccessPerfTable)
        self.assertTrue(empty.table.is_empty())
        self.assertIn(DTLBAccessPerfTable, table_types)


if __name__ == "__main__":
    unittest.main()
