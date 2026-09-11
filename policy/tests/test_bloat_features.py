"""Unprivileged, offline tests for full-probe causal feature construction."""

import json
import os
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from bloat.features import build_features

NS = 1_000_000_000
PID = 115476
MIB = 2**20
COUNTERS = ("dtlb_misses", "itlb_misses", "tlb_flushes", "dtlb_walk_duration")


class BloatFeaturesTests(unittest.TestCase):
    def build(self, tables, ticks=(NS, 2 * NS), **kwargs):
        return build_features(
            tables,
            kwargs.pop("redis_pid", PID),
            np.asarray(ticks, dtype=np.int64),
            phase_start_ns=kwargs.pop("phase_start_ns", 0),
            logical_cpus=kwargs.pop("logical_cpus", 4),
            **kwargs,
        )

    def assert_values(self, features, group, scope, stat, expected):
        np.testing.assert_allclose(
            features[f"f__{group}__{scope}__{stat}"], expected, equal_nan=True
        )

    def counter(self, name="dtlb_misses", **changes):
        data = {
            "collection_id": ["a"] * 3,
            "cpu": [0, 0, 0],
            "ts_uptime_us": [100_000, 200_000, 300_000],
            f"cumulative_{name}": [100, 110, 120],
            "pmu_enabled_time_us": [10, 110, 210],
            "pmu_running_time_us": [10, 60, 110],
        }
        data.update(changes)
        return pl.DataFrame(data)

    def test_catalog_contract_and_missing_tables(self):
        features, catalog, qc = self.build({})
        self.assertEqual(len(features), 56)
        self.assertEqual(set(features), {entry["name"] for entry in catalog})
        for entry in catalog:
            self.assertEqual(
                set(entry),
                {
                    "name",
                    "group",
                    "scope",
                    "unit",
                    "description",
                    "target_linked",
                    "availability",
                    "limitations",
                },
            )
            self.assertFalse(entry["target_linked"])
            self.assertTrue(entry["name"].startswith("f__"))
            self.assertTrue(entry["availability"])
            self.assertTrue(entry["limitations"])
        for values in features.values():
            self.assertEqual(values.dtype, np.float64)
            self.assertEqual(values.shape, (2,))
            self.assertTrue(np.isnan(values).all())
        self.assertEqual(qc["tables"]["file_data"]["status"], "missing_table")
        json.dumps(catalog, allow_nan=False)
        json.dumps(qc, allow_nan=False)

    def test_empty_event_counts_vs_metadata_only_unknown(self):
        empty = pl.DataFrame({"collection_id": pl.Series([], dtype=pl.String)})
        features, catalog, qc = self.build(
            {"zswap_runtime": empty, "quanta_runtime": empty, "dtlb_misses": empty}
        )
        for operation in ("store", "load", "invalidate"):
            for scope in ("redis", "system"):
                self.assert_values(
                    features,
                    "zswap_runtime",
                    scope,
                    f"{operation}_completed_count",
                    [0, 0],
                )
                self.assert_values(
                    features,
                    "zswap_runtime",
                    scope,
                    f"{operation}_latency_mean_ms",
                    [np.nan, np.nan],
                )
        self.assert_values(
            features, "quanta_runtime", "redis", "completed_count", [0, 0]
        )
        self.assert_values(
            features, "quanta_runtime", "redis", "duration_sum_s", [np.nan, np.nan]
        )
        self.assert_values(
            features, "dtlb_misses", "system", "rate_per_s", [np.nan, np.nan]
        )
        self.assertIn("observed_zero", qc["tables"]["zswap_runtime"]["zero_semantics"])
        self.assertIn("capture_completeness_unknown", json.dumps(catalog))
        metadata_only = pl.DataFrame({"collection_id": ["a"]})
        features, _, qc = self.build({"zswap_runtime": metadata_only})
        self.assert_values(
            features,
            "zswap_runtime",
            "redis",
            "store_completed_count",
            [np.nan, np.nan],
        )
        self.assertEqual(qc["tables"]["zswap_runtime"]["status"], "unknown_schema")

    def test_scheduler_full_tgid_scope_completion_guard_and_no_cutoffs(self):
        for table, duration in (
            ("quanta_runtime", "quanta_run_length_us"),
            ("quanta_queued_time", "quanta_queued_time_us"),
        ):
            with self.subTest(table=table):
                frame = pl.DataFrame(
                    {
                        "tgid": [PID, PID, PID + 1, PID, PID, PID],
                        "pid": [PID, PID + 7, PID, PID, PID, PID],
                        "ts_uptime_us": [
                            1_100_000,
                            1_900_000,
                            1_500_000,
                            2_000_000,
                            1_800_000,
                            -1,
                        ],
                        duration: [100_000, 1_200_000, 200_000, 300_000, -1, 10],
                    }
                )
                ticks = [2 * NS, 2 * NS + 1000, 3 * NS]
                features, _, qc = self.build(
                    {table: frame}, ticks=ticks, phase_start_ns=NS
                )
                self.assert_values(
                    features, table, "redis", "completed_count", [2, 3, 1]
                )
                self.assert_values(
                    features, table, "redis", "duration_sum_s", [1.3, 1.6, 0.3]
                )
                self.assert_values(
                    features, table, "redis", "duration_mean_ms", [650, 1600 / 3, 300]
                )
                self.assert_values(
                    features, table, "redis", "duration_p95_ms", [1145, 1110, 300]
                )
                self.assert_values(
                    features,
                    table,
                    "system_other",
                    "duration_sum_s",
                    [0.2, 0.2, np.nan],
                )
                self.assertEqual(qc["tables"][table]["rejected_rows"], 2)
                changed = frame.with_columns(
                    pl.when(pl.col("ts_uptime_us") >= 2_000_000)
                    .then(900_000)
                    .otherwise(pl.col(duration))
                    .alias(duration)
                )
                later, _, _ = self.build(
                    {table: changed}, ticks=ticks, phase_start_ns=NS
                )
                for name in features:
                    np.testing.assert_equal(features[name][0], later[name][0])

    def test_entry_probes_windows_units_and_invalid_ranges(self):
        file_data = pl.DataFrame(
            {
                "ts_uptime_us": [1_000_000, 1_500_000, 1_600_000, 2_000_000],
                "tgid": [PID, PID, PID + 1, PID],
                "pid": [PID + 9, PID + 10, PID, PID],
                "file_inode": [7, 7, 7, 8],
                "file_size_bytes": [MIB, 3 * MIB, 8 * MIB, 100 * MIB],
            }
        )
        unmap = pl.DataFrame(
            {
                "ts_ns": [NS, NS + 1, 2 * NS, NS + 3, NS + 4],
                "tgid": [PID, PID, PID, PID, PID + 1],
                "start": [0, 0, MIB, 100, 0],
                "end": [MIB, MIB, 3 * MIB, 99, 9 * MIB],
                "is_huge": [True] * 5,
            }
        )
        features, _, qc = self.build(
            {"file_data": file_data, "unmap_range": unmap},
            ticks=[2 * NS, 3 * NS],
            phase_start_ns=NS,
        )
        self.assert_values(features, "file_data", "redis", "observed_hit_count", [2, 1])
        self.assert_values(
            features, "file_data", "redis", "inode_size_mean_mib", [2, 100]
        )
        self.assert_values(
            features, "file_data", "system", "observed_hit_count", [3, 1]
        )
        self.assert_values(
            features, "unmap_range", "redis", "observed_request_count", [2, 0]
        )
        self.assert_values(
            features, "unmap_range", "redis", "requested_sum_mib", [3, np.nan]
        )
        self.assertEqual(qc["tables"]["unmap_range"]["rejected_rows"], 1)
        self.assertFalse(
            any("freed" in key or "unique_inode" in key for key in features)
        )

    def test_zswap_end_availability_and_executing_scope(self):
        frame = pl.DataFrame(
            {
                "name": [
                    "zswap_store",
                    "zswap_store",
                    "zswap_load",
                    "zswap_invalidate",
                    "zswap_store",
                    "unknown",
                ],
                "start_ts": [
                    NS - 10_000_000,
                    NS - 20_000_000,
                    NS - 1,
                    NS // 2,
                    3 * NS,
                    1,
                ],
                "end_ts": [NS, NS + 1, NS, NS // 2 + 2_000_000, 2 * NS, 2],
                "tgid": [PID, PID, PID + 1, PID, PID, PID],
                "pid": [PID + 1, PID + 2, PID, PID + 3, PID, PID],
            }
        )
        features, _, qc = self.build({"zswap_runtime": frame})
        self.assert_values(
            features, "zswap_runtime", "redis", "store_completed_count", [1, 1]
        )
        self.assert_values(
            features, "zswap_runtime", "redis", "store_latency_mean_ms", [10, 20.000001]
        )
        self.assert_values(
            features, "zswap_runtime", "redis", "load_completed_count", [0, 0]
        )
        self.assert_values(
            features, "zswap_runtime", "system", "load_completed_count", [1, 0]
        )
        self.assert_values(
            features,
            "zswap_runtime",
            "redis",
            "invalidate_latency_mean_ms",
            [2, np.nan],
        )
        self.assertEqual(qc["tables"]["zswap_runtime"]["rejected_rows"], 2)

    def test_scan_handler_end_owner_scope_and_no_runtime_or_unmapped(self):
        frame = pl.DataFrame(
            {
                "start_ts_ns": [10, NS - 1, NS - 1, 20],
                "end_ts_ns": [20, NS, NS + 1, 10],
                "tgid": [PID, PID, PID, PID],
                "pid": [PID + 4] * 4,
                "none_or_zero": [2, 6, 999, 999],
                "referenced": [10, 30, 999, 999],
                "writeable": [0, 2, 0, 0],
                "unmapped": [True] * 4,
            }
        )
        features, catalog, _ = self.build({"trace_mm_khugepaged_scan_pmd": frame})
        self.assert_values(features, "scan_pmd", "redis", "observed_count", [2, 1])
        self.assert_values(features, "scan_pmd", "redis", "none_or_zero_mean", [4, 999])
        self.assert_values(features, "scan_pmd", "redis", "referenced_mean", [20, 999])
        self.assert_values(
            features, "scan_pmd", "redis", "writeable_fraction", [0.5, 0]
        )
        self.assertFalse(
            any(
                "duration" in key or "unmapped" in key
                for key in features
                if "scan_pmd" in key
            )
        )
        scan_catalog = [entry for entry in catalog if entry["group"] == "scan_pmd"]
        self.assertIn("mm->owner", json.dumps(scan_catalog))
        changed = frame.with_columns(pl.lit(False).alias("unmapped"))
        other, _, _ = self.build({"trace_mm_khugepaged_scan_pmd": changed})
        for key in features:
            np.testing.assert_equal(features[key], other[key])

    def test_counters_cpu_group_sort_scale_and_ignore_pid(self):
        for name in COUNTERS:
            with self.subTest(counter=name):
                first = self.counter(name)
                second = self.counter(
                    name, cpu=[1, 1, 1], **{f"cumulative_{name}": [1000, 1030, 1060]}
                )
                frame = (
                    pl.concat([first, second])
                    .reverse()
                    .with_columns(
                        pl.lit(PID + 30).alias("tgid"), pl.lit(PID + 50).alias("pid")
                    )
                )
                features, _, qc = self.build({name: frame})
                self.assert_values(
                    features, name, "system", "rate_per_s", [800, np.nan]
                )
                self.assert_values(
                    features, name, "system", "coverage_cpu_fraction", [0.5, np.nan]
                )
                self.assert_values(
                    features, name, "system", "mean_covered_fraction", [0.1, np.nan]
                )
                self.assertEqual(qc["tables"][name]["interval_qc"]["first_samples"], 2)
                self.assertEqual(
                    qc["tables"][name]["interval_qc"]["usable_intervals"], 4
                )
                other, _, _ = self.build({name: frame.drop("pid", "tgid")}, redis_pid=1)
                for key in features:
                    np.testing.assert_equal(features[key], other[key])

    def test_counter_both_endpoints_inside_window_phase_and_quantization(self):
        frame = self.counter(ts_uptime_us=[1_000_000, 1_100_000, 1_200_000])
        features, _, _ = self.build(
            {"dtlb_misses": frame},
            ticks=[1_200_000_000, 1_200_001_000, 2_000_000_000],
            phase_start_ns=NS,
            window_ns=200_000_000,
        )
        self.assert_values(
            features, "dtlb_misses", "system", "rate_per_s", [np.nan, 200, np.nan]
        )
        self.assert_values(
            features,
            "dtlb_misses",
            "system",
            "mean_covered_fraction",
            [np.nan, 0.125, np.nan],
        )
        features, _, qc = self.build(
            {"dtlb_misses": frame}, ticks=[2 * NS], phase_start_ns=1_150_000_000
        )
        self.assert_values(features, "dtlb_misses", "system", "rate_per_s", [np.nan])
        self.assertEqual(qc["tables"]["dtlb_misses"]["interval_qc"]["before_phase"], 2)
        features, _, _ = self.build({"dtlb_misses": frame[:1]})
        self.assert_values(
            features, "dtlb_misses", "system", "rate_per_s", [np.nan, np.nan]
        )

    def test_counter_invalid_intervals_do_not_bridge_or_invent_wraps(self):
        cases = [
            (
                {"cumulative_dtlb_misses": [2**32 - 2, 3, 13]},
                "negative_counter_delta",
                1,
                200,
            ),
            (
                {"cumulative_dtlb_misses": [100, 90, 80]},
                "negative_counter_delta",
                2,
                np.nan,
            ),
            (
                {"cumulative_dtlb_misses": [100, 2**32, 110]},
                "invalid_endpoint",
                2,
                np.nan,
            ),
            ({"cumulative_dtlb_misses": [100, -1, 110]}, "invalid_endpoint", 2, np.nan),
            (
                {"ts_uptime_us": [100_000, 100_000, 200_000]},
                "nonincreasing_timestamp",
                1,
                200,
            ),
            (
                {"ts_uptime_us": [100_000, 350_001, 600_002]},
                "gap_over_250ms",
                2,
                np.nan,
            ),
            ({"pmu_running_time_us": [10, 10, 10]}, "invalid_pmu_delta", 2, np.nan),
            ({"pmu_enabled_time_us": [10, 10, 10]}, "invalid_pmu_delta", 2, np.nan),
            ({"pmu_running_time_us": [10, 210, 410]}, "invalid_pmu_delta", 2, np.nan),
            ({"pmu_running_time_us": [10, 0, 110]}, "invalid_endpoint", 2, np.nan),
            ({"pmu_running_time_us": [10, 9, 8]}, "invalid_pmu_delta", 2, np.nan),
        ]
        for changes, reason, expected_count, expected_rate in cases:
            with self.subTest(changes=changes):
                features, _, qc = self.build({"dtlb_misses": self.counter(**changes)})
                self.assert_values(
                    features,
                    "dtlb_misses",
                    "system",
                    "rate_per_s",
                    [expected_rate, np.nan],
                )
                self.assertEqual(
                    qc["tables"]["dtlb_misses"]["interval_qc"][reason], expected_count
                )
        features, _, _ = self.build(
            {
                "dtlb_misses": self.counter(
                    ts_uptime_us=[100_000, 350_000, 600_000],
                    cumulative_dtlb_misses=[100, 100, 100],
                )
            }
        )
        self.assert_values(features, "dtlb_misses", "system", "rate_per_s", [0, np.nan])
        self.assert_values(
            features, "dtlb_misses", "system", "mean_covered_fraction", [0.125, np.nan]
        )

    def test_counter_unknown_schema_and_collection(self):
        for frame, status in (
            (self.counter().drop("pmu_running_time_us"), "unknown_schema"),
            (
                self.counter().with_columns(pl.lit("bad").alias("ts_uptime_us")),
                "unknown_schema",
            ),
            (self.counter(collection_id=["a", "b", "a"]), "unknown_collection"),
        ):
            features, _, qc = self.build({"dtlb_misses": frame})
            self.assert_values(
                features, "dtlb_misses", "system", "rate_per_s", [np.nan, np.nan]
            )
            self.assertEqual(qc["tables"]["dtlb_misses"]["status"], status)

    def test_counter_future_values_and_sparse_cpu_ids(self):
        frame = self.counter(cpu=[17, 17, 17])
        ticks = [250_001_000, 350_001_000]
        features, _, _ = self.build({"dtlb_misses": frame}, ticks=ticks)
        self.assert_values(features, "dtlb_misses", "system", "rate_per_s", [200, 200])
        self.assert_values(
            features, "dtlb_misses", "system", "coverage_cpu_fraction", [0.25, 0.25]
        )
        changed = self.counter(
            cpu=[17, 17, 17], cumulative_dtlb_misses=[100, 110, 2**32 - 1]
        )
        other, _, _ = self.build({"dtlb_misses": changed}, ticks=ticks)
        for key in features:
            np.testing.assert_equal(features[key][0], other[key][0])
        truncated, _, _ = self.build({"dtlb_misses": frame[:2]}, ticks=ticks)
        for key in features:
            np.testing.assert_equal(features[key][0], truncated[key][0])

    def test_future_entry_and_completion_values_do_not_change_earlier_features(self):
        fixtures = {
            "file_data": pl.DataFrame(
                {
                    "ts_uptime_us": [100_000, 1_000_001],
                    "tgid": [PID] * 2,
                    "file_size_bytes": [MIB, 2 * MIB],
                }
            ),
            "unmap_range": pl.DataFrame(
                {
                    "ts_ns": [100_000, NS + 1],
                    "tgid": [PID] * 2,
                    "start": [0, 0],
                    "end": [MIB, 2 * MIB],
                }
            ),
            "zswap_runtime": pl.DataFrame(
                {
                    "start_ts": [100_000, NS - 1],
                    "end_ts": [200_000, NS + 1],
                    "tgid": [PID] * 2,
                    "name": ["zswap_store"] * 2,
                }
            ),
            "trace_mm_khugepaged_scan_pmd": pl.DataFrame(
                {
                    "end_ts_ns": [100_000, NS + 1],
                    "tgid": [PID] * 2,
                    "none_or_zero": [1, 2],
                    "referenced": [2, 3],
                    "writeable": [0, 1],
                }
            ),
        }
        fields = {
            "file_data": ("ts_uptime_us", "file_size_bytes"),
            "unmap_range": ("ts_ns", "end"),
            "zswap_runtime": ("end_ts", "start_ts"),
            "trace_mm_khugepaged_scan_pmd": ("end_ts_ns", "referenced"),
        }
        features, _, _ = self.build(fixtures)
        changed = {}
        for name, frame in fixtures.items():
            timestamp, value = fields[name]
            changed[name] = frame.with_columns(
                pl.when(pl.col(timestamp) == frame[timestamp][-1])
                .then(999_999)
                .otherwise(pl.col(value))
                .alias(value)
            )
        other, _, _ = self.build(changed)
        prefix, _, _ = self.build({name: frame[:1] for name, frame in fixtures.items()})
        for key in features:
            np.testing.assert_equal(features[key][0], other[key][0])
            np.testing.assert_equal(features[key][0], prefix[key][0])

    def test_large_timestamps_preserve_exact_quantization_boundary(self):
        base_us = 2**53 // 1000 + 1
        frame = pl.DataFrame(
            {"ts_uptime_us": [base_us], "tgid": [PID], "quanta_run_length_us": [10]}
        )
        proof = base_us * 1000 + 1000
        features, _, _ = self.build({"quanta_runtime": frame}, ticks=[proof - 1, proof])
        self.assert_values(
            features, "quanta_runtime", "redis", "completed_count", [0, 1]
        )

    def memory(self, **changes):
        data = {
            "ts_uptime_us": [500_000, 1_500_000, 2_500_000],
            "mem_available_bytes": [10 * MIB, 20 * MIB, 900 * MIB],
            "swap_total_bytes": [30 * MIB] * 3,
            "swap_free_bytes": [20 * MIB, 10 * MIB, 0],
        }
        data.update(changes)
        return pl.DataFrame(data)

    def test_memory_conservative_prior_value_proof_and_phase_left_carry(self):
        ticks = [1_500_000_000, 1_500_001_000, 2_500_000_000, 2_500_001_000, 10 * NS]
        features, catalog, qc = self.build(
            {"memory_usage": self.memory()}, ticks=ticks, phase_start_ns=NS
        )
        self.assert_values(
            features,
            "memory_usage",
            "system",
            "available_known_prior_mib",
            [np.nan, 10, 10, 20, 20],
        )
        self.assert_values(
            features,
            "memory_usage",
            "system",
            "swap_used_known_prior_mib",
            [np.nan, 10, 10, 20, 20],
        )
        self.assert_values(
            features, "memory_usage", "system", "anon_known_prior_mib", [np.nan] * 5
        )
        self.assertTrue(qc["tables"]["memory_usage"]["physical_clock_lag_unknown"])
        self.assertIn(
            "clock_uncalibrated_conservative_availability", json.dumps(catalog)
        )
        # The next read's timestamp proves the prior read, not its own value.
        changed = self.memory(mem_available_bytes=[10 * MIB, 80 * MIB, 1234 * MIB])
        other, _, _ = self.build(
            {"memory_usage": changed}, ticks=ticks, phase_start_ns=NS
        )
        np.testing.assert_equal(
            features["f__memory_usage__system__available_known_prior_mib"][:3],
            other["f__memory_usage__system__available_known_prior_mib"][:3],
        )
        self.assert_values(
            other,
            "memory_usage",
            "system",
            "available_known_prior_mib",
            [np.nan, 10, 10, 80, 80],
        )
        # Proofs must be in phase, although the associated values may be older.
        features, _, _ = self.build(
            {"memory_usage": self.memory()},
            ticks=[2 * NS],
            phase_start_ns=1_600_000_000,
        )
        self.assert_values(
            features, "memory_usage", "system", "available_known_prior_mib", [np.nan]
        )

    def test_memory_different_boottime_offsets_never_assume_clock_equality(self):
        for offset in (0, 5_000_000, 50_000_000):
            with self.subTest(boottime_offset_us=offset):
                frame = self.memory(
                    ts_uptime_us=[
                        500_000 + offset,
                        1_500_000 + offset,
                        2_500_000 + offset,
                    ]
                )
                proof = (1_500_000 + offset) * 1000 + 1000
                features, _, _ = self.build(
                    {"memory_usage": frame}, ticks=[NS, proof - 1, proof, proof + NS]
                )
                self.assert_values(
                    features,
                    "memory_usage",
                    "system",
                    "available_known_prior_mib",
                    [np.nan, np.nan, 10, 20],
                )

    def test_memory_requires_two_strict_serial_timestamps_and_valid_values(self):
        for frame in (
            pl.DataFrame(),
            self.memory()[:1],
            self.memory().drop("ts_uptime_us"),
            self.memory(ts_uptime_us=[500_000, 500_000, 2_500_000]),
            self.memory(ts_uptime_us=[500_000, 2_500_000, 1_500_000]),
            self.memory(ts_uptime_us=[500_000, None, 2_500_000]),
        ):
            features, _, _ = self.build({"memory_usage": frame})
            self.assert_values(
                features,
                "memory_usage",
                "system",
                "available_known_prior_mib",
                [np.nan, np.nan],
            )
        frame = self.memory(
            mem_available_bytes=[-1, 20 * MIB, 0],
            swap_free_bytes=[40 * MIB, 10 * MIB, 0],
        )
        features, _, _ = self.build({"memory_usage": frame}, ticks=[2 * NS, 3 * NS])
        self.assert_values(
            features,
            "memory_usage",
            "system",
            "available_known_prior_mib",
            [np.nan, 20],
        )
        self.assert_values(
            features,
            "memory_usage",
            "system",
            "swap_used_known_prior_mib",
            [np.nan, 20],
        )

    def test_missing_values_do_not_erase_known_entry_counts(self):
        frame = pl.DataFrame({"ts_uptime_us": [100_000], "tgid": [PID]})
        features, _, qc = self.build({"file_data": frame})
        self.assert_values(features, "file_data", "redis", "observed_hit_count", [1, 0])
        self.assert_values(
            features, "file_data", "redis", "inode_size_mean_mib", [np.nan, np.nan]
        )
        self.assertIn(
            "file_size_bytes", qc["tables"]["file_data"]["unknown_value_columns"]
        )
        features, _, qc = self.build({"file_data": frame.drop("tgid")})
        self.assert_values(
            features, "file_data", "redis", "observed_hit_count", [np.nan, np.nan]
        )
        self.assertEqual(qc["tables"]["file_data"]["status"], "unknown_schema")

    def test_unsafe_and_target_tables_only_appear_as_exclusions(self):
        excluded = {
            name: pl.DataFrame({"future_status": [1]})
            for name in (
                "block_io",
                "madvise",
                "collapse_hugepage",
                "mm_rss_stat",
                "dtlb_accesses",
            )
        }
        features, catalog, qc = self.build(excluded)
        baseline, _, _ = self.build({})
        for key in features:
            np.testing.assert_equal(features[key], baseline[key])
        for name in excluded:
            self.assertEqual(qc["tables"][name]["status"], "excluded")
            self.assertTrue(qc["exclusions"][name])
            self.assertFalse(any(entry["group"] == name for entry in catalog))
        self.assertIn("Nonunique", qc["exclusions"]["block_io"])
        self.assertIn("success-only", qc["exclusions"]["madvise"])
        self.assertIn("future", qc["exclusions"]["collapse_hugepage"])

    def test_empty_grid_invalid_arguments_and_no_input_mutation(self):
        frame = self.counter()
        before = frame.clone()
        features, _, _ = self.build({"dtlb_misses": frame}, ticks=[])
        self.assertTrue(all(values.shape == (0,) for values in features.values()))
        self.assertTrue(frame.equals(before))
        for kwargs in (
            {"logical_cpus": 0},
            {"window_ns": 0},
            {"redis_pid": 0},
            {"phase_start_ns": -1},
            {"ticks": [0], "phase_start_ns": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build({}, **kwargs)
        with self.assertRaises(ValueError):
            build_features({}, PID, np.array([1.5]), phase_start_ns=0, logical_cpus=1)

    @unittest.skipUnless(
        os.environ.get("BLOAT_FEATURES_ARCHIVE_TEST") == "1",
        "opt-in read-only archive smoke test",
    )
    def test_archived_all_60_full_probe_collections(self):
        root = Path(__file__).resolve().parents[2]
        audit = root / "data/derived/bloat/full-probes-20260908/audit"
        runs, phases = (
            pl.read_csv(audit / "runs.csv"),
            pl.read_csv(audit / "phases.csv"),
        )
        self.assertEqual(runs.height, 60)
        self.assertEqual(runs.filter(pl.col("thp") == "always").height, 30)
        self.assertEqual(runs.filter(pl.col("thp") == "never").height, 30)
        self.assertEqual(phases.height, 120)
        for run in runs.iter_rows(named=True):
            directory = (root / run["collection_directory"]).resolve()
            self.assertTrue(directory.is_relative_to(root / "data/curated/redis"))
            tables = {}
            for path in sorted(directory.glob("*.parquet")):
                name = path.name.split(".")[0]
                frame = pl.read_parquet(path)
                tables[name] = (
                    pl.concat([tables[name], frame], how="diagonal_relaxed")
                    if name in tables
                    else frame
                )
            logical_cpus = tables["system_info"]["logical_cores"][0]
            for phase in phases.filter(
                pl.col("collection_id") == run["collection_id"]
            ).iter_rows(named=True):
                with self.subTest(
                    collection=run["collection_id"], phase=phase["phase"]
                ):
                    ticks = np.arange(
                        phase["start_ns"] + NS, phase["end_ns"] + 1, NS, dtype=np.int64
                    )
                    features, catalog, qc = self.build(
                        tables,
                        ticks=ticks,
                        redis_pid=run["redis_pid"],
                        phase_start_ns=phase["start_ns"],
                        logical_cpus=logical_cpus,
                    )
                    self.assertEqual(len(features), 56)
                    self.assertGreater(ticks.size, 0)
                    for values in features.values():
                        self.assertEqual(values.dtype, np.float64)
                        self.assertEqual(values.shape, ticks.shape)
                        self.assertFalse(np.isinf(values).any())
                    for name in (
                        "quanta_runtime",
                        "quanta_queued_time",
                        "file_data",
                        "unmap_range",
                        "memory_usage",
                        *COUNTERS,
                    ):
                        self.assertEqual(qc["tables"][name]["status"], "ok", name)
                    self.assertTrue(
                        np.isfinite(
                            features["f__quanta_runtime__redis__completed_count"]
                        ).all()
                    )
                    self.assertTrue(
                        np.isfinite(
                            features["f__dtlb_misses__system__rate_per_s"]
                        ).any()
                    )
                    json.dumps(catalog, allow_nan=False)
                    json.dumps(qc, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
