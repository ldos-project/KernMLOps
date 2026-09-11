"""Offline analysis contracts; only output guards use disposable /tmp archives."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import analyze
import audit
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

NS = 1_000_000_000
PID = 4242
RSS_PREFIX = "f__rss__redis__"
TARGETS = ("excess_now_mib", "excess_1s_mib", "excess_change_1s_mib")


def rss_frame(rows=()):
    return pl.DataFrame(
        rows,
        schema={"ts_ns": pl.Int64, "member": pl.String, "count": pl.Int64},
        orient="row",
    )


def window(
    collection_id, elapsed_s, observed, future=None, *, thp="never", phase="run"
):
    return {
        "collection_id": collection_id,
        "thp": thp,
        "phase": phase,
        "elapsed_s": elapsed_s,
        "observed_rss_mib": None if observed is None else float(observed),
        "observed_rss_1s_mib": None if future is None else float(future),
    }


def snapshot(directory):
    return {
        path.relative_to(directory): (
            path.read_bytes() if path.is_file() else None,
            path.lstat().st_mtime_ns,
            path.lstat().st_mode,
        )
        for path in directory.rglob("*")
    }


class BloatAnalysisTests(unittest.TestCase):
    def test_missing_rss_schema_or_component_stays_unknown(self):
        anon = rss_frame([(0, "MM_ANONPAGES", 256)])
        for rss in (pl.DataFrame(), rss_frame(), *(anon.drop(c) for c in anon.columns)):
            with self.subTest(schema=rss.schema, rows=rss.height):
                values, ages = analyze.rss_component(
                    rss, "MM_ANONPAGES", np.array([0, NS], dtype=np.int64)
                )
                self.assertTrue(np.isnan(values).all())
                self.assertTrue(np.isnan(ages).all())
        for member in ("MM_ANONPAGES", "MM_FILEPAGES"):
            with self.subTest(missing=member):
                rss = rss_frame(
                    [
                        (
                            0,
                            "MM_FILEPAGES"
                            if member == "MM_ANONPAGES"
                            else "MM_ANONPAGES",
                            256,
                        )
                    ]
                )
                _, _, features, future = analyze.rss_windows(rss, 0, 2 * NS)
                for stat in (
                    "observed_mib",
                    "observed_change_1s_mib",
                    "last_update_age_s",
                ):
                    self.assertTrue(np.isnan(features[RSS_PREFIX + stat]).all())
                missing = "anon_mib" if member == "MM_ANONPAGES" else "file_mib"
                known = "file_mib" if member == "MM_ANONPAGES" else "anon_mib"
                self.assertTrue(np.isnan(features[RSS_PREFIX + missing]).all())
                np.testing.assert_array_equal(features[RSS_PREFIX + known], [1, 1])
                self.assertTrue(np.isnan(future).all())

    def test_proxy_is_anon_plus_file_not_total_rss_or_swap(self):
        rss = rss_frame([(0, "MM_ANONPAGES", 256), (0, "MM_FILEPAGES", 512)])
        ticks, _, features, future = analyze.rss_windows(rss, 0, 2 * NS)
        shmem, shmem_age = analyze.rss_component(rss, "MM_SHMEMPAGES", ticks)
        self.assertTrue(np.isnan(shmem).all())
        self.assertTrue(np.isnan(shmem_age).all())
        np.testing.assert_array_equal(features[RSS_PREFIX + "observed_mib"], [3, 3])
        np.testing.assert_equal(future, [3, np.nan])
        self.assertEqual(analyze.TARGET_KIND, "observed_anonymous_plus_file_rss_excess")
        unrelated = rss_frame(
            [(0, "MM_SWAPENTS", 1_000_000), (0, "MM_SHMEMPAGES", 2_000_000)]
        )
        _, _, other, other_future = analyze.rss_windows(
            pl.concat([rss, unrelated]), 0, 2 * NS
        )
        self.assertEqual(features.keys(), other.keys())
        for name in features:
            np.testing.assert_equal(features[name], other[name])
        np.testing.assert_equal(future, other_future)
        self.assertFalse(any("shmem" in name or "swap" in name for name in features))

    def test_rss_lookup_sorts_without_backward_fill_and_keeps_ns_precision(self):
        for base in (0, 2**53 + 17):
            with self.subTest(base=base):
                rss = rss_frame(
                    [
                        (base + 2 * NS, "MM_ANONPAGES", 512),
                        (base + NS, "MM_ANONPAGES", 256),
                        (-1, "MM_ANONPAGES", 9999),
                        (None, "MM_ANONPAGES", 9999),
                        (base, "MM_FILEPAGES", 9999),
                    ]
                )
                before = rss.clone()
                ticks = base + np.array(
                    [NS - 1, NS, NS + 1, 2 * NS - 1, 2 * NS, 3 * NS], dtype=np.int64
                )
                values, ages = analyze.rss_component(rss, "MM_ANONPAGES", ticks)
                np.testing.assert_equal(values, [np.nan, 1, 1, 1, 2, 2])
                np.testing.assert_allclose(
                    ages, [np.nan, 0, 1e-9, 1 - 1e-9, 0, 1], rtol=0, atol=1e-15
                )
                assert_frame_equal(rss, before)

    def test_four_kib_pages_to_mib_zero_valid_negative_invalid(self):
        rss = rss_frame(
            [
                (i * NS, "MM_ANONPAGES", pages)
                for i, pages in enumerate((1, 256, 512, 0, -1))
            ]
        )
        values, ages = analyze.rss_component(
            rss, "MM_ANONPAGES", np.arange(5, dtype=np.int64) * NS
        )
        np.testing.assert_equal(values, [1 / 256, 1, 2, 0, np.nan])
        np.testing.assert_equal(ages, [0, 0, 0, 0, np.nan])

    def test_invalid_and_conflicting_observations_are_barriers_until_next_valid(self):
        for counts in ((-1,), (None,), (256, 512), (256, None), (-1, 256)):
            with self.subTest(counts=counts):
                rss = rss_frame(
                    [
                        (0, "MM_ANONPAGES", 256),
                        *[(NS, "MM_ANONPAGES", count) for count in counts],
                        (3 * NS, "MM_ANONPAGES", 512),
                    ]
                ).reverse()
                ticks = np.array(
                    [0, NS - 1, NS, 2 * NS, 3 * NS - 1, 3 * NS, 4 * NS], dtype=np.int64
                )
                values, ages = analyze.rss_component(rss, "MM_ANONPAGES", ticks)
                np.testing.assert_equal(values, [1, 1, np.nan, np.nan, np.nan, 2, 2])
                np.testing.assert_equal(
                    ages, [0, 1 - 1e-9, np.nan, np.nan, np.nan, 0, 1]
                )
                rss = pl.concat([rss, rss_frame([(0, "MM_FILEPAGES", 256)])])
                _, _, features, future = analyze.rss_windows(rss, 0, 4 * NS)
                np.testing.assert_equal(
                    features[RSS_PREFIX + "observed_mib"], [np.nan, np.nan, 3, 3]
                )
                np.testing.assert_equal(
                    features[RSS_PREFIX + "observed_change_1s_mib"],
                    [np.nan, np.nan, np.nan, 0],
                )
                np.testing.assert_equal(future, [np.nan, 3, 3, np.nan])

    def test_identical_duplicate_observations_are_one_valid_level(self):
        rss = rss_frame(
            [
                (0, "MM_ANONPAGES", 256),
                (NS, "MM_ANONPAGES", 256),
                (NS, "MM_ANONPAGES", 256),
                (2 * NS, "MM_ANONPAGES", 512),
            ]
        )
        values, ages = analyze.rss_component(
            rss, "MM_ANONPAGES", np.array([NS, 2 * NS - 1, 2 * NS], dtype=np.int64)
        )
        np.testing.assert_equal(values, [1, 1, 2])
        np.testing.assert_equal(ages, [0, 1 - 1e-9, 0])

    def test_rss_timestamps_and_counts_must_be_integers(self):
        rss = rss_frame([(0, "MM_ANONPAGES", 256)])
        for column in ("ts_ns", "count"):
            for dtype in (pl.Float64, pl.String):
                with self.subTest(column=column, dtype=dtype):
                    with self.assertRaisesRegex(ValueError, "must be integers"):
                        analyze.rss_component(
                            rss.with_columns(pl.col(column).cast(dtype)),
                            "MM_ANONPAGES",
                            np.array([NS], dtype=np.int64),
                        )

    def test_current_rss_and_past_changes_use_launcher_relative_one_second_grid(self):
        # The run grid begins at launch, not at an inferred post-Maven work start.
        start = 17 * NS + 123
        rss = rss_frame(
            [
                (start, "MM_ANONPAGES", 256),
                (start, "MM_FILEPAGES", 512),
                (start + NS // 2, "MM_ANONPAGES", 768),
                (start + NS + NS // 4, "MM_FILEPAGES", 128),
                (start + 2 * NS + NS // 2, "MM_ANONPAGES", 256),
            ]
        )
        ticks, elapsed, features, future = analyze.rss_windows(
            rss, start, start + 3 * NS + NS // 4
        )
        np.testing.assert_array_equal(ticks, start + np.array([1, 2, 3]) * NS)
        np.testing.assert_array_equal(elapsed, [1, 2, 3])
        self.assertEqual(ticks.dtype, np.int64)
        self.assertEqual(elapsed.dtype, np.int64)
        expected = {
            "observed_mib": [5, 3.5, 1.5],
            "anon_mib": [3, 3, 1],
            "file_mib": [2, 0.5, 0.5],
            "observed_change_1s_mib": [2, -1.5, -2],
            "anon_change_1s_mib": [2, 0, -2],
            "file_change_1s_mib": [0, -1.5, 0],
            "last_update_age_s": [0.5, 0.75, 0.5],
        }
        self.assertEqual(set(features), {RSS_PREFIX + name for name in expected})
        for name, values in expected.items():
            np.testing.assert_equal(features[RSS_PREFIX + name], values)
        np.testing.assert_equal(future, [3.5, 1.5, np.nan])

    def test_first_component_observations_cannot_fill_earlier_windows_or_changes(self):
        rss = rss_frame([(NS // 2, "MM_ANONPAGES", 256), (NS + 1, "MM_FILEPAGES", 512)])
        _, _, features, future = analyze.rss_windows(rss, 0, 3 * NS)
        np.testing.assert_equal(features[RSS_PREFIX + "observed_mib"], [np.nan, 3, 3])
        np.testing.assert_equal(
            features[RSS_PREFIX + "observed_change_1s_mib"], [np.nan, np.nan, 0]
        )
        np.testing.assert_equal(future, [3, 3, np.nan])
        for end in (0, NS - 1):
            ticks, elapsed, features, future = analyze.rss_windows(rss, 0, end)
            self.assertEqual(ticks.size, 0)
            self.assertEqual(elapsed.size, 0)
            self.assertEqual(future.size, 0)
            self.assertTrue(all(values.size == 0 for values in features.values()))

    def test_future_rss_and_event_mutations_cannot_change_any_feature_at_t(self):
        rss = rss_frame(
            [
                (0, "MM_ANONPAGES", 256),
                (0, "MM_FILEPAGES", 256),
                (NS + 1, "MM_ANONPAGES", 512),
            ]
        )
        unmap = pl.DataFrame(
            {
                "ts_ns": [NS // 2, NS + 1],
                "tgid": [PID] * 2,
                "start": [0, 0],
                "end": [2**20, 2**21],
            }
        )
        zswap = pl.DataFrame(
            {
                "start_ts": [NS // 4, NS - 1],
                "end_ts": [NS // 2, NS + 1],
                "tgid": [PID] * 2,
                "name": ["zswap_store"] * 2,
            }
        )
        frames = []
        for variant in ("original", "mutated_future", "past_only"):
            current_rss, current_unmap, current_zswap = rss, unmap, zswap
            if variant == "mutated_future":
                current_rss = rss.with_columns(
                    pl.when(pl.col("ts_ns") > NS)
                    .then(256_000)
                    .otherwise(pl.col("count"))
                    .alias("count")
                )
                current_unmap = unmap.with_columns(
                    pl.when(pl.col("ts_ns") > NS)
                    .then(2**30)
                    .otherwise(pl.col("end"))
                    .alias("end")
                )
                current_zswap = zswap.with_columns(
                    pl.when(pl.col("end_ts") > NS)
                    .then(0)
                    .otherwise(pl.col("start_ts"))
                    .alias("start_ts")
                )
            elif variant == "past_only":
                current_rss = rss.filter(pl.col("ts_ns") <= NS)
                current_unmap = unmap.filter(pl.col("ts_ns") <= NS)
                current_zswap = zswap.filter(pl.col("end_ts") <= NS)
            ticks, _, values, future = analyze.rss_windows(current_rss, 0, 2 * NS)
            probes, catalog, _ = analyze.build_features(
                {"unmap_range": current_unmap, "zswap_runtime": current_zswap},
                PID,
                ticks,
                phase_start_ns=0,
                logical_cpus=2,
            )
            self.assertFalse(set(values) & set(probes))
            self.assertEqual(set(probes), {entry["name"] for entry in catalog})
            frames.append(
                pl.DataFrame({**values, **probes, "observed_rss_1s_mib": future})
            )
        names = [name for name in frames[0].columns if name.startswith("f__")]
        for other in frames[1:]:
            assert_frame_equal(frames[0].select(names)[:1], other.select(names)[:1])
        self.assertEqual(frames[0]["observed_rss_1s_mib"][0], 3)
        self.assertEqual(frames[1]["observed_rss_1s_mib"][0], 1001)
        self.assertEqual(frames[2]["observed_rss_1s_mib"][0], 2)
        for name in (
            RSS_PREFIX + "observed_mib",
            "f__unmap_range__redis__requested_sum_mib",
            "f__zswap_runtime__redis__store_latency_mean_ms",
        ):
            self.assertNotEqual(frames[0][name][1], frames[1][name][1])

    def test_future_labels_at_phase_end_are_null_even_with_later_rss_and_reference(
        self,
    ):
        rows = []
        for phase, start, duration in (
            ("load", 0, 2 * NS),
            ("run", 10 * NS, 2 * NS + NS // 2),
        ):
            rss = rss_frame(
                [
                    (start, "MM_ANONPAGES", 256),
                    (start, "MM_FILEPAGES", 256),
                    (start + 2 * NS, "MM_ANONPAGES", 512),
                    (start + 3 * NS, "MM_ANONPAGES", 256_000),
                ]
            )
            ticks, elapsed, features, future = analyze.rss_windows(
                rss, start, start + duration
            )
            np.testing.assert_array_equal(ticks, start + np.array([1, 2]) * NS)
            np.testing.assert_equal(future, [3, np.nan])
            rows.extend(
                window("always", int(t), now, later, thp="always", phase=phase)
                for t, now, later in zip(
                    elapsed, features[RSS_PREFIX + "observed_mib"], future, strict=True
                )
            )
            rows.extend(
                window(f"never-{i}", t, 5, phase=phase)
                for i in range(15)
                for t in (1, 2, 3)
            )
        windows = pl.DataFrame(rows).with_columns(pl.col(pl.Float64).fill_nan(None))
        labeled, _ = analyze.add_reference(windows)
        for phase in ("load", "run"):
            always = labeled.filter(
                (pl.col("thp") == "always") & (pl.col("phase") == phase)
            )
            self.assertEqual(always["excess_1s_mib"].to_list(), [-2, None])
            self.assertEqual(always["observed_rss_1s_mib"].to_list(), [3, None])
            self.assertEqual(always["reference_1s_mib"].to_list(), [5, 5])
            self.assertIsNone(always["excess_change_1s_mib"][-1])

    def test_reference_is_never_only_median_iqr_with_default_15_distinct_runs(self):
        for support in (14, 15, 16):
            with self.subTest(support=support):
                levels = [*range(1, support), 1000]
                rows = [
                    window(f"never-{i}", 1, value) for i, value in enumerate(levels)
                ]
                rows.extend(
                    window(f"invalid-{i}", 1, value)
                    for i, value in enumerate((None, np.nan, np.inf, -np.inf))
                )
                rows.extend(
                    window(f"always-{i}", 1, 10_000, thp="always") for i in range(30)
                )
                rows.append(window("other-policy", 1, 1_000_000, thp="madvise"))
                windows = pl.DataFrame(rows).with_columns(pl.lit(0).alias("replicate"))
                before = windows.clone()
                labeled, reference = analyze.add_reference(windows)
                self.assertEqual(reference.height, 1)
                row = reference.row(0, named=True)
                self.assertEqual(row["reference_n"], support)
                self.assertEqual((row["phase"], row["elapsed_s"]), ("run", 1))
                for name, value in (
                    ("reference_mib", np.median(levels)),
                    ("reference_q25_mib", np.quantile(levels, 0.25)),
                    ("reference_q75_mib", np.quantile(levels, 0.75)),
                ):
                    self.assertEqual(row[name], value if support >= 15 else None)
                always = labeled.filter(pl.col("thp") == "always")
                self.assertEqual(
                    always["excess_now_mib"].to_list(),
                    [10_000 - np.median(levels) if support >= 15 else None] * 30,
                )
                others = labeled.filter(pl.col("thp") != "always")
                self.assertTrue(
                    all(others[name].null_count() == others.height for name in TARGETS)
                )
                assert_frame_equal(windows, before)

    def test_one_second_reference_matching_keeps_negative_excess_and_time_gaps(self):
        rows = [
            window(f"never-{i}", t, baseline + i - 7, 999_999)
            for i in range(15)
            for t, baseline in ((1, 100), (2, 120), (4, 400))
        ]
        rows.extend(
            window(cid, t, now, future, thp="always")
            for cid, t, now, future in (
                ("a", 1, 90, 105),
                ("a", 2, 105, 150),
                ("a", 3, 150, 390),
                ("a", 4, 390, None),
                ("b", 1, 110, 140),
            )
        )
        labeled, _ = analyze.add_reference(pl.DataFrame(rows).reverse())
        always = labeled.filter(pl.col("thp") == "always")
        self.assertEqual(
            always.select("collection_id", "elapsed_s").rows(),
            [("a", 1), ("a", 2), ("a", 3), ("a", 4), ("b", 1)],
        )
        self.assertEqual(
            always["reference_1s_mib"].to_list(), [120, None, 400, None, 120]
        )
        self.assertEqual(always["reference_1s_n"].to_list(), [15, None, 15, None, 15])
        self.assertEqual(
            always.select(TARGETS).rows(),
            [
                (-10, -15, -5),
                (-15, None, None),
                (None, -10, None),
                (-10, None, None),
                (10, 20, 10),
            ],
        )

    def test_reference_support_cannot_be_borrowed_from_other_phases_or_times(self):
        rows = [
            window(f"never-{i}", t, value, phase=phase)
            for phase, t, value, support in (
                ("run", 1, 10, 14),
                ("run", 2, 20, 15),
                ("load", 1, 100, 15),
            )
            for i in range(support)
        ]
        rows.extend(
            window("always", t, 50, 60, thp="always", phase=phase)
            for phase, t in (("load", 1), ("run", 1), ("run", 3))
        )
        labeled, _ = analyze.add_reference(pl.DataFrame(rows))
        always = labeled.filter(pl.col("thp") == "always")
        self.assertEqual(always["reference_n"].to_list(), [15, 14, None])
        self.assertEqual(always["reference_mib"].to_list(), [100, None, None])
        self.assertEqual(
            always.select(TARGETS).rows(),
            [(-50, None, None), (None, 40, None), (None, None, None)],
        )
        no_never, reference = analyze.add_reference(
            pl.DataFrame([window("a", 1, 10, 20, thp="always")])
        )
        self.assertTrue(reference.is_empty())
        self.assertEqual(no_never.select(TARGETS).row(0), (None, None, None))

    def test_reference_joins_preserve_phase_collection_window_identity_and_features(
        self,
    ):
        rows = []
        for phase, baseline in (("load", 1000), ("run", 100)):
            for t in (1, 2):
                rows.extend(
                    window(f"never-{i}", t, baseline + 10 * t + i - 7, phase=phase)
                    for i in range(15)
                )
                rows.extend(
                    window(cid, t, value, value + 5, thp="always", phase=phase)
                    for cid, value in (("a", 50), ("b", 60))
                )
        windows = (
            pl.DataFrame(rows)
            .with_row_index("row_id")
            .with_columns(
                pl.lit(0).alias("replicate"),
                pl.lit(PID).alias("redis_pid"),
                ((1000 - pl.col("row_id").cast(pl.Int64)) * NS).alias("time_ns"),
                pl.col("row_id").cast(pl.Float64).alias("f__fixture__redis__past"),
                pl.col("observed_rss_mib").alias(RSS_PREFIX + "observed_mib"),
                pl.lit(analyze.TARGET_KIND).alias("target_kind"),
            )
            .reverse()
        )
        before = windows.clone()
        labeled, reference = analyze.add_reference(windows)
        keys = ["collection_id", "phase", "elapsed_s"]
        self.assertEqual(labeled.height, windows.height)
        assert_frame_equal(labeled.select(windows.columns), windows.sort(keys))
        self.assertEqual(
            reference.select("phase", "elapsed_s", "reference_mib").rows(),
            [("load", 1, 1010), ("load", 2, 1020), ("run", 1, 110), ("run", 2, 120)],
        )
        for row in labeled.filter(pl.col("thp") == "always").iter_rows(named=True):
            baseline = (1000 if row["phase"] == "load" else 100) + 10 * row["elapsed_s"]
            self.assertEqual(row["excess_now_mib"], row["observed_rss_mib"] - baseline)
            self.assertEqual(
                row["reference_1s_mib"],
                baseline + 10 if row["elapsed_s"] == 1 else None,
            )
        changed = windows.with_columns(
            pl.when(pl.col("thp") == "always")
            .then(9999.0)
            .otherwise(pl.col("observed_rss_mib"))
            .alias("observed_rss_mib"),
            pl.lit(8888.0).alias("observed_rss_1s_mib"),
        )
        other, other_reference = analyze.add_reference(changed)
        assert_frame_equal(reference, other_reference)
        names = [name for name in labeled.columns if name.startswith("f__")]
        self.assertEqual(
            set(names), {name for name in windows.columns if name.startswith("f__")}
        )
        assert_frame_equal(labeled.select(*keys, *names), other.select(*keys, *names))
        self.assertFalse(labeled.select(TARGETS).equals(other.select(TARGETS)))
        assert_frame_equal(windows, before)

    def test_duplicate_windows_rejected_not_deduplicated_or_counted_as_runs(self):
        windows = pl.DataFrame(
            [
                *[window(f"never-{i}", 1, i) for i in range(15)],
                window("always", 1, 50, 60, thp="always"),
            ]
        ).with_columns(pl.lit(0).alias("replicate"), pl.lit(NS).alias("time_ns"))
        for thp in ("never", "always"):
            original = windows.filter(pl.col("thp") == thp)[:1]
            for conflicting in (False, True):
                with self.subTest(thp=thp, conflicting=conflicting):
                    duplicate = original
                    if conflicting:
                        duplicate = duplicate.with_columns(
                            pl.lit(9999.0).alias("observed_rss_mib"),
                            pl.lit("always" if thp == "never" else "never").alias(
                                "thp"
                            ),
                            pl.lit(99).alias("replicate"),
                            pl.lit(2 * NS).alias("time_ns"),
                        )
                    with self.assertRaisesRegex(
                        ValueError, "duplicate collection/phase/time windows"
                    ):
                        analyze.add_reference(
                            pl.concat([windows, duplicate], how="vertical_relaxed")
                        )

    def test_reference_minimum_outside_one_to_thirty_is_rejected(self):
        for minimum in (0, 31):
            with (
                self.subTest(minimum=minimum),
                self.assertRaisesRegex(ValueError, "between 1 and 30"),
            ):
                analyze.add_reference(pl.DataFrame(), min_reference_runs=minimum)

    def test_never_reference_rows_never_become_correlation_targets_or_runs(self):
        rows = [
            window(f"never-{i}", t, 100 + t + i - 7)
            for i in range(15)
            for t in range(1, 13)
        ]
        rows.extend(
            window(
                cid,
                t,
                100 + t + sign * np.sin(t),
                101 + t + sign * np.sin(t + 1) if t < 12 else None,
                thp="always",
            )
            for cid, sign in (("a", 1), ("b", -1))
            for t in range(1, 13)
        )
        windows = pl.DataFrame(
            rows, schema_overrides={"observed_rss_1s_mib": pl.Float64}
        ).with_columns(pl.col("elapsed_s").sin().alias("f__fixture__redis__past"))
        labeled, _ = analyze.add_reference(windows)
        never = labeled.filter(pl.col("thp") == "never")
        self.assertTrue(
            all(never[name].null_count() == never.height for name in TARGETS)
        )
        before = labeled.clone()
        expected = analyze.summarize_correlations(
            labeled.filter(pl.col("thp") == "always"), [], bootstrap_samples=11
        )
        poisoned = labeled.with_columns(
            *[
                pl.when(pl.col("thp") == "never")
                .then(-9999.0 * pl.col("elapsed_s"))
                .otherwise(pl.col(name))
                .alias(name)
                for name in TARGETS
            ]
        )
        for candidate in (labeled, poisoned):
            summary, per_run = analyze.summarize_correlations(
                candidate, [], bootstrap_samples=11
            )
            assert_frame_equal(summary, expected[0])
            assert_frame_equal(per_run, expected[1])
            self.assertEqual(set(per_run["collection_id"]), {"a", "b"})
            self.assertEqual(set(per_run["target"]), set(TARGETS))
            self.assertEqual(summary["n_candidate_runs"].unique().to_list(), [2])
            raw = per_run.filter(
                (pl.col("target") == "excess_now_mib") & (pl.col("adjustment") == "raw")
            )
            self.assertEqual(raw["status"].unique().to_list(), ["ok"])
            self.assertEqual(raw["n_points"].unique().to_list(), [12])
        assert_frame_equal(labeled, before)


class BloatAnalysisOutputGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="bloat-analysis-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "KernMLOps"
        self.root.mkdir()
        for module in (audit, analyze):
            patcher = mock.patch.object(module, "ROOT", self.root)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.experiment = self.root / "data/experiments/archive"
        self.raw = self.root / "data/curated/redis/archive"
        self.derived = self.root / "data/derived/bloat"
        self.audit_directory = self.derived / "fixture/audit"
        self.output = self.derived / "fixture/correlations"
        for directory, name in (
            (self.experiment, "plan.json"),
            (self.raw, "capture.bin"),
            (self.audit_directory, "audit.json"),
        ):
            directory.mkdir(parents=True)
            source = directory / name
            source.write_bytes(b"read-only archive sentinel\n")
            source.chmod(0o444)
        lock = self.experiment / ".lock"
        lock.write_bytes(b"existing experiment lock\n")
        lock.chmod(0o444)
        for directory in (self.experiment, self.raw, self.audit_directory):
            directory.chmod(0o555)
            self.addCleanup(directory.chmod, 0o755)

    def test_output_must_be_inside_separate_derived_bloat_tree(self):
        outside = self.root.parent / "outside"
        outside.mkdir()
        (self.derived / "raw-alias").symlink_to(self.raw, target_is_directory=True)
        (self.derived / "outside-alias").symlink_to(outside, target_is_directory=True)
        before = snapshot(self.root.parent)
        with (
            mock.patch.object(
                analyze,
                "full_runs",
                side_effect=AssertionError("archive analysis must not start"),
            ) as full,
            mock.patch.object(analyze, "plot_results") as plot,
        ):
            for output in (
                self.root / "new-analysis",
                self.experiment / "analysis",
                self.raw / "analysis",
                self.audit_directory,
                self.derived,
                self.root / "data/derived/bloat-other/analysis",
                self.derived / "../analysis",
                self.derived / "raw-alias/analysis",
                self.derived / "outside-alias/analysis",
                outside / "analysis",
                "/KernMLOps/data/derived/bloat/../../curated/analysis",
            ):
                with (
                    self.subTest(output=output),
                    self.assertRaisesRegex(ValueError, "outside|new output directory"),
                ):
                    analyze.analyze(self.experiment, self.audit_directory, output)
            full.assert_not_called()
            plot.assert_not_called()
        self.assertEqual(snapshot(self.root.parent), before)

    def test_existing_empty_nonempty_file_and_symlink_outputs_are_never_rewritten(self):
        empty = self.output.with_name("empty")
        empty.mkdir()
        self.output.mkdir()
        (self.output / "manifest.json").write_bytes(b"existing manifest\n")
        (self.output / "windows.parquet").write_bytes(b"existing windows sentinel\n")
        file = self.output.with_name("existing-file")
        file.write_bytes(b"existing output file\n")
        alias = self.output.with_name("existing-alias")
        alias.symlink_to(self.output, target_is_directory=True)
        before = snapshot(self.root.parent)
        with (
            mock.patch.object(analyze, "full_runs") as full,
            mock.patch.object(analyze, "plot_results") as plot,
        ):
            for output in (empty, self.output, file, alias):
                with (
                    self.subTest(output=output),
                    self.assertRaisesRegex(ValueError, "new output directory"),
                ):
                    analyze.analyze(self.experiment, self.audit_directory, output)
            full.assert_not_called()
            plot.assert_not_called()
        self.assertEqual(snapshot(self.root.parent), before)

    def test_safe_rebased_outputs_pass_guard_without_writing_or_analyzing_archives(
        self,
    ):
        before = snapshot(self.root.parent)
        stop = RuntimeError("stop at archive-read boundary")
        for style in ("absolute", "relative", "container"):
            paths = [self.experiment, self.audit_directory, self.output]
            if style != "absolute":
                paths = [path.relative_to(self.root) for path in paths]
            if style == "container":
                paths = [Path("/KernMLOps") / path for path in paths]
            with (
                self.subTest(style=style),
                mock.patch.object(analyze, "full_runs", side_effect=stop) as full,
                mock.patch.object(analyze, "plot_results") as plot,
                mock.patch.object(analyze.tempfile, "TemporaryDirectory") as stage,
            ):
                with self.assertRaises(RuntimeError) as caught:
                    analyze.analyze(*paths)
                self.assertIs(caught.exception, stop)
                full.assert_called_once_with(self.experiment)
                plot.assert_not_called()
                stage.assert_not_called()
            self.assertFalse(self.output.exists())
            self.assertEqual(snapshot(self.root.parent), before)

    def test_combined_or_unknown_phase_is_rejected_before_archive_analysis(self):
        before = snapshot(self.root.parent)
        with mock.patch.object(analyze, "full_runs") as full:
            for phase in ("all", "load,run", "unknown"):
                with (
                    self.subTest(phase=phase),
                    self.assertRaisesRegex(
                        ValueError, "phases must be analyzed separately"
                    ),
                ):
                    analyze.analyze(
                        self.experiment, self.audit_directory, self.output, phase=phase
                    )
            full.assert_not_called()
        self.assertEqual(snapshot(self.root.parent), before)


if __name__ == "__main__":
    unittest.main()
