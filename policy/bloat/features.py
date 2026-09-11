"""Offline, target-independent features from audited historical full-probe tables.

The caller owns collection/PID validation, phase grids, RSS labels and statistics.
No collector/schema imports, I/O, clock-offset estimation or future-value filling
are performed here. System features include Redis unless named system_other.
"""

import numpy as np
import polars as pl

_COUNTERS = ("dtlb_misses", "itlb_misses", "tlb_flushes", "dtlb_walk_duration")
_EXCLUSIONS = {
    "block_io": (
        "Nonunique historical join keys cannot establish "
        "matched completion availability."
    ),
    "madvise": (
        "Start timestamp with success-only emission and no end timestamp "
        "leaks future success."
    ),
    "collapse_hugepage": (
        "Merged status can come from a future or unmatched collapse; "
        "no safe completion timestamp."
    ),
    "mm_rss_stat": (
        "RSS components, labels and target-linked predictors belong to the caller, "
        "not this feature set."
    ),
    "dtlb_accesses": (
        "Not captured historically; no DTLB miss/access ratio is available."
    ),
}
_CAPTURE = "capture_completeness_unknown; observed counts are not loss-corrected"
_MEMORY_CLOCK = "clock_uncalibrated_conservative_availability"


def _integers(frame: pl.DataFrame, column: str) -> np.ndarray:
    """Keep nanosecond timestamps/addresses exact; -1 denotes an invalid integer."""
    if column not in frame.columns or not frame.schema[column].is_integer():
        return np.full(frame.height, -1, dtype=np.int64)
    return frame[column].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()


def _numbers(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns or not (
        frame.schema[column].is_numeric() or frame.schema[column] == pl.Boolean
    ):
        return np.full(frame.height, np.nan)
    return frame[column].cast(pl.Float64).to_numpy()


def _timestamps(frame: pl.DataFrame, column: str) -> np.ndarray:
    values = _integers(frame, column)
    if column == "ts_uptime_us":
        valid = (values >= 0) & (values <= (np.iinfo(np.int64).max - 1000) // 1000)
        return np.where(valid, np.where(valid, values, 0) * 1000, -1)
    return values


def build_features(
    tables: dict[str, pl.DataFrame],
    redis_pid: int,
    ticks_ns: np.ndarray,
    *,
    phase_start_ns: int,
    logical_cpus: int,
    window_ns: int = 1_000_000_000,
) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    """Return 56 float64 feature arrays, their catalog, and per-table QC.

    Events use (t-window_ns, t], restricted to this phase. Microsecond BPF
    timestamps get a 1000 ns availability guard. Durations are completion-binned
    (not clipped time occupancy), and sums/means without events remain NaN.
    Counters use consecutive per-CPU endpoints wholly inside the window, with
    the right endpoint's quantization guard also elapsed. Only memory gauges
    may carry older context: value_i needs timestamp_(i+1) as a serial-read proof.
    A present, zero-row event table permits observed zero counts, not proof of
    lossless capture. Nonempty metadata-only tables are unknown, not empty.
    """
    ticks = np.asarray(ticks_ns)
    for name, value, minimum in (
        ("redis_pid", redis_pid, 1),
        ("logical_cpus", logical_cpus, 1),
        ("window_ns", window_ns, 1),
        ("phase_start_ns", phase_start_ns, 0),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= value <= np.iinfo(np.int64).max
        ):
            raise ValueError(f"{name} must be an integer >= {minimum} within int64")
    if (
        ticks.ndim != 1
        or ticks.dtype.kind not in "iu"
        or np.any(ticks > np.iinfo(np.int64).max)
    ):
        raise ValueError("ticks_ns must be a one-dimensional int64-compatible array")
    ticks = ticks.astype(np.int64, copy=False)
    if np.any(ticks < phase_start_ns):
        raise ValueError("ticks_ns must not precede phase_start_ns")
    left = ticks - int(window_ns)
    features, catalog = {}, []
    qc = {
        "phase_start_ns": int(phase_start_ns),
        "window_ns": int(window_ns),
        "logical_cpus": int(logical_cpus),
        "tables": {},
        "exclusions": dict(_EXCLUSIONS),
        "notes": [
            "Input collection identity, Redis PID and invocation boundaries "
            "are caller-audited.",
            "No RSS labels, target-linked features, DTLB access ratio, "
            "interpolation or future-value fill.",
            "All event zeros mean observed_zero with capture_completeness_unknown.",
            "System scopes include Redis; executing task and memory-owner scopes "
            "are distinguished in the catalog.",
        ],
    }

    def prepare(name, required, optional=(), text=()):
        frame = tables.get(name)
        info = {
            "status": "missing_table",
            "rows": None,
            "capture_completeness": "unknown",
            "missing_columns": [],
            "invalid_columns": [],
        }
        qc["tables"][name] = info
        if frame is None:
            return pl.DataFrame()
        if not isinstance(frame, pl.DataFrame):
            info["status"] = "invalid_shape"
            return pl.DataFrame()
        info["rows"] = frame.height
        info["missing_columns"] = [
            c for c in (*required, *text) if c not in frame.columns
        ]
        info["invalid_columns"] = [
            c
            for c in required
            if c in frame.columns and not frame.schema[c].is_integer()
        ] + [c for c in text if c in frame.columns and frame.schema[c] != pl.String]
        info["unknown_value_columns"] = [
            c
            for c in optional
            if c not in frame.columns
            or not (frame.schema[c].is_numeric() or frame.schema[c] == pl.Boolean)
        ]
        if not frame.height:
            info["status"] = "empty"
            info["zero_semantics"] = (
                "observed_zero; capture_completeness_unknown; event counts only"
            )
        elif info["missing_columns"] or info["invalid_columns"]:
            info["status"] = "unknown_schema"
        else:
            info["status"] = "ok"
        return frame

    def add(group, scope, stat, unit, description, availability, limitations):
        name = f"f__{group}__{scope}__{stat}"
        values = np.full(ticks.size, np.nan, dtype=np.float64)
        features[name] = values
        catalog.append(
            {
                "name": name,
                "group": group,
                "scope": scope,
                "unit": unit,
                "description": description,
                "target_linked": False,
                "availability": availability,
                "limitations": list(limitations),
            }
        )
        return values

    def event_stats(
        table, group, scope, times, valid, metrics, availability, limitations
    ):
        """One sorted trailing-window selection shared by a small set of reductions."""
        arrays = [
            add(
                group,
                scope,
                stat,
                unit,
                description,
                availability
                + (
                    "; observed_zero when no valid events"
                    if op == "count"
                    else "; no event/value => NaN"
                ),
                [_CAPTURE, *limitations],
            )
            for stat, unit, description, _, op in metrics
        ]
        if qc["tables"][table]["status"] not in ("ok", "empty"):
            return
        order = np.flatnonzero(valid & (times >= phase_start_ns))
        order = order[np.argsort(times[order], kind="stable")]
        lo = np.searchsorted(times[order], left, side="right")
        hi = np.searchsorted(times[order], ticks, side="right")
        for array, (_, _, _, values, op) in zip(arrays, metrics, strict=True):
            if op == "count":
                array[:] = hi - lo
                continue
            ordered = values[order]
            for i, (start, end) in enumerate(zip(lo, hi, strict=True)):
                sample = ordered[start:end]
                sample = sample[np.isfinite(sample)]
                if sample.size:
                    if op == "sum":
                        array[i] = sample.sum()
                    elif op == "mean":
                        array[i] = sample.mean()
                    else:
                        array[i] = np.quantile(sample, 0.95, method="linear")

    for table, duration_column in (
        ("quanta_runtime", "quanta_run_length_us"),
        ("quanta_queued_time", "quanta_queued_time_us"),
    ):
        frame = prepare(table, ("ts_uptime_us", "tgid", duration_column))
        raw = _timestamps(frame, "ts_uptime_us")
        duration = _numbers(frame, duration_column)
        tgid = _integers(frame, "tgid")
        valid = (raw >= 0) & (tgid >= 0) & np.isfinite(duration) & (duration >= 0)
        valid &= duration <= np.maximum(raw, 0) / 1000
        qc["tables"][table]["rejected_rows"] = int((~valid).sum())
        valid &= raw >= phase_start_ns
        metrics = [
            (
                "completed_count",
                "events",
                "Observed completed scheduler quanta.",
                None,
                "count",
            ),
            (
                "duration_sum_s",
                "s",
                "Sum of durations of completed quanta.",
                duration / 1e6,
                "sum",
            ),
            (
                "duration_mean_ms",
                "ms",
                "Mean completed quantum duration.",
                duration / 1000,
                "mean",
            ),
            (
                "duration_p95_ms",
                "ms",
                "Linear p95 completed quantum duration.",
                duration / 1000,
                "p95",
            ),
        ]
        for scope, mask in (
            ("redis", tgid == redis_pid),
            ("system_other", tgid != redis_pid),
        ):
            event_stats(
                table,
                table,
                scope,
                raw + 1000,
                valid & mask,
                metrics if scope == "redis" else metrics[:2],
                "MONOTONIC completion ts_uptime_us*1000 + 1000 ns "
                "in the phase trailing window",
                [
                    "Full TGID filter includes Redis worker threads; "
                    "system_other excludes Redis.",
                    "Completion-binned sums are NOT utilization; a completed duration "
                    "may extend before the window/phase.",
                    "No existing filtered-view duration cutoffs; "
                    "recorded u32 durations may be truncated.",
                ],
            )

    table = "file_data"
    frame = prepare(table, ("ts_uptime_us", "tgid"), ("file_size_bytes",))
    raw, tgid = _timestamps(frame, "ts_uptime_us"), _integers(frame, "tgid")
    valid = (raw >= 0) & (tgid >= 0)
    qc["tables"][table]["rejected_rows"] = int((~valid).sum())
    size = _numbers(frame, "file_size_bytes")
    size = np.where(size >= 0, size / 2**20, np.nan)
    for scope, mask in (("redis", tgid == redis_pid), ("system", tgid >= 0)):
        event_stats(
            table,
            table,
            scope,
            raw + 1000,
            valid & mask & (raw >= phase_start_ns),
            [
                (
                    "observed_hit_count",
                    "events",
                    "Observed file entry-probe hits.",
                    None,
                    "count",
                ),
                (
                    "inode_size_mean_mib",
                    "MiB",
                    "Event-weighted mean observed inode size.",
                    size,
                    "mean",
                ),
            ],
            "MONOTONIC entry ts_uptime_us*1000 + 1000 ns in the phase trailing window",
            [
                "TGID is the executing process; system includes Redis.",
                "Entry hits, not successful I/O or transferred bytes; "
                "repeated hits are not unique inodes.",
                "Historical probes suppress some zero-size opens; "
                "inode size is not I/O volume.",
            ],
        )

    table = "unmap_range"
    frame = prepare(table, ("ts_ns", "tgid", "start", "end"))
    times, tgid = _timestamps(frame, "ts_ns"), _integers(frame, "tgid")
    start, end = _integers(frame, "start"), _integers(frame, "end")
    valid = (times >= 0) & (tgid >= 0) & (start >= 0) & (end >= start)
    qc["tables"][table]["rejected_rows"] = int((~valid).sum())
    requested = (np.maximum(end, 0) - np.maximum(start, 0)).astype(np.float64) / 2**20
    event_stats(
        table,
        table,
        "redis",
        times,
        valid & (tgid == redis_pid),
        [
            (
                "observed_request_count",
                "events",
                "Observed unmap entry requests.",
                None,
                "count",
            ),
            (
                "requested_sum_mib",
                "MiB",
                "Sum of requested end-start address-range lengths.",
                requested,
                "sum",
            ),
        ],
        "MONOTONIC entry ts_ns in the phase trailing window",
        [
            "Executing TGID equals Redis; range ends must be >= starts.",
            "Requested ranges can overlap or repeat; "
            "not actual freed bytes, THP freed, or completion results.",
        ],
    )

    table = "zswap_runtime"
    frame = prepare(table, ("start_ts", "end_ts", "tgid"), text=("name",))
    start, end = _timestamps(frame, "start_ts"), _timestamps(frame, "end_ts")
    tgid = _integers(frame, "tgid")
    names = (
        frame["name"].to_numpy()
        if "name" in frame.columns and frame.schema["name"] == pl.String
        else np.full(frame.height, "")
    )
    valid = (start >= 0) & (end >= start) & (tgid >= 0)
    valid &= np.isin(names, ["zswap_store", "zswap_load", "zswap_invalidate"])
    qc["tables"][table]["rejected_rows"] = int((~valid).sum())
    duration_ms = (np.maximum(end, 0) - np.maximum(start, 0)).astype(np.float64) / 1e6
    for operation in ("store", "load", "invalidate"):
        for scope, mask in (("redis", tgid == redis_pid), ("system", tgid >= 0)):
            event_stats(
                table,
                table,
                scope,
                end,
                valid & mask & (names == f"zswap_{operation}"),
                [
                    (
                        f"{operation}_completed_count",
                        "events",
                        f"Observed completed zswap {operation} calls.",
                        None,
                        "count",
                    ),
                    (
                        f"{operation}_latency_mean_ms",
                        "ms",
                        f"Mean completed zswap {operation} latency.",
                        duration_ms,
                        "mean",
                    ),
                ],
                "MONOTONIC end_ts in the phase trailing window; start_ts <= end_ts",
                [
                    "TGID is the executing process, not ownership of swapped pages; "
                    "system includes Redis.",
                    "Completed calls are not necessarily successful; "
                    "no bytes, compression ratio or live operations inferred.",
                    "Completion-binned latency can extend before the window/phase.",
                ],
            )

    table = "trace_mm_khugepaged_scan_pmd"
    frame = prepare(
        table, ("end_ts_ns", "tgid"), ("none_or_zero", "referenced", "writeable")
    )
    times, tgid = _timestamps(frame, "end_ts_ns"), _integers(frame, "tgid")
    valid = (times >= 0) & (tgid >= 0)
    if "start_ts_ns" in frame.columns:
        start = _timestamps(frame, "start_ts_ns")
        valid &= (start >= 0) & (start <= times)
    qc["tables"][table]["rejected_rows"] = int((~valid).sum())
    metrics = [
        (
            "observed_count",
            "events",
            "Observed scan-PMD tracepoint records.",
            None,
            "count",
        )
    ]
    for column in ("none_or_zero", "referenced", "writeable"):
        values = _numbers(frame, column)
        values = np.where(np.isfinite(values) & (values >= 0), values, np.nan)
        if column == "writeable":
            values = np.where(np.isfinite(values), (values > 0).astype(float), np.nan)
        metrics.append(
            (
                "writeable_fraction" if column == "writeable" else f"{column}_mean",
                "fraction" if column == "writeable" else "pages",
                f"Scan-PMD {column}: "
                + (
                    "nonzero event fraction."
                    if column == "writeable"
                    else "event-weighted mean."
                ),
                values,
                "mean",
            )
        )
    for scope, mask in (("redis", tgid == redis_pid), ("system", tgid >= 0)):
        event_stats(
            table,
            "scan_pmd",
            scope,
            times,
            valid & mask,
            metrics if scope == "redis" else metrics[:1],
            "MONOTONIC end_ts_ns of tracepoint handler in the phase trailing window",
            [
                "TGID is mm->owner, not the executing khugepaged task; "
                "system includes Redis-owned mm records.",
                "Handler end proves field availability, NOT actual scan runtime; "
                "no duration feature.",
                "Historical unmapped was reduced to boolean and is ignored; "
                "no collapse result inferred.",
            ],
        )

    for table in _COUNTERS:
        column = f"cumulative_{table}"
        frame = prepare(
            table,
            (
                "cpu",
                "ts_uptime_us",
                column,
                "pmu_enabled_time_us",
                "pmu_running_time_us",
            ),
            text=("collection_id",),
        )
        info = qc["tables"][table]
        info["max_gap_ns"] = 250_000_000
        info["counter_bits"] = 32
        info["pid_filter_applied"] = False
        availability = (
            "Consecutive same-collection/CPU MONOTONIC endpoints both in (t-window,t] "
            "and >= phase_start; end + 1000 ns <= t"
        )
        limitations = [
            _CAPTURE,
            "System CPU counters, never filtered by PID/TGID and not Redis-specific.",
            "Negative deltas are ambiguous wrap/reset and discarded; "
            "no modular reconstruction. Short gaps cannot rule out hidden u32 wraps.",
            "Only gaps <= 0.25 s, positive enabled/running deltas "
            "and running <= enabled are usable.",
            "Rates sum per-CPU scaled-count/covered-second rates over observed CPUs "
            "only; no extrapolation to missing CPUs.",
            "dtlb_walk_duration uses hardware counter units, "
            "not milliseconds or a DTLB miss/access ratio.",
        ]
        rate = add(
            table,
            "system",
            "rate_per_s",
            "counter_units/s",
            "Sum of per-observed-CPU rates using deltaC * deltaE / deltaR.",
            availability,
            limitations,
        )
        coverage = add(
            table,
            "system",
            "coverage_cpu_fraction",
            "fraction",
            "CPUs with usable intervals / logical_cpus; no intervals => NaN.",
            availability,
            limitations,
        )
        covered_fraction = add(
            table,
            "system",
            "mean_covered_fraction",
            "fraction",
            "Sum of covered interval nanoseconds / (logical_cpus * window_ns); "
            "no intervals => NaN.",
            availability,
            limitations,
        )
        if info["status"] != "ok":
            continue
        if (
            frame["collection_id"].null_count()
            or frame["collection_id"].n_unique() != 1
        ):
            info["status"] = "unknown_collection"
            continue
        cpu = _integers(frame, "cpu")
        info["invalid_cpu_rows"] = int((cpu < 0).sum())
        frame = frame.filter(pl.Series(cpu >= 0))
        rate_sum, cpu_count, covered_sum = (np.zeros(ticks.size) for _ in range(3))
        reasons = dict.fromkeys(
            (
                "first_samples",
                "invalid_endpoint",
                "nonincreasing_timestamp",
                "negative_counter_delta",
                "invalid_pmu_delta",
                "gap_over_250ms",
                "before_phase",
                "usable_intervals",
            ),
            0,
        )
        for part in frame.partition_by(["collection_id", "cpu"]):
            part = part.sort("ts_uptime_us", maintain_order=True)
            reasons["first_samples"] += 1
            times = _timestamps(part, "ts_uptime_us")
            counts = _integers(part, column)
            enabled, running = (
                _integers(part, c)
                for c in ("pmu_enabled_time_us", "pmu_running_time_us")
            )
            endpoint = (times >= 0) & (counts >= 0) & (counts <= 2**32 - 1)
            endpoint &= (enabled > 0) & (running > 0) & (running <= enabled)
            # Invalid middle samples remain barriers; never bridge over them.
            dt = np.diff(np.maximum(times, 0))
            dc = np.diff(np.maximum(counts, 0))
            de, dr = np.diff(np.maximum(enabled, 0)), np.diff(np.maximum(running, 0))
            bad = {
                "invalid_endpoint": ~(endpoint[:-1] & endpoint[1:]),
                "nonincreasing_timestamp": dt <= 0,
                "negative_counter_delta": dc < 0,
                "invalid_pmu_delta": (de <= 0) | (dr <= 0) | (dr > de),
                "gap_over_250ms": dt > 250_000_000,
                "before_phase": times[:-1] < phase_start_ns,
            }
            valid = np.ones(dt.size, dtype=bool)
            for reason, rejected in bad.items():
                reasons[reason] += int(rejected.sum())
                valid &= ~rejected
            reasons["usable_intervals"] += int(valid.sum())
            starts, ends = times[:-1][valid], times[1:][valid] + 1000
            scaled = dc[valid].astype(float) * de[valid] / dr[valid]
            lo = np.searchsorted(starts, left, side="right")
            hi = np.maximum(lo, np.searchsorted(ends, ticks, side="right"))
            count_sum = np.r_[0.0, np.cumsum(scaled)]
            time_sum = np.r_[0.0, np.cumsum(dt[valid].astype(float))]
            covered = time_sum[hi] - time_sum[lo]
            observed = covered > 0
            rate_sum[observed] += (count_sum[hi] - count_sum[lo])[observed] / (
                covered[observed] / 1e9
            )
            cpu_count += observed
            covered_sum += covered
        observed = cpu_count > 0
        rate[observed] = rate_sum[observed]
        coverage[observed] = cpu_count[observed] / logical_cpus
        covered_fraction[observed] = covered_sum[observed] / (
            float(logical_cpus) * window_ns
        )
        info["interval_qc"] = reasons
        info["interval_qc_note"] = (
            "Rejection reasons overlap; usable_intervals is before per-tick window "
            "selection. First samples are baselines only."
        )

    table = "memory_usage"
    memory_columns = {
        "available": "mem_available_bytes",
        "free": "mem_free_bytes",
        "cache": "cached_bytes",
        "anon": "anon_pages_total_bytes",
        "anon_hugepages": "anon_hugepages_total_bytes",
        "dirty": "dirty_bytes",
        "writeback": "writeback_bytes",
        "shmem": "shmem_total_bytes",
    }
    frame = prepare(
        table,
        ("ts_uptime_us",),
        (*memory_columns.values(), "swap_total_bytes", "swap_free_bytes"),
    )
    info = qc["tables"][table]
    info.update(
        {
            "clock": _MEMORY_CLOCK,
            "physical_clock_lag_unknown": True,
            "proof": (
                "value_i available only at BOOTTIME timestamp_(i+1)*1000 + 1000 ns; "
                "Linux BOOTTIME >= MONOTONIC"
            ),
            "left_carry": (
                "A value may predate phase/window; its next-sample proof must be "
                ">= phase_start and <= tick. Carry only proven prior values."
            ),
            "source_timestamp": (
                "The value retains sample_i's timestamp, "
                "never sample_(i+1)'s timestamp."
            ),
        }
    )
    times = _timestamps(frame, "ts_uptime_us")
    usable = info["status"] == "ok" and times.size >= 2
    if usable and (np.any(times < 0) or np.any(np.diff(times) <= 0)):
        info["status"] = "nonincreasing_or_invalid_timestamps"
        usable = False
    elif info["status"] in ("ok", "empty") and times.size < 2:
        info["status"] = "insufficient_samples"
    values_by_name = {
        name: _numbers(frame, column) for name, column in memory_columns.items()
    }
    total, free = (
        _numbers(frame, "swap_total_bytes"),
        _numbers(frame, "swap_free_bytes"),
    )
    values_by_name["swap_used"] = np.where(
        (total >= free) & (free >= 0), total - free, np.nan
    )
    for name, values in values_by_name.items():
        array = add(
            table,
            "system",
            f"{name}_known_prior_mib",
            "MiB",
            f"Last proven prior system {name} value, not a current aligned gauge.",
            _MEMORY_CLOCK + "; serial next-sample BOOTTIME proof + 1000 ns; "
            "proof in [phase_start,t]; last-known context may predate window",
            [
                "Physical clock lag/suspend offset is unknown; do NOT assume "
                "BOOTTIME equals MONOTONIC or exact one-second gauge alignment.",
                "Only sample_i's value is used, never the future sample_(i+1)'s value; "
                "the final sample has no completion proof.",
                "Overlapping memory categories are not summed; "
                "swap_used is same-sample total minus free.",
                "System context, not Redis THP use; no hugetlb counts. "
                "Historical missing procfs fields may have been defaulted to zero.",
            ],
        )
        if not usable:
            continue
        proof = times[1:] + 1000
        valid = (
            (proof >= phase_start_ns) & np.isfinite(values[:-1]) & (values[:-1] >= 0)
        )
        proof, prior = proof[valid], values[:-1][valid]
        index = np.searchsorted(proof, ticks, side="right") - 1
        known = index >= 0
        array[known] = prior[index[known]] / 2**20

    for name in sorted(set(tables) | set(_EXCLUSIONS)):
        if name not in qc["tables"]:
            reason = _EXCLUSIONS.get(
                name,
                "Metadata/unsupported table is not a causal predictor "
                "in this feature set.",
            )
            qc["exclusions"][name] = reason
            frame = tables.get(name)
            qc["tables"][name] = {
                "status": "excluded",
                "reason": reason,
                "rows": frame.height if isinstance(frame, pl.DataFrame) else None,
            }
    return features, catalog, qc
