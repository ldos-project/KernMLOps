"""Capture YCSB clients without parsing or constructing tables during a benchmark."""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from data_schema.schema import CollectionTable


_MEASUREMENT = re.compile(r"\[([^\]\r\n]*)\]\s*,\s*([^,\r\n]*),(.*)")
_EXPORT_LIKE = re.compile(r"^\s*\[[^\]\r\n]*\]\s*,|^\s*\[[^\]\r\n,]*,")
_STATUS = re.compile(r"\[[^\]\r\n]*:\s*Count\s*=")
_UNIT = re.compile(r"\(([^()]*)\)$")
_POLL_INTERVAL = 0.02
_TERM_TIMEOUT = 1.0
_KILL_TIMEOUT = 1.0


class _MalformedMeasurement(ValueError):
    pass


def _parse_ycsb_line(line: str) -> dict[str, str | float | None] | None:
    """Parse one final export row; reject broken exports, not progress or logging."""
    if _STATUS.match(line.lstrip()):
        return None
    match = _MEASUREMENT.fullmatch(line.strip())
    if match is None:
        if _EXPORT_LIKE.match(line):
            raise _MalformedMeasurement("malformed measurement")
        return None
    section, metric, value_raw = (part.strip() for part in match.groups())
    if not section or not metric or not value_raw:
        raise _MalformedMeasurement(
            "malformed measurement: empty section, metric, or value"
        )

    try:
        value = float(value_raw)
        if not math.isfinite(value):
            value = None
    except ValueError:
        value = None
    unit_match = _UNIT.search(metric)
    unit = (unit_match.group(1).strip() or None) if unit_match else None
    if unit is None and (
        metric in {"Operations", "Count"} or metric.startswith("Return=")
    ):
        unit = "count"
    return {
        "section": section,
        "metric": metric,
        "value_raw": value_raw,
        "value": value,
        "unit": unit,
    }


@dataclass
class _Invocation:
    collection_id: str
    invocation_id: int
    benchmark_name: str
    run_label: str | None
    phase: Literal["load", "run"]
    iteration: int
    outer_iteration: int | None
    argv: list[str]
    start_monotonic_ns: int
    stdout_path: str
    stderr_path: str
    end_monotonic_ns: int | None = None
    exit_code: int | None = None
    execution_state: str = "running"
    parse_state: str = "no_metrics"
    error: str | None = None

    def record_error(self, message: str) -> None:
        self.error = f"{self.error}; {message}" if self.error else message


class YCSBCapture:
    """Single-collection capture; call finalize only after all repeat cycles end.

    Use this object's wait/poll rather than reaping its Popen objects directly.
    This lets cancellation keep group leaders unreaped until group signalling ends.
    Finalize returns cached tables (two typed empty tables when configured but unused)
    and never writes Parquet; the collector owns all post-benchmark table persistence.
    """

    def __init__(self, benchmark_name: str):
        self._benchmark_name = benchmark_name
        self._lock = threading.Lock()
        self._collection_id: str | None = None
        self._collection_dir: Path | None = None
        self._run_label: str | None = None
        self._owner_ids: tuple[int, int] | None = None
        self._invocations: list[_Invocation] = []
        self._processes: dict[subprocess.Popen, _Invocation] = {}
        self._created_logs: dict[int, list[str]] = {}
        self._cancelled = False
        self._tables: list[CollectionTable] | None = None

    @property
    def configured(self) -> bool:
        with self._lock:
            return self._collection_dir is not None

    def configure_output(
        self,
        *,
        collection_id: str,
        collection_dir: Path,
        run_label: str | None = None,
        owner_ids: tuple[int, int] | None = None,
    ) -> None:
        """Create only ycsb/ inside an existing collection directory."""
        with self._lock:
            if self._collection_dir is not None:
                raise RuntimeError("YCSB capture is already configured")
            if self._cancelled or self._tables is not None:
                raise RuntimeError("YCSB capture is closed")
            collection_dir = collection_dir.absolute()
            raw_dir = collection_dir / "ycsb"
            raw_dir.mkdir(exist_ok=True)
            if owner_ids is not None:
                os.chown(raw_dir, *owner_ids)
            self._collection_id = collection_id
            self._collection_dir = collection_dir
            self._run_label = run_label
            self._owner_ids = owner_ids

    def spawn(
        self,
        argv: list[str],
        *,
        phase: Literal["load", "run"],
        iteration: int,
        outer_iteration: int | None = None,
        preexec_fn: Callable[[], None] | None = None,
    ) -> subprocess.Popen:
        with self._lock:
            if self._cancelled or self._tables is not None:
                raise RuntimeError("YCSB capture is closed")
            if self._collection_dir is None or self._collection_id is None:
                raise RuntimeError("YCSB capture is not configured")
            if phase not in ("load", "run"):
                raise ValueError("YCSB phase must be load or run")
            if not 0 <= iteration < 2**32 or (
                outer_iteration is not None and not 0 <= outer_iteration < 2**32
            ):
                raise ValueError("YCSB iteration numbers must fit UInt32")

            invocation_id = len(self._invocations)
            prefix = f"ycsb/{invocation_id:04d}.{phase}"
            invocation = _Invocation(
                collection_id=self._collection_id,
                invocation_id=invocation_id,
                benchmark_name=self._benchmark_name,
                run_label=self._run_label,
                phase=phase,
                iteration=iteration,
                outer_iteration=outer_iteration,
                argv=list(argv),
                start_monotonic_ns=time.monotonic_ns(),
                stdout_path=f"{prefix}.stdout.log",
                stderr_path=f"{prefix}.stderr.log",
            )
            self._invocations.append(invocation)
            self._created_logs[invocation_id] = []
            try:
                with ExitStack() as stack:
                    streams = {}
                    for source, relative_path in (
                        ("stdout", invocation.stdout_path),
                        ("stderr", invocation.stderr_path),
                    ):
                        stream = stack.enter_context(
                            (self._collection_dir / relative_path).open("xb")
                        )
                        self._created_logs[invocation_id].append(source)
                        if self._owner_ids is not None:
                            os.fchown(stream.fileno(), *self._owner_ids)
                        streams[source] = stream
                    process = subprocess.Popen(
                        invocation.argv,
                        stdout=streams["stdout"],
                        stderr=streams["stderr"],
                        start_new_session=True,
                        preexec_fn=preexec_fn,
                    )
                    self._processes[process] = invocation
            except Exception as exc:
                invocation.end_monotonic_ns = time.monotonic_ns()
                invocation.execution_state = "spawn_error"
                invocation.record_error(f"spawn error: {type(exc).__name__}: {exc}")
                raise
            return process

    def _observe_end(
        self, process: subprocess.Popen, invocation: _Invocation
    ) -> int | None:
        """Called with the capture lock held, including every child reap."""
        exit_code = process.poll()
        if exit_code is not None and invocation.end_monotonic_ns is None:
            invocation.end_monotonic_ns = time.monotonic_ns()
            invocation.exit_code = exit_code
            if invocation.execution_state != "cancelled":
                invocation.execution_state = "completed" if exit_code == 0 else "failed"
        return exit_code

    def poll(self, process: subprocess.Popen) -> int | None:
        with self._lock:
            if process not in self._processes:
                raise ValueError("process does not belong to this YCSB capture")
            return self._observe_end(process, self._processes[process])

    def wait(self, process: subprocess.Popen, timeout: float | None = None) -> int:
        # Do not hold the lock across a blocking wait: the collector may cancel us.
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            exit_code = self.poll(process)
            if exit_code is not None:
                return exit_code
            with self._lock:
                if self._cancelled:
                    exit_code = self._observe_end(process, self._processes[process])
                    if exit_code is None:
                        raise RuntimeError(
                            "YCSB client is still active after cancellation"
                        )
                    return exit_code
            delay = _POLL_INTERVAL
            if deadline is not None and timeout is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout)
                delay = min(delay, remaining)
            time.sleep(delay)

    def cancel(self) -> None:
        """Stop launches and give active groups a bounded TERM/KILL/reap window."""
        with self._lock:
            self._cancelled = True
            active = []
            failures = []
            for process, invocation in self._processes.items():
                if self._observe_end(process, invocation) is not None:
                    continue
                invocation.execution_state = "cancelled"
                active.append((process, invocation))
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    invocation.record_error(f"SIGTERM error: {exc}")

            if not active:
                return
            # Keep leaders unreaped during the grace period. Even if TERM exits the
            # leader, KILL must reach resistant descendants without risking PID reuse.
            time.sleep(_TERM_TIMEOUT)
            for process, invocation in active:
                if process.returncode is not None:
                    continue
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    message = f"SIGKILL error: {exc}"
                    invocation.record_error(message)
                    failures.append(message)

            deadline = time.monotonic() + _KILL_TIMEOUT
            for process, invocation in active:
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                    self._observe_end(process, invocation)
                except subprocess.TimeoutExpired:
                    message = f"client {process.pid} did not exit after SIGKILL"
                    invocation.record_error(message)
                    failures.append(message)
            if failures:
                raise RuntimeError("; ".join(failures))

    def finalize(self) -> list[CollectionTable]:
        """Seal capture and parse once. The caller must first finish all cycles."""
        with self._lock:
            if self._tables is not None:
                return self._tables
            if self._collection_dir is None:
                self._tables = []
                return self._tables
            for process, invocation in self._processes.items():
                if self._observe_end(process, invocation) is None:
                    raise RuntimeError("cannot finalize while YCSB clients are active")

            import polars as pl
            from data_schema.ycsb import YCSBInvocationsTable, YCSBMetricsTable

            metrics: list[dict[str, object]] = []
            for invocation in self._invocations:
                first_metric = len(metrics)
                problems = []
                malformed = False
                for source, relative_path in (
                    ("stdout", invocation.stdout_path),
                    ("stderr", invocation.stderr_path),
                ):
                    if source not in self._created_logs[invocation.invocation_id]:
                        problems.append(f"{source}: raw log was not created")
                        continue
                    line_number = 0
                    try:
                        with (self._collection_dir / relative_path).open(
                            "rb"
                        ) as stream:
                            for line_number, raw_line in enumerate(stream, start=1):
                                try:
                                    line = raw_line.decode("utf-8")
                                except UnicodeDecodeError as exc:
                                    problems.append(
                                        f"{source}:{line_number}: decode error: {exc}"
                                    )
                                    continue
                                try:
                                    measurement = _parse_ycsb_line(line)
                                except _MalformedMeasurement as exc:
                                    malformed = True
                                    problems.append(f"{source}:{line_number}: {exc}")
                                    continue
                                if measurement is not None:
                                    metrics.append(
                                        {
                                            "collection_id": invocation.collection_id,
                                            "invocation_id": invocation.invocation_id,
                                            "source": source,
                                            "line_number": line_number,
                                            **measurement,
                                        }
                                    )
                    except FileNotFoundError as exc:
                        problems.append(f"{source}: missing raw log: {exc}")
                    except Exception as exc:
                        problems.append(
                            f"{source}:{line_number}: read/parse error: "
                            f"{type(exc).__name__}: {exc}"
                        )

                has_metrics = len(metrics) > first_metric
                if problems:
                    invocation.parse_state = (
                        "partial" if malformed or has_metrics else "error"
                    )
                    invocation.record_error("; ".join(problems))
                else:
                    invocation.parse_state = "ok" if has_metrics else "no_metrics"

            self._tables = [
                YCSBInvocationsTable.from_df(
                    pl.DataFrame(
                        [asdict(invocation) for invocation in self._invocations],
                        schema=YCSBInvocationsTable.schema(),
                    )
                ),
                YCSBMetricsTable.from_df(
                    pl.DataFrame(metrics, schema=YCSBMetricsTable.schema())
                ),
            ]
            return self._tables
