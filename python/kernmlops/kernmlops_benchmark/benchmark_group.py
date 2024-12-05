"""Benchmark with multiple benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, make_dataclass
from typing import Literal, cast
import time
import subprocess

from data_schema import GraphEngine, demote
from kernmlops_benchmark.benchmark import Benchmark, GenericBenchmarkConfig
from kernmlops_benchmark.stress_ng import StressNgBenchmark, StressNgBenchmarkConfig
from kernmlops_benchmark.errors import (
    BenchmarkNotInCollectionData,
    BenchmarkNotRunningError,
    BenchmarkRunningError,
)
from kernmlops_config import ConfigBase


@dataclass(frozen=True)
class ChildBenchmarkConfig(ConfigBase):
    name: str
    config: dict[str, ConfigBase]


@dataclass(frozen=True)
class BenchmarkGroupConfig(ConfigBase):
    benchmarks: list[ChildBenchmarkConfig] = field(default_factory=list)


class BenchmarkGroup(Benchmark):
    @classmethod
    def name(cls) -> str:
        return "benchmark_group"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return BenchmarkGroupConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> Benchmark:
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        benchmark_group_config = cast(BenchmarkGroupConfig, getattr(config, cls.name()))
        return BenchmarkGroup(
            generic_config=generic_config, config=benchmark_group_config
        )

    def __init__(
        self, *, generic_config: GenericBenchmarkConfig, config: BenchmarkGroupConfig
    ):
        super().__init__()
        self.generic_config = generic_config
        self.config = config
        self.benchmarks: list[StressNgBenchmark] = []
        for benchmark_config_dict in self.config.benchmarks:
            benchmark_config: ChildBenchmarkConfig = make_dataclass(
                "ChildBenchmarkConfig",
                ((k, type(v)) for k, v in benchmark_config_dict.items()),
            )(**benchmark_config_dict)
            if benchmark_config.name == StressNgBenchmark.name():
                stress_ng_config: StressNgBenchmarkConfig = make_dataclass(
                    "StressNgBenchmarkConfig",
                    ((k, type(v)) for k, v in benchmark_config.config.items()),
                )(**benchmark_config.config)
                self.benchmarks.append(
                    StressNgBenchmark(
                        generic_config=generic_config, config=stress_ng_config
                    )
                )
            else:
                raise NotImplementedError(
                    f"Benchmark {benchmark_config.name} not supported in BenchmarkGroup yet."
                )
        self.processes: list[subprocess.Popen] = []
        self.running_process_idx: int = -1

    def is_configured(self) -> bool:
        return all(benchmark.is_configured() for benchmark in self.benchmarks)

    def setup(self) -> None:
        self.generic_config.generic_setup()

    def run(self) -> None:
        self.running_process_idx = 0
        self.start_timestamp = int(time.clock_gettime_ns(time.CLOCK_BOOTTIME) / 1000)

    def poll(self) -> int | None:
        return_code = None
        if len(self.processes) < self.running_process_idx + 1:
            self.benchmarks[self.running_process_idx].setup()
            time.sleep(1)
            self.benchmarks[self.running_process_idx].run()
            self.processes.append(self.benchmarks[self.running_process_idx].process)

        benchmark_return_code = self.benchmarks[self.running_process_idx].poll()

        if benchmark_return_code is not None:
            self.running_process_idx += 1
        
        if self.running_process_idx == len(self.benchmarks):
            return_code = 0
            self.finish_timestamp = int(time.clock_gettime_ns(time.CLOCK_BOOTTIME) / 1000)

        return return_code

    def wait(self) -> None:
        for process in self.processes:
            process.wait()

    def kill(self) -> None:
        for process in self.processes:
            process.terminate()
        self.finish_timestamp = int(time.clock_gettime_ns(time.CLOCK_BOOTTIME) / 1000)

    @classmethod
    def plot_events(cls, graph_engine: GraphEngine) -> None:
        if graph_engine.collection_data.benchmark != cls.name():
            raise BenchmarkNotInCollectionData()

    def to_run_info_dict(self) -> dict[str, list]:
        return {
            "benchmark": [benchmark.name() for benchmark in self.benchmarks],
            "args": [" ".join(benchmark.config.args) for benchmark in self.benchmarks],
            "start_ts_us": [benchmark.start_timestamp for benchmark in self.benchmarks],
            "finish_ts_us": [benchmark.finish_timestamp for benchmark in self.benchmarks],
            "return_code": [process.returncode for process in self.processes],
        }
