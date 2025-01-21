"""Benchmark with multiple stress-ng set benchmarks."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import cast

from data_schema import GraphEngine
from kernmlops_benchmark.benchmark import Benchmark, GenericBenchmarkConfig
from kernmlops_benchmark.errors import BenchmarkNotInCollectionData
from kernmlops_benchmark.stress_ng_set import (
    StressNgSetBenchmark,
    StressNgSetBenchmarkConfig,
)
from kernmlops_config import ConfigBase


@dataclass(frozen=True)
class StressNgBenchmarkGroupConfig(ConfigBase):
    starting_idx: int = 0
    num_exps: int = 0
    num_reps: int = 1

class StressNgBenchmarkGroup(Benchmark):
    @classmethod
    def name(cls) -> str:
        return "stress_ng_benchmarks"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return StressNgBenchmarkGroupConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> Benchmark:
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        benchmark_group_config = cast(StressNgBenchmarkGroupConfig, getattr(config, cls.name()))
        return StressNgBenchmarkGroup(
            generic_config=generic_config, config=benchmark_group_config
        )

    def __init__(
        self, *, generic_config: GenericBenchmarkConfig, config: StressNgBenchmarkGroupConfig
    ):
        super().__init__()
        self.generic_config = generic_config
        self.config = config
        self.benchmarks: list[StressNgSetBenchmark] = []
        for exp_id in range(self.config.starting_idx, self.config.starting_idx + self.config.num_exps):
            for _ in range(self.config.num_reps):
                stress_ng_set_config = StressNgSetBenchmarkConfig(
                    stress_ng_benchmark="stress-ng-set",
                    args=[str(exp_id)],
                )
                self.benchmarks.append(
                    StressNgSetBenchmark(
                        generic_config=generic_config, config=stress_ng_set_config
                    )
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
