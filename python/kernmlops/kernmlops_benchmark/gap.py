import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
import time

from data_schema import GraphEngine, demote
from kernmlops_benchmark.benchmark import Benchmark, GenericBenchmarkConfig
from kernmlops_benchmark.errors import (
  BenchmarkNotInCollectionData,
  BenchmarkNotRunningError,
  BenchmarkRunningError,
)
from kernmlops_config import ConfigBase


@dataclass(frozen=True)
class GapBenchmarkConfig(ConfigBase):
  gap_benchmark: Literal["pr"] = "pr"
  gap_benchmark_size: int = 25
  trials: int = 2


class GapBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "gap"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return GapBenchmarkConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> "Benchmark":
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        gap_config = cast(GapBenchmarkConfig, getattr(config, cls.name()))
        return GapBenchmark(generic_config=generic_config, config=gap_config)

    def __init__(self, *, generic_config: GenericBenchmarkConfig, config: GapBenchmarkConfig):
        self.generic_config = generic_config
        self.config = config
        self.benchmark_dir = self.generic_config.get_benchmark_dir() / self.name()
        self.process: subprocess.Popen | None = None
        super().__init__()

    def get_input_file_path(self) -> Path:
        return Path(self.benchmark_dir / "graphs" / f"kron{self.config.gap_benchmark_size}.sg")

    def is_configured(self) -> bool:
        return self.benchmark_dir.is_dir()

    def setup(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        if not self.get_input_file_path().is_file():
          create_graph_process = subprocess.Popen([
            str(self.benchmark_dir / "converter"),
            "-m",
            "-g",
            f"{self.config.gap_benchmark_size}",
            "-b",
            str(self.get_input_file_path()),
          ])
          create_graph_process.wait()
        self.generic_config.generic_setup()

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        self.process = subprocess.Popen(
            [
                str(self.benchmark_dir / self.config.gap_benchmark),
                "-f",
                str(self.get_input_file_path()),
                "-n",
                str(self.config.trials),
            ],
            preexec_fn=demote(),
            stdout=subprocess.DEVNULL,
        )
        self.start_timestamp = int(time.clock_gettime_ns(time.CLOCK_BOOTTIME) / 1000)

    def poll(self) -> int | None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.finish_timestamp = int(time.clock_gettime_ns(time.CLOCK_BOOTTIME) / 1000)
        return self.process.poll()

    def wait(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.process.wait()

    def kill(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.process.terminate()
        self.finish_timestamp = int(time.clock_gettime_ns(time.CLOCK_BOOTTIME) / 1000)

    @classmethod
    def plot_events(cls, graph_engine: GraphEngine) -> None:
        if graph_engine.collection_data.benchmark != cls.name():
            raise BenchmarkNotInCollectionData()
        # TODO(Patrick): plot when a trial starts/ends

    def to_run_info_dict(self) -> dict[str, list]:
        return {
            "benchmark": [self.name()],
            "start_ts_us": [self.start_timestamp],
            "finish_ts_us": [self.finish_timestamp],
        }
