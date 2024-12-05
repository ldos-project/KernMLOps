import subprocess
from dataclasses import dataclass
from random import shuffle
from typing import Literal, cast

from data_schema import GraphEngine, demote
from kernmlops_benchmark.benchmark import Benchmark, GenericBenchmarkConfig
from kernmlops_benchmark.errors import (
  BenchmarkNotInCollectionData,
  BenchmarkNotRunningError,
  BenchmarkRunningError,
)
from kernmlops_config import ConfigBase


@dataclass(frozen=True)
class LinnosBenchmarkConfig(ConfigBase):
  type: Literal["baseline", "failover"] = "baseline"
  device_0: str = "/dev/nvme0n1"
  device_1: str = "/dev/nvme1n1"
  device_2: str = "/dev/nvme2n1"


class LinnosBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "linnos"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return LinnosBenchmarkConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> "Benchmark":
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        linnos_config = cast(LinnosBenchmarkConfig, getattr(config, cls.name()))
        return LinnosBenchmark(generic_config=generic_config, config=linnos_config)

    def __init__(self, *, generic_config: GenericBenchmarkConfig, config: LinnosBenchmarkConfig):
        self.generic_config = generic_config
        self.config = config
        self.benchmark_dir = self.generic_config.get_benchmark_dir() / self.name() / "src" / "linnos"
        self.process: subprocess.Popen | None = None

    def is_configured(self) -> bool:
        return self.benchmark_dir.is_dir()

    def setup(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        self.generic_config.generic_setup()
        subprocess.Popen(
            [
                "bash",
                "gen_azure.sh",
            ],
            cwd=str(self.benchmark_dir / "trace_tools"),
            preexec_fn=demote(),
            stdout=subprocess.DEVNULL,
        )
        subprocess.Popen(
            [
                "bash",
                "gen_bing_i.sh",
            ],
            cwd=str(self.benchmark_dir / "trace_tools"),
            preexec_fn=demote(),
            stdout=subprocess.DEVNULL,
        )
        subprocess.Popen(
            [
                "bash",
                "gen_cosmos.sh",
            ],
            cwd=str(self.benchmark_dir / "trace_tools"),
            preexec_fn=demote(),
            stdout=subprocess.DEVNULL,
        )

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        traces = [
            str(self.benchmark_dir / "trace_tools" / "azure" / "azure1.trace"),
            str(self.benchmark_dir / "trace_tools" / "bing_i" / "bing_i1.trace"),
            str(self.benchmark_dir / "trace_tools" / "cosmos" / "cosmos1.trace"),
        ]
        shuffle(traces)
        # this must be run as root
        self.process = subprocess.Popen(
            [
                str(self.benchmark_dir / "io_replayer" / "replayer"),
                self.config.type,
                "3ssds",
                "3",
                f"{self.config.device_0}-{self.config.device_1}-{self.config.device_2}",
                *traces,
            ],
            cwd=str(self.benchmark_dir / "io_replayer"),
            stdout=subprocess.DEVNULL,
        )

    def poll(self) -> int | None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        return self.process.poll()

    def wait(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.process.wait()

    def kill(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.process.terminate()

    @classmethod
    def plot_events(cls, graph_engine: GraphEngine) -> None:
        if graph_engine.collection_data.benchmark != cls.name():
            raise BenchmarkNotInCollectionData()
