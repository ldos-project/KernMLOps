import subprocess
from dataclasses import dataclass, field
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


def _default_devices() -> list[str]:
    return ["/dev/nvme0n1", "/dev/nvme1n1", "/dev/nvme2n1"]


def _default_traces() -> list[str]:
    return ["azure", "bing_i", "cosmos"]


@dataclass(frozen=True)
class LinnosBenchmarkConfig(ConfigBase):
  use_root: bool = False # required for accessing raw devices
  shuffle_traces: bool = False
  type: Literal["baseline", "failover"] = "baseline"
  devices: list[str] = field(default_factory=_default_devices) # devices cannot contain '-'
  traces: list[str] = field(default_factory=_default_traces)


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
        for trace in self.config.traces:
            subprocess.Popen(
                [
                    "bash",
                    f"gen_{trace}.sh",
                ],
                cwd=str(self.benchmark_dir / "trace_tools"),
                preexec_fn=demote(),
                stdout=subprocess.DEVNULL,
            )

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        traces = [
            str(self.benchmark_dir / "trace_tools" / trace / f"{trace}1.trace")
            for trace in self.config.traces
        ]
        if self.config.shuffle_traces:
            shuffle(traces)
        # this must be run as root if accessing raw devices
        if not self.config.use_root:
            self.process = subprocess.Popen(
                [
                    str(self.benchmark_dir / "io_replayer" / "replayer"),
                    self.config.type,
                    "logfile",
                    str(len(self.config.devices)),
                    ",".join(self.config.devices),
                    *traces,
                ],
                cwd=str(self.benchmark_dir / "io_replayer"),
                preexec_fn=demote(),
            )
        else:
            self.process = subprocess.Popen(
                [
                    str(self.benchmark_dir / "io_replayer" / "replayer"),
                    self.config.type,
                    "logfile",
                    str(len(self.config.devices)),
                    ",".join(self.config.devices),
                    *traces,
                ],
                cwd=str(self.benchmark_dir / "io_replayer"),
                preexec_fn=demote(),
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
