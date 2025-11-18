import subprocess
from dataclasses import dataclass
from typing import cast

from data_schema import FileDataTable, GraphEngine, demote
from kernmlops_benchmark.benchmark import Benchmark, GenericBenchmarkConfig
from kernmlops_benchmark.errors import (
    BenchmarkNotInCollectionData,
    BenchmarkNotRunningError,
    BenchmarkRunningError,
)
from kernmlops_config import ConfigBase


@dataclass(frozen=True)
class LinuxBuildBenchmarkConfig(ConfigBase):
    pass


class LinuxBuildBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "linux_build"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return LinuxBuildBenchmarkConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> "Benchmark":
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        linux_config = cast(LinuxBuildBenchmarkConfig, getattr(config, cls.name()))
        return LinuxBuildBenchmark(generic_config=generic_config, config=linux_config)

    def __init__(self, *, generic_config: GenericBenchmarkConfig, config: LinuxBuildBenchmarkConfig):
        self.generic_config = generic_config
        self.config = config
        self.benchmark_dir = self.generic_config.get_benchmark_dir() / self.name()
        self.process: subprocess.Popen | None = None

    def is_configured(self) -> bool:
        return self.benchmark_dir.is_dir()

    def setup(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        if (self.benchmark_dir / "Makefile").exists():
            subprocess.check_call(
                ["make", "-C", self.benchmark_dir, "clean"],
                preexec_fn=demote(),
                stdout=subprocess.DEVNULL,
            )
        subprocess.check_call(
            [
                "make",
                "-C",
                str(self.benchmark_dir / "../linux_kernel"),
                f"O={self.benchmark_dir}",
                "defconfig",
            ],
            preexec_fn=demote(),
            stdout=subprocess.DEVNULL,
        )
        self.generic_config.generic_setup()

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        jobs = f"-j{self.generic_config.cpus}" if self.generic_config.cpus else "-j"
        self.process = subprocess.Popen(
            ["make", "-C", self.benchmark_dir, jobs],
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
        file_data = graph_engine.collection_data.get(FileDataTable)
        if file_data is None:
            return None

        graph_engine.plot_event_as_sec(ts_us=file_data.get_first_occurrence_us("make"))
        graph_engine.plot_event_as_sec(ts_us=file_data.get_last_occurrence_us("bzImage"))
        graph_engine.plot_event_as_sec(ts_us=file_data.get_last_occurrence_us("vmlinux.bin"))
        graph_engine.plot_event_as_sec(ts_us=file_data.get_last_occurrence_us("vmlinux.o"))
        graph_engine.plot_event_as_sec(ts_us=file_data.get_last_occurrence_us("vmlinux"))
