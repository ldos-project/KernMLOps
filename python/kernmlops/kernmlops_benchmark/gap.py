import subprocess
from pathlib import Path

from data_schema import GraphEngine, demote

from kernmlops_benchmark.benchmark import Benchmark
from kernmlops_benchmark.errors import (
    BenchmarkNotInCollectionData,
    BenchmarkNotRunningError,
    BenchmarkRunningError,
)


class GapBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "gap"

    def __init__(self, benchmark_dir: Path, cpus: int | None):
        self.benchmark_dir = benchmark_dir / self.name()
        self.graph_algo = "pr"
        self.trials = 2
        self.process: subprocess.Popen | None = None

    def is_configured(self) -> bool:
        return self.benchmark_dir.is_dir()

    def setup(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        subprocess.check_call(
            ["bash", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
            stdout=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["bash", "-c", "echo always > /sys/kernel/mm/transparent_hugepage/enabled"],
            stdout=subprocess.DEVNULL,
        )

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        self.process = subprocess.Popen(
            [
                str(self.benchmark_dir / self.graph_algo),
                "-f",
                str(self.benchmark_dir / "graphs" / "kron25.sg"),
                "-n",
                str(self.trials),
            ],
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
        # TODO(Patrick): plot when a trial starts/ends
