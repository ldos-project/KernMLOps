import subprocess
import time
from dataclasses import dataclass
from typing import cast

from data_schema import GraphEngine, demote
from kernmlops_benchmark.benchmark import Benchmark, GenericBenchmarkConfig
from kernmlops_benchmark.errors import (
    BenchmarkError,
    BenchmarkNotInCollectionData,
    BenchmarkNotRunningError,
    BenchmarkRunningError,
)
from kernmlops_config import ConfigBase
from pytimeparse.timeparse import timeparse


@dataclass(frozen=True)
class PytorchConfig(ConfigBase):
    repeat: int = 1
    # Path (relative to benchmark_dir) of your training script
    script: str = "train.py"
    # Core training hyperparameters
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cpu"
    # If you want a pause between repeats, e.g. "5s", "1m"
    sleep: str | None = None


class PyTorchBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "pytorch"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return PytorchConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> "Benchmark":
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        pt_config      = cast(PytorchConfig,      getattr(config, cls.name()))
        return PyTorchBenchmark(generic_config=generic_config, config=pt_config)

    def __init__(self, *, generic_config: GenericBenchmarkConfig, config: PytorchConfig):
        self.generic_config = generic_config
        self.config         = config
        # assume your train.py lives under <benchmark_dir>/pytorch/
        self.benchmark_dir = self.generic_config.get_benchmark_dir() / "pytorch"
        self.process: subprocess.Popen | None = None

    def is_configured(self) -> bool:
        return (self.benchmark_dir / self.config.script).exists()

    def setup(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError("Benchmark already running")
        # any one‑time setup steps: e.g., download dataset, prepare env
        self.generic_config.generic_setup()

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError("Benchmark already running")

        # parse optional sleep interval
        pause = None if self.config.sleep is None else timeparse(self.config.sleep)
        proc: subprocess.Popen | None = None

        for i in range(self.config.repeat):
            if proc is not None:
                proc.wait()
                if proc.returncode != 0:
                    self.process = proc
                    raise BenchmarkError(f"PyTorch run #{i} failed (exit {proc.returncode})")
                if pause is not None:
                    time.sleep(pause)

            cmd = [
                "python",
                str(self.benchmark_dir / self.config.script),
                "--epochs",       str(self.config.epochs),
                "--batch-size",   str(self.config.batch_size),
                "--learning-rate",str(self.config.learning_rate),
                "--device",       self.config.device,
            ]
            proc = subprocess.Popen(cmd, preexec_fn=demote())
        # keep last run for polling/waiting
        self.process = proc

    def poll(self) -> int | None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        ret = self.process.poll()
        if ret is not None:
            # nothing to clean up here
            pass
        return ret

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
        # implement any post‑hoc plotting you'd like here
        raise NotImplementedError("No event plotting defined for PyTorch")
