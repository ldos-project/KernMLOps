# File: kernmlops_benchmark/pytorch_benchmark.py
#!/usr/bin/env python3
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from data_schema import GraphEngine
from kernmlops_benchmark.benchmark import Benchmark, GenericBenchmarkConfig
from kernmlops_benchmark.errors import (
    BenchmarkError,
    BenchmarkNotInCollectionData,
    BenchmarkNotRunningError,
    BenchmarkRunningError,
)
from kernmlops_config import ConfigBase
from pytimeparse.timeparse import timeparse
from torchvision import datasets, transforms


@dataclass(frozen=True)
class PyTorchConfig(ConfigBase):
    repeat: int = 1
    train_script: str = "python/kernmlops/kernmlops_benchmark/train.py"
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cpu"
    sleep: str | None = None

class PyTorchBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "pytorch"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return PyTorchConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> "Benchmark":
        generic = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        pt_cfg  = cast(PyTorchConfig, getattr(config, cls.name()))
        return PyTorchBenchmark(generic_config=generic, config=pt_cfg)

    def __init__(self, *, generic_config: GenericBenchmarkConfig, config: PyTorchConfig):
        self.generic_config = generic_config
        self.config         = config
        self.benchmark_dir  = self.generic_config.get_benchmark_dir() / "pytorch"
        self.process: subprocess.Popen | None = None

    def is_configured(self) -> bool:
        return Path(self.config.train_script).is_file()

    def setup(self) -> None:
        # ensure benchmark directories
        if self.process is not None:
            raise BenchmarkRunningError("Benchmark already running")
        self.generic_config.generic_setup()

        # download dataset if missing
        data_root = self.generic_config.get_benchmark_dir() / "dataset" / self.name()
        if not data_root.exists():
            data_root.mkdir(parents=True, exist_ok=True)
            # perform download using torchvision
            transform = transforms.Compose([transforms.ToTensor()])
            # download both train and test splits
            datasets.FashionMNIST(root=str(data_root), train=True, download=True, transform=transform)
            datasets.FashionMNIST(root=str(data_root), train=False, download=True, transform=transform)

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError("Benchmark already running")

        pause : int | float | None = None if self.config.sleep is None else timeparse(self.config.sleep)

        last_proc: subprocess.Popen | None = None

        for i in range(self.config.repeat):
            if last_proc is not None:
                last_proc.wait()
                if last_proc.returncode != 0:
                    self.process = last_proc
                    raise BenchmarkError(f"PyTorch run #{i} failed (exit {last_proc.returncode})")
                if pause:
                    time.sleep(pause)

            # script_path = self.config.train_script
            # data_root = self.generic_config.get_benchmark_dir() / "dataset" / "pytorch"

            cmd = ["python", "python/kernmlops/kernmlops_benchmark/train2.py"]
            # cmd = [
            #     "python",
            #     str(script_path),
            #     "--epochs", str(self.config.epochs),
            #     "--batch-size", str(self.config.batch_size),
            #     "--learning-rate", str(self.config.learning_rate),
            #     "--device", self.config.device,
            #     "--data-root",   str(data_root),
            # ]
            last_proc = subprocess.Popen(cmd, env=os.environ.copy())

        self.process = last_proc
        print(f"Process done, return code: {last_proc.returncode}")

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
        raise NotImplementedError("No event plotting defined for PyTorch")
