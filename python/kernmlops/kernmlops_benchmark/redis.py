import subprocess
from dataclasses import dataclass
from typing import cast

from data_schema import GraphEngine, demote
from kernmlops_benchmark.benchmark import Benchmark, GenericBenchmarkConfig
from kernmlops_benchmark.errors import (
  BenchmarkNotInCollectionData,
  BenchmarkNotRunningError,
  BenchmarkRunningError,
)
from kernmlops_config import ConfigBase


@dataclass(frozen=True)
class RedisConfig(ConfigBase):
    # Core operation parameters
    operation_count: int = 1000000
    record_count: int = 1000000
    read_proportion: float = 0.5
    update_proportion: float = 0.5
    scan_proportion: float = 0.0
    insert_proportion: float = 0.0

    # Distribution and performance parameters
    request_distribution: str = "uniform"
    thread_count: int = 16
    target: int = 10000


class RedisBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "redis"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return RedisConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> "Benchmark":
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        redis_config = cast(RedisConfig, getattr(config, cls.name()))
        return RedisBenchmark(generic_config=generic_config, config=redis_config)

    def __init__(self, *, generic_config: GenericBenchmarkConfig, config: RedisConfig):
        self.generic_config = generic_config
        self.config = config
        self.benchmark_dir = self.generic_config.get_benchmark_dir() / "ycsb"
        self.process: subprocess.Popen | None = None

    def is_configured(self) -> bool:
        return self.benchmark_dir.is_dir()

    def setup(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        self.generic_config.generic_setup()

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()

        self.process = subprocess.Popen(
            [
                f"{self.benchmark_dir}/ycsb-0.17.0/bin/ycsb",
                "run",
                "redis",
                "-s",
                "-P",
                f"{self.benchmark_dir}/ycsb-0.17.0/workloads/workloada-redis",
                "-p",
                f"operationcount={self.config.operation_count}",
                "-p",
                f"recordcount={self.config.record_count}",
                "-p",
                "workload=site.ycsb.workloads.CoreWorkload",
                "-p",
                f"readproportion={self.config.read_proportion}",
                "-p",
                f"updateproportion={self.config.update_proportion}",
                "-p",
                f"scanproportion={self.config.scan_proportion}",
                "-p",
                f"insertproportion={self.config.insert_proportion}",
                "-p",
                "redis.host=127.0.0.1",
                "-p",
                "redis.port=6379",
                "-p",
                f"requestdistribution={self.config.request_distribution}",
                "-p",
                f"threadcount={self.config.thread_count}",
                "-p",
                f"target={self.config.target}"
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
