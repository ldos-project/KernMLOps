import signal
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


@dataclass(frozen=True)
class RedisConfig(ConfigBase):
    # Core operation parameters
    operation_count: int = 1000000
    record_count: int = 1000000
    read_proportion: float = 0.5
    update_proportion: float = 0.5
    scan_proportion: float = 0.0
    insert_proportion: float = 0.0
    rmw_proportion: float = 0.00
    scan_proportion: float = 0.00
    delete_proportion: float = 0.00

    # Distribution and performance parameters
    request_distribution: str = "uniform"
    thread_count: int = 16
    target: int = 10000

kill_redis = [
    "killall",
    "-9",
    "redis-server",
]

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
        self.server: subprocess.Popen | None = None

    def is_configured(self) -> bool:
        return self.benchmark_dir.is_dir()

    def setup(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        self.generic_config.generic_setup()

        # Kill Redis
        kill_proc = subprocess.Popen(kill_redis)
        kill_proc.wait()

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        if self.server is not None:
            raise BenchmarkRunningError()


        # start the redis server
        start_redis = [
            "redis-server",
            "./scripts/redis.conf",
        ]
        self.server = subprocess.Popen(start_redis,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

        # Wait for redis
        ping_redis = subprocess.run(["redis-cli", "ping"])
        i = 0
        while i < 10 and ping_redis.returncode != 0:
            time.sleep(1)
            ping_redis = subprocess.run(["redis-cli", "ping"])

        if ping_redis.returncode != 0:
            raise BenchmarkError("Redis Failed To Start")

        self.server = subprocess.Popen(start_redis)

        # Load Server
        load_redis = [
                "python",
                f"{self.benchmark_dir}/YCSB/bin/ycsb",
                "load",
                "redis",
                "-s",
                "-P",
                f"{self.benchmark_dir}/YCSB/workloads/workloada",
                "-p",
                "redis.host=127.0.0.1",
                "-p",
                "redis.port=6379",
                "-p",
                f"recordcount={self.config.record_count}",
        ]

        load_redis = subprocess.Popen(load_redis, preexec_fn=demote())

        load_redis.wait()
        if load_redis.returncode != 0:
            raise BenchmarkError("Loading Redis Failing")

        # Run Benchmark
        run_redis = [
                f"{self.benchmark_dir}/YCSB/bin/ycsb",
                "run",
                "redis",
                "-s",
                "-P",
                f"{self.benchmark_dir}/YCSB/workloads/workloada-redis",
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
                f"readmodifywriteproportion={self.config.rmw_proportion}",
                "-p",
                f"scanproportion={self.config.scan_proportion}",
                "-p",
                f"deleteproportion={self.config.delete_proportion}",
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
        ]
        self.process = subprocess.Popen(run_redis, preexec_fn=demote(), stdout=subprocess.DEVNULL)

    def poll(self) -> int | None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        ret = self.process.poll()
        if ret is None:
            return ret
        self.end_server()
        return ret

    def wait(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.process.wait()
        self.end_server()

    def kill(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.process.terminate()
        self.end_server()

    def end_server(self) -> None:
        if self.server is None:
            return
        self.server.send_signal(signal.SIGINT)
        if self.server.wait(10) is None:
            self.server.terminate()
        self.server = None
        kill_proc = subprocess.Popen(kill_redis)
        kill_proc.wait()

    @classmethod
    def plot_events(cls, graph_engine: GraphEngine) -> None:
        if graph_engine.collection_data.benchmark != cls.name():
            raise BenchmarkNotInCollectionData()
