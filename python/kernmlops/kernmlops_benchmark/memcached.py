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
class MemcachedConfig(ConfigBase):
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
    thread_count: int = 1
    target: int = 10000


class MemcachedBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "memcached"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return MemcachedConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> "Benchmark":
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        memcached_config = cast(MemcachedConfig, getattr(config, cls.name()))
        return MemcachedBenchmark(generic_config=generic_config, config=memcached_config)

    def __init__(self, *, generic_config: GenericBenchmarkConfig, config: MemcachedConfig):
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

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        if self.server is not None:
            raise BenchmarkRunningError()

        start_memcached = [
            "memcached",
        ]

        # Start memcached
        self.server = subprocess.Popen(start_memcached, preexec_fn=demote())


        # Ping Memcached
        set_foo_memcached = [
            "echo",
            "-e",
            "set foo 0 300 3\r\nbar\r",
            "|",
            "nc",
            "-q",
            "0",
            "localhost",
            "11211",
        ]
        ping_memcached = subprocess.run(set_foo_memcached, shell=True)
        i = 0
        while i < 10 and ping_memcached.returncode != 0:
            time.sleep(1)
            ping_memcached = subprocess.run(set_foo_memcached, shell=True)
            i+=1

        if ping_memcached.returncode != 0:
            raise BenchmarkError("Memcached Failed To Start")

        delete_foo_memcached = [
            "echo",
            "-e",
            "delete foo",
            "|",
            "nc",
            "-q",
            "0",
            "localhost",
            "11211",
        ]
        delete_foo = subprocess.run(delete_foo_memcached, shell=True)
        if delete_foo.returncode != 0:
            raise BenchmarkError("Delete Single Memcached Record Failing")

        # Run loads
        load_memcached = [
                "python",
                f"{self.benchmark_dir}/YCSB/bin/ycsb",
                "load",
                "memcached",
                "-s",
                "-P",
                f"{self.benchmark_dir}/YCSB/workloads/workloada",
                "-p",
                "memcached.hosts=localhost:11211",
                "-p",
                f"recordcount={self.config.record_count}",
        ]
        load_memcached = subprocess.run(load_memcached, preexec_fn=demote())

        if load_memcached.returncode != 0:
            raise BenchmarkError("Load Memcached Failing")

        run_memcached = [
            f"{self.benchmark_dir}/YCSB/bin/ycsb",
            "run",
            "memcached",
            "-s",
            "-P",
            f"{self.benchmark_dir}/YCSB/workloads/workloada",
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
            "memcached.hosts=localhost:11211",
            "-p",
            f"requestdistribution={self.config.request_distribution}",
            "-p",
            f"threadcount={self.config.thread_count}",
            "-p",
            f"target={self.config.target}"
        ]
        self.process = subprocess.Popen(run_memcached, preexec_fn=demote())

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

    @classmethod
    def plot_events(cls, graph_engine: GraphEngine) -> None:
        if graph_engine.collection_data.benchmark != cls.name():
            raise BenchmarkNotInCollectionData()
