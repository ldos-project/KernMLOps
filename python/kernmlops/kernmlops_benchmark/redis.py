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
from pytimeparse.timeparse import timeparse


@dataclass(frozen=True)
class RedisConfig(ConfigBase):
    repeat: int = 1
    outer_repeat: int = 1
    tcmalloc: bool = False
    mem_cgroup_size: str | None = None
    # Core operation parameters
    field_count: int = 256
    field_length: int = 16
    min_field_length: int = 16
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
    field_length_distribution: str = "uniform"
    request_distribution: str = "uniform"
    thread_count: int = 1
    target: int = 10000
    sleep: str | None = None
    server_sleep: str | None = None
    explicit_purge: bool = False

size_redis = [
    "redis-cli",
    "DBSIZE",
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

    def redis_server_name(self) -> str:
        return "redis-server-tcmalloc" if self.config.tcmalloc else "redis-server"

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

    def purge_server(self) -> None:
        # Purge Redis
        purge_redis = subprocess.run(["redis-cli", "MEMORY", "PURGE"])
        if purge_redis.returncode != 0:
            raise BenchmarkError("Redis Failed To Start")

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        if self.server is not None:
            raise BenchmarkRunningError()

        cgroup_path = ""
        if self.config.mem_cgroup_size:
            cgroup_path = "/sys/fs/cgroup/redis_mem"
            subprocess.run(["mkdir", "-p", cgroup_path], check=True)
            with open(f"{cgroup_path}/memory.max", "w") as f:
                f.write(str(self.config.mem_cgroup_size))

        # start the redis server
        start_redis = [
            self.redis_server_name(),
            "./config/redis.conf",
        ]
        self.server = subprocess.Popen(start_redis)

        # Wait for redis
        ping_redis = subprocess.run(["redis-cli", "ping"])
        i = 0
        while i < 10 and ping_redis.returncode != 0:
            time.sleep(1)
            ping_redis = subprocess.run(["redis-cli", "ping"])
            i+=1

        if ping_redis.returncode != 0:
            raise BenchmarkError("Redis Failed To Start")

        # Add the Redis process to the cgroup if set
        if self.config.mem_cgroup_size:
            ps_output = subprocess.check_output(["pgrep", "-f", "redis-server"])
            redis_pid = ps_output.decode().strip()
            try:
                with open(f"{cgroup_path}/cgroup.procs", "w") as f:
                    f.write(redis_pid)
            except Exception as e:
                print(f"WARNING: Failed to set Redis cgroup: {e}")

        server_space : int | float | None = None if self.config.server_sleep is None else timeparse(self.config.server_sleep)
        if server_space is not None :
            time.sleep(server_space)

        space : int | float | None = None if self.config.sleep is None else timeparse(self.config.sleep)
        process: subprocess.Popen | None = None
        for out_i in range(self.config.outer_repeat):
            for i in range(self.config.repeat):
                if process is not None:
                    process.wait()
                    if self.config.explicit_purge:
                        self.purge_server()
                    if space is not None :
                        time.sleep(space)
                    if process.returncode != 0:
                        self.process = process
                        raise BenchmarkError(f"Redis Run {(2 * i) - 1} Failed")

                insert_start = out_i * self.config.record_count
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
                        "-p",
                        f"fieldcount={self.config.field_count}",
                        "-p",
                        f"fieldlength={self.config.field_length}",
                        "-p",
                        f"minfieldlength={self.config.min_field_length}",
                        "-p",
                        f"insertstart={insert_start}",
                        "-p",
                        f"fieldlengthdistribution={self.config.field_length_distribution}",
                ]

                if i == 0:
                    load_redis = subprocess.Popen(load_redis, preexec_fn=demote())

                    load_redis.wait()
                    if load_redis.returncode != 0:
                        raise BenchmarkError("Loading Redis Failing")

                    if self.config.explicit_purge:
                        self.purge_server()

                subprocess.run(size_redis)

                record_count = (out_i + 1) * self.config.record_count

                run_redis = [
                        f"{self.benchmark_dir}/YCSB/bin/ycsb",
                        "run",
                        "redis",
                        "-s",
                        "-P",
                        f"{self.benchmark_dir}/YCSB/workloads/workloada",
                        "-p",
                        f"operationcount={self.config.operation_count}",
                        "-p",
                        f"recordcount={record_count}",
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
                        f"target={self.config.target}",
                        "-p",
                        f"fieldcount={self.config.field_count}",
                        "-p",
                        f"fieldlength={self.config.field_length}",
                        "-p",
                        f"minfieldlength={self.config.min_field_length}",
                        "-p",
                        f"fieldlengthdistribution={self.config.field_length_distribution}",
                ]
                process = subprocess.Popen(run_redis, preexec_fn=demote())
        self.process = process

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
        subprocess.run(size_redis)
        self.server.send_signal(signal.SIGINT)
        try:
            self.server.wait(10)
        except subprocess.TimeoutExpired:
            self.server.terminate()

        self.server = None

    @classmethod
    def plot_events(cls, graph_engine: GraphEngine) -> None:
        if graph_engine.collection_data.benchmark != cls.name():
            raise BenchmarkNotInCollectionData()
