import logging
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import cast

from data_schema import CollectionTable, GraphEngine, demote
from kernmlops_benchmark.benchmark import Benchmark, GenericBenchmarkConfig
from kernmlops_benchmark.errors import (
    BenchmarkError,
    BenchmarkNotInCollectionData,
    BenchmarkNotRunningError,
    BenchmarkRunningError,
)
from kernmlops_benchmark.ycsb import YCSBCapture
from kernmlops_config import ConfigBase
from pytimeparse.timeparse import timeparse


@dataclass(frozen=True)
class RedisConfig(ConfigBase):
    repeat: int = 1
    outer_repeat: int = 1
    tcmalloc: bool = False
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
    load_from_rdb: bool = False


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
        self.ycsb = YCSBCapture(self.name())
        self._running = False
        self._cancelled = False
        self._server_lock = Lock()

    def configure_output(
        self,
        *,
        collection_id: str,
        collection_dir: Path,
        run_label: str | None = None,
        owner_ids: tuple[int, int] | None = None,
    ) -> None:
        self.ycsb.configure_output(
            collection_id=collection_id,
            collection_dir=collection_dir,
            run_label=run_label,
            owner_ids=owner_ids,
        )

    def finalize(self) -> list[CollectionTable]:
        if self._running:
            raise BenchmarkRunningError()
        return self.ycsb.finalize()

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
        if self._cancelled:
            raise BenchmarkError("Redis benchmark was cancelled")
        if not self.ycsb.configured:
            raise BenchmarkError("Redis benchmark output must be configured before run")
        if self.config.repeat < 1 or self.config.outer_repeat < 1:
            raise BenchmarkError("Redis repeat counts must be positive")

        if not self.config.load_from_rdb:
            dump = Path("dump.rdb")
            if dump.exists():
                shutil.move(dump, dump.with_suffix(".rdb.bak"))

        # start the redis server
        start_redis = [
            self.redis_server_name(),
            "./config/redis.conf",
        ]
        with self._server_lock:
            if self._cancelled:
                raise BenchmarkError("Redis benchmark was cancelled")
            self._running = True
            self.server = subprocess.Popen(start_redis)

        # Wait for redis
        ping_redis = subprocess.run(["redis-cli", "ping"])
        i = 0
        while i < 10 and ping_redis.returncode != 0:
            time.sleep(1)
            ping_redis = subprocess.run(["redis-cli", "ping"])
            i += 1

        if ping_redis.returncode != 0:
            raise BenchmarkError("Redis Failed To Start")

        server_space: int | float | None = (
            None
            if self.config.server_sleep is None
            else timeparse(self.config.server_sleep)
        )
        if server_space is not None:
            time.sleep(server_space)

        space: int | float | None = (
            None if self.config.sleep is None else timeparse(self.config.sleep)
        )
        process: subprocess.Popen | None = None
        for out_i in range(self.config.outer_repeat):
            for i in range(self.config.repeat):
                if process is not None:
                    self.ycsb.wait(process)
                    if self.config.explicit_purge:
                        self.purge_server()
                    if space is not None:
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
                    load_process = self.ycsb.spawn(
                        load_redis,
                        phase="load",
                        iteration=i,
                        outer_iteration=out_i,
                        preexec_fn=demote(),
                    )

                    if self.ycsb.wait(load_process) != 0:
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
                process = self.ycsb.spawn(
                    run_redis,
                    phase="run",
                    iteration=i,
                    outer_iteration=out_i,
                    preexec_fn=demote(),
                )
        self.process = process

    def poll(self) -> int | None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        ret = self.ycsb.poll(self.process)
        if ret is None:
            return ret
        self.end_server()
        self._running = False
        return ret

    def wait(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.ycsb.wait(self.process)
        self.end_server()
        self._running = False

    def kill(self) -> None:
        self._cancelled = True
        try:
            self.ycsb.cancel()
        finally:
            try:
                self.end_server()
            finally:
                self._running = False

    def end_server(self) -> None:
        with self._server_lock:
            if self.server is None:
                return
            if self.server.poll() is None:
                try:
                    subprocess.run(size_redis, timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    logging.getLogger(__name__).exception("Could not read Redis size")
                self.server.send_signal(signal.SIGINT)
                try:
                    self.server.wait(10)
                except subprocess.TimeoutExpired:
                    self.server.kill()
                    self.server.wait(5)
            self.server = None

    @classmethod
    def plot_events(cls, graph_engine: GraphEngine) -> None:
        if graph_engine.collection_data.benchmark != cls.name():
            raise BenchmarkNotInCollectionData()
