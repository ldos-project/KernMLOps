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
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


@dataclass(frozen=True)
class MongoDbConfig(ConfigBase):
    repeat: int = 1
    field_count: int = 256
    field_length: int = 16
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
    url: str = "mongodb://localhost:27017/"

kill_mongod = [
    "killall",
    "-9",
    "mongod",
]

class MongoDbBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "mongodb"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return MongoDbConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> "Benchmark":
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        mongodb_config = cast(MongoDbConfig, getattr(config, cls.name()))
        return MongoDbBenchmark(generic_config=generic_config, config=mongodb_config)

    def __init__(self, *, generic_config: GenericBenchmarkConfig, config: MongoDbConfig):
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
        subprocess.run(kill_mongod)

    def ping_mongodb(self, url) -> None | MongoClient:
        try:
            client = MongoClient(self.config.url)
            client.admin.command("ping")
            return client
        except ConnectionFailure:
            return None
        subprocess.run(kill_mongod)

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        if self.server is not None:
            raise BenchmarkRunningError()

        # start the MongoDB server
        start_mongod = [
            "mongod",
            "--config",
            "/etc/mongod.conf",
        ]
        self.server = subprocess.Popen(start_mongod)

        # Wait for MongoDB
        ping_mongod = self.ping_mongodb(self.config.url)
        i = 0
        while i < 10 and ping_mongod is None:
            time.sleep(1)
            ping_mongod = self.ping_mongodb(self.config.url)
            i+=1

        if ping_mongod is None:
            raise BenchmarkError("MongoDB Failed To Start")

        process: subprocess.Popen | None = None
        for i in range(self.config.repeat):
            if process is not None:
                process.wait()
                if process.returncode != 0:
                    self.process = process
                    raise BenchmarkError(f"MongoDB Run {(2 * i) - 1} Failed")

            insert_start = i * self.config.record_count
            # Load Server
            load_mongod = [
                    "python",
                    f"{self.benchmark_dir}/YCSB/bin/ycsb",
                    "load",
                    "mongodb",
                    "-s",
                    "-P",
                    f"{self.benchmark_dir}/YCSB/workloads/workloada",
                    "-p",
                    f"mongodb.url={self.config.url}",
                    "-p",
                    f"recordcount={self.config.record_count}",
                    "-p",
                    f"fieldcount={self.config.field_count}",
                    "-p",
                    f"fieldlength={self.config.field_length}",
                    "-p",
                    f"insertstart={insert_start}",
            ]

            load_mongod = subprocess.Popen(load_mongod, preexec_fn=demote())

            load_mongod.wait()
            if load_mongod.returncode != 0:
                raise BenchmarkError("Loading MongoDB Failed")

            record_count = (i + 1) * self.config.record_count

            # Run first benchmark cycle
            run_mongodb = [
                    f"{self.benchmark_dir}/YCSB/bin/ycsb",
                    "run",
                    "mongodb",
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
                    f"mongodb.url={self.config.url}",
                    "-p",
                    f"requestdistribution={self.config.request_distribution}",
                    "-p",
                    f"threadcount={self.config.thread_count}",
                    "-p",
                    f"target={self.config.target}",
                    "-p",
                    f"insertstart={insert_start}",
                    "-p",
                    f"fieldcount={self.config.field_count}",
                    "-p",
                    f"fieldlength={self.config.field_length}",
                    "-p",
                    "mongodb.writeConcern=acknowledged"
            ]
            process = subprocess.Popen(run_mongodb, preexec_fn=demote())
            if process is not None:
                process.wait()
                if process.returncode != 0:
                    self.process = process
                    raise BenchmarkError(f"MongoDB Run {2 * i} Failed")

            # Run second benchmark cycle with same parameters
            run_mongodb = [
                    f"{self.benchmark_dir}/YCSB/bin/ycsb",
                    "run",
                    "mongodb",
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
                    f"mongodb.url={self.config.url}",
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
                    "mongodb.writeConcern=acknowledged"
            ]
            process = subprocess.Popen(run_mongodb, preexec_fn=demote())

        self.process = process

    def poll(self) -> int | None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        ret = self.process.poll()
        if ret is not None:
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

        if self.server is not None:
            self.server.send_signal(signal.SIGINT)
            if self.server.wait(10) is None:
                self.server.terminate()

        self.end_server()

    def end_server(self) -> None:
        if self.server is None:
            return

        # Drop databases
        client = MongoClient(self.config.url)
        for db in client.list_databases():
            client.drop_database(db['name'])

        # Terminate server
        self.server.terminate()
        self.server = None

    @classmethod
    def plot_events(cls, graph_engine: GraphEngine) -> None:
        if graph_engine.collection_data.benchmark != cls.name():
            raise BenchmarkNotInCollectionData()
