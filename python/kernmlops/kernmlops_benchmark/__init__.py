from dataclasses import field, make_dataclass
from typing import Mapping

from kernmlops_benchmark.benchmark import (
    Benchmark,
    FauxBenchmark,
    GenericBenchmarkConfig,
)
from kernmlops_benchmark.benchmark_group import BenchmarkGroup
from kernmlops_benchmark.errors import (
    BenchmarkError,
    BenchmarkNotConfiguredError,
    BenchmarkNotRunningError,
    BenchmarkRunningError,
)
from kernmlops_benchmark.gap import GapBenchmark
from kernmlops_benchmark.linnos import LinnosBenchmark
from kernmlops_benchmark.linux_build import LinuxBuildBenchmark
from kernmlops_benchmark.memcached import MemcachedBenchmark
from kernmlops_benchmark.mongodb import MongoDbBenchmark
from kernmlops_benchmark.redis import RedisBenchmark
from kernmlops_benchmark.stress_ng import StressNgBenchmark
from kernmlops_benchmark.stress_ng_benchmark_group import StressNgBenchmarkGroup
from kernmlops_benchmark.stress_ng_set import StressNgSetBenchmark
from kernmlops_benchmark.benchbase import BenchbaseBenchmark
from kernmlops_config import ConfigBase

benchmarks: Mapping[str, type[Benchmark]] = {
    FauxBenchmark.name(): FauxBenchmark,
    LinuxBuildBenchmark.name(): LinuxBuildBenchmark,
    GapBenchmark.name(): GapBenchmark,
    MongoDbBenchmark.name(): MongoDbBenchmark,
    LinnosBenchmark.name(): LinnosBenchmark,
    RedisBenchmark.name(): RedisBenchmark,
    MemcachedBenchmark.name(): MemcachedBenchmark,
    StressNgBenchmark.name(): StressNgBenchmark,
    StressNgSetBenchmark.name(): StressNgSetBenchmark,
    BenchmarkGroup.name(): BenchmarkGroup,
    StressNgBenchmarkGroup.name(): StressNgBenchmarkGroup,
    BenchbaseBenchmark.name(): BenchbaseBenchmark,
}

BenchmarkConfig = make_dataclass(
    cls_name="BenchmarkConfig",
    bases=(ConfigBase,),
    fields=[
        (
            "generic",
            GenericBenchmarkConfig,
            field(default=GenericBenchmarkConfig()),
        )
    ] + [
        (name, ConfigBase, field(default=benchmark.default_config()))
        for name, benchmark in benchmarks.items()
    ],
    frozen=True,
)


__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkError",
    "BenchmarkRunningError",
    "BenchmarkNotConfiguredError",
    "BenchmarkNotRunningError",
    "FauxBenchmark",
    "LinnosBenchmark",
    "LinuxBuildBenchmark",
    "BenchbaseBenchmark",
    "GapBenchmark",
    "RedisBenchmark",
    "MongoDbBenchmark",
    "MemcachedBenchmark",
    "benchmarks",
]
