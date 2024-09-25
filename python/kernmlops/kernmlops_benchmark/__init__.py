from typing import Mapping

from kernmlops_benchmark.benchmark import Benchmark, FauxBenchmark
from kernmlops_benchmark.errors import (
    BenchmarkError,
    BenchmarkNotConfiguredError,
    BenchmarkNotRunningError,
    BenchmarkRunningError,
)
from kernmlops_benchmark.linux_build import LinuxBuildBenchmark

benchmarks: Mapping[str, type[Benchmark]] = {
    FauxBenchmark.name(): FauxBenchmark,
    LinuxBuildBenchmark.name(): LinuxBuildBenchmark,
}

__all__ = [
    "Benchmark",
    "BenchmarkError",
    "BenchmarkRunningError",
    "BenchmarkNotConfiguredError",
    "BenchmarkNotRunningError",
    "FauxBenchmark",
    "LinuxBuildBenchmark",
    "benchmarks",
]
