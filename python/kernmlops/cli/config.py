from dataclasses import field, make_dataclass

from kernmlops_benchmark import BenchmarkConfig
from kernmlops_config import ConfigBase

KernmlopsConfig = make_dataclass(
    cls_name="KernmlopsConfig",
    bases=(ConfigBase,),
    fields=[
        (
            "benchmark_config",
            BenchmarkConfig,
            field(default=BenchmarkConfig()),
        )
    ],
    frozen=True,
)


__all__ = [
  "KernmlopsConfig",
]
