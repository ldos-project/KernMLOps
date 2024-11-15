"""Abstract definition of a benchmark."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from data_schema import GraphEngine
from kernmlops_config import ConfigBase
from typing_extensions import Protocol


@dataclass(frozen=True)
class GenericBenchmarkConfig(ConfigBase):
  benchmark: str = "faux"
  benchmark_dir: str = ""
  cpus: int = 0
  skip_clear_page_cache: bool = False
  transparent_hugepages: Literal["always", "madvise", "never"] = "always"

  def get_benchmark_dir(self) -> Path:
    if self.benchmark_dir:
      return Path(self.benchmark_dir)
    elif "UNAME" in os.environ:
      return Path(f"/home/{os.environ['UNAME']}/kernmlops-benchmark")
    return Path.home() / "kernmlops-benchmark"

  def generic_setup(self):
    """This will set pertinent generic settings, should be called after specific benchmark setup."""
    if not self.skip_clear_page_cache:
      subprocess.check_call(
          ["bash", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
          stdout=subprocess.DEVNULL,
      )
    subprocess.check_call(
        [
          "bash",
          "-c",
          f"echo {self.transparent_hugepages} > /sys/kernel/mm/transparent_hugepage/enabled",
        ],
        stdout=subprocess.DEVNULL,
    )


@dataclass(frozen=True)
class FauxBenchmarkConfig(ConfigBase):
  pass


class Benchmark(Protocol):
  """Runnable benchmark that terminates naturally in finite time."""

  @classmethod
  def name(cls) -> str: ...

  @classmethod
  def default_config(cls) -> ConfigBase: ...

  @classmethod
  def from_config(cls, config: ConfigBase) -> "Benchmark": ...

  def is_configured(self) -> bool:
    """Returns True if the environment has been setup to run the benchmark."""
    ...

  def setup(self) -> None: ...

  def run(self) -> None: ...

  def poll(self) -> int | None:
    """Returns None when benchmark is running, nonzero when crashed."""
    ...

  def wait(self) -> None: ...

  def kill(self) -> None: ...

  @classmethod
  def plot_events(cls, graph_engine: GraphEngine) -> None:
    """Given a collection of data, plot important events for this benchmark."""
    ...


class FauxBenchmark(Benchmark):
  """Benchmark that does nothing and allows users to collect the running system's data."""

  @classmethod
  def name(cls) -> str:
    return "faux"

  @classmethod
  def default_config(cls) -> ConfigBase:
    return FauxBenchmarkConfig()

  @classmethod
  def from_config(cls, config: ConfigBase) -> "Benchmark":
    generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
    faux_config = cast(FauxBenchmarkConfig, getattr(config, cls.name()))
    return FauxBenchmark(generic_config=generic_config, config=faux_config)

  def __init__(self, *, generic_config: GenericBenchmarkConfig, config: FauxBenchmarkConfig):
    self.generic_config = generic_config
    self.config = config

  def is_configured(self) -> bool:
    return True

  def setup(self) -> None:
    self.generic_config.generic_setup()

  def run(self) -> None:
    print("\nHit Ctrl+C to terminate...")

  def poll(self) -> int | None:
    """This benchmark will never self terminate."""
    return None

  def wait(self) -> None:
    pass

  def kill(self) -> None:
    pass

  @classmethod
  def plot_events(cls, graph_engine: GraphEngine) -> None:
    pass
