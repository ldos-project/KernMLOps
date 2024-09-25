"""Abstract definition of a benchmark."""

from typing_extensions import Protocol

# TODO(Patrick): Add flush page-cache


class Benchmark(Protocol):
  """Runnable benchmark that terminates naturally in finite time."""

  @classmethod
  def name(cls) -> str: ...

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


class FauxBenchmark(Benchmark):
  """Benchmark that does nothing and allows users to collect the running system's data."""

  @classmethod
  def name(cls) -> str:
    return "faux"

  def is_configured(self) -> bool:
    return True

  def setup(self) -> None:
    pass

  def run(self) -> None:
    pass

  def poll(self) -> int | None:
    """This benchmark will never self terminate."""
    return None

  def wait(self) -> None:
    pass

  def kill(self) -> None:
    pass
