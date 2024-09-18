"""Abstract definition of a BPF program."""

import polars as pl
from typing_extensions import Protocol


class BPFProgram(Protocol):
  """Loadable BPF program that returns performance data."""

  def load(self) -> None: ...

  def poll(self) -> None: ...

  def data(self) -> pl.DataFrame: ...

  def clear(self): ...

  def pop_data(self) -> pl.DataFrame: ...
