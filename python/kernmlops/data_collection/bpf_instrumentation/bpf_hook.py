"""Abstract definition of a BPF program."""


from data_schema import CollectionTable
from typing_extensions import Protocol


class BPFProgram(Protocol):
  """Loadable BPF program that returns performance data."""

  @classmethod
  def name(cls) -> str: ...

  def load(self, collection_id: str) -> None: ...

  def poll(self) -> None: ...

  def close(self) -> None: ...

  def data(self) -> list[CollectionTable]: ...

  def clear(self): ...

  def pop_data(self) -> list[CollectionTable]: ...
