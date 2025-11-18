"""Abstract definition of a BPF program."""


from data_schema import CollectionTable
from typing_extensions import Final, Protocol

POLL_TIMEOUT_MS: Final[int] = 5


class BPFProgram(Protocol):
  """Loadable BPF program that returns performance data."""

  @classmethod
  def name(cls) -> str: ...

  def load(self, collection_id: str) -> None: ...

  def poll(self) -> None: ...

  def close(self) -> None: ...

  def data(self) -> list[CollectionTable]: ...

  # def last_k_ms(self, ms: int) -> list[CollectionTable]: ...

  # def last_k_data(self, k: int) -> list[CollectionTable]:
  #   return [
  #     table.limit(k)
  #     for table in self.data()
  #   ]

  # def last_data(self) -> list[CollectionTable]:
  #   return self.last_data(1)

  def clear(self): ...

  def pop_data(self) -> list[CollectionTable]: ...
