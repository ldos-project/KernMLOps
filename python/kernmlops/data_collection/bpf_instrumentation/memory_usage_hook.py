import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import polars as pl
from data_schema import CollectionTable
from data_schema.memory_usage import MemoryUsageTable

from data_collection.bpf_instrumentation.bpf_hook import BPFProgram


# Documentation: https://access.redhat.com/solutions/406773
@dataclass(frozen=True)
class MemoryUsageData:
  ts_uptime_us: int

  mem_total_bytes: int
  mem_free_bytes: int
  mem_available_bytes: int
  buffers_bytes: int
  cached_bytes: int

  swap_total_bytes: int
  swap_free_bytes: int

  # data waiting to be written back to disk
  dirty_bytes: int
  writeback_bytes: int

  anon_pages_total_bytes: int
  anon_hugepages_total_bytes: int
  mapped_total_bytes: int
  shmem_total_bytes: int

  hugepages_total: int
  hugepages_free: int
  hugepages_reserved: int
  hugepage_size_bytes: int

  hardware_corrupted_bytes: int

  @classmethod
  def from_procfs_map(cls, ts_uptime_us: int, procfs_map: Mapping[str, int]) -> "MemoryUsageData":
    return MemoryUsageData(
      ts_uptime_us=ts_uptime_us,

      mem_total_bytes=procfs_map.get("MemTotal", 0) * 1024,
      mem_free_bytes=procfs_map.get("MemFree", 0) * 1024,
      mem_available_bytes=procfs_map.get("MemAvailable", 0) * 1024,
      buffers_bytes=procfs_map.get("Buffers", 0) * 1024,
      cached_bytes=procfs_map.get("Cached", 0) * 1024,

      swap_total_bytes=procfs_map.get("SwapTotal", 0) * 1024,
      swap_free_bytes=procfs_map.get("SwapFree", 0) * 1024,

      dirty_bytes=procfs_map.get("Dirty", 0) * 1024,
      writeback_bytes=procfs_map.get("Writeback", 0) * 1024,

      anon_pages_total_bytes=procfs_map.get("AnonPages", 0) * 1024,
      anon_hugepages_total_bytes=procfs_map.get("AnonHugePages", 0) * 1024,
      mapped_total_bytes=procfs_map.get("Mapped", 0) * 1024,
      shmem_total_bytes=procfs_map.get("Shmem", 0) * 1024,

      hugepages_total=procfs_map.get("HugePages_Total", 0) * 1024,
      hugepages_free=procfs_map.get("HugePages_Free", 0) * 1024,
      hugepages_reserved=procfs_map.get("HugePages_Rsvd", 0) * 1024,
      hugepage_size_bytes=procfs_map.get("Hugepagesize", 0) * 1024,

      hardware_corrupted_bytes=procfs_map.get("HardwareCorrupted", 0) * 1024,
    )


@dataclass(frozen=True)
class MemoryUsageDataRaw:
  ts_uptime_us: int
  procfs_dump: str

  def parse(self) -> MemoryUsageData:
    procfs_lines = [
      line.split(":", maxsplit=1)
      for line in self.procfs_dump.splitlines()
    ]
    procfs_map = {
      line[0].lstrip().rstrip(): int(line[1].removesuffix("kB").lstrip().rstrip())
      for line in procfs_lines
      if len(line) == 2
    }
    return MemoryUsageData.from_procfs_map(self.ts_uptime_us, procfs_map)



class MemoryUsageHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "memory_usage"

  @classmethod
  def _procfs_file(cls) -> Path:
    return Path("/proc/meminfo")

  def __init__(self):
    pass

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.memory_usage = list[MemoryUsageDataRaw]()

  def poll(self):
    self.memory_usage.append(
      MemoryUsageDataRaw(
        ts_uptime_us=int(time.clock_gettime_ns(time.CLOCK_BOOTTIME) / 1000),
        procfs_dump=self._procfs_file().open("r").read(),
      )
    )

  def close(self):
    pass

  def data(self) -> list[CollectionTable]:
    return [
      MemoryUsageTable.from_df_id(
        pl.DataFrame([
          raw_data.parse()
          for raw_data in self.memory_usage
        ]),
        collection_id=self.collection_id,
      )
    ]

  def clear(self):
    self.memory_usage.clear()

  def pop_data(self) -> list[CollectionTable]:
    memory_table = self.data()
    self.clear()
    return memory_table
