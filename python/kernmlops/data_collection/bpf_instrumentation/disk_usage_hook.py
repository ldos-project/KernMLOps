from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import polars as pl
from data_collection.bpf_instrumentation.bpf_hook import BPFProgram
from data_schema import CollectionTable
from data_schema.disk_usage import DiskUsageTable


def num_sectors_to_kb(num_sectors: int, sector_size: int = 512) -> int:
  return int(num_sectors * sector_size / 1024)

@dataclass(frozen=True)
class DiskUsageData:
  ts_uptime_us: int

  num_sector_read: int
  num_sector_write: int

  @classmethod
  def from_procfs_map(cls, ts_uptime_us: int, procfs_map: Mapping[str, int]) -> DiskUsageData:
    return DiskUsageData(
      ts_uptime_us=ts_uptime_us,
      num_sector_read=procfs_map.get("sda")[0],
      num_sector_write=procfs_map.get("sda")[1],
    )


@dataclass(frozen=True)
class DiskUsageDataRaw:
  ts_uptime_us: int
  procfs_dump: str

  def parse(self) -> DiskUsageData:
    procfs_lines = [
      [e.lstrip().rstrip() for e in line.split()]
      for line in self.procfs_dump.splitlines()
    ]
    procfs_map = {
      # name: num_sector_read, num_sector_write
      line[2]: (num_sectors_to_kb(int(line[5])), num_sectors_to_kb(int(line[9])))
      for line in procfs_lines
    }
    return DiskUsageData.from_procfs_map(self.ts_uptime_us, procfs_map)



class DiskUsageHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "disk_usage"

  @classmethod
  def _procfs_file(cls) -> Path:
    return Path("/proc/diskstats")

  def __init__(self):
    pass

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.disk_usage = list[DiskUsageDataRaw]()

  def poll(self):
    self.disk_usage.append(
      DiskUsageDataRaw(
        ts_uptime_us=int(time.clock_gettime_ns(time.CLOCK_BOOTTIME) / 1000),
        procfs_dump=self._procfs_file().open("r").read(),
      )
    )

  def close(self):
    pass

  def data(self) -> list[CollectionTable]:
    return [
      DiskUsageTable.from_df_id(
        pl.DataFrame([
          raw_data.parse()
          for raw_data in self.disk_usage
        ]),
        collection_id=self.collection_id,
      )
    ]

  def clear(self):
    self.disk_usage.clear()

  def pop_data(self) -> list[CollectionTable]:
    disk_table = self.data()
    self.clear()
    return disk_table
