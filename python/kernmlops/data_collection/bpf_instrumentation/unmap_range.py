from dataclasses import dataclass
from pathlib import Path

import polars as pl
from bcc import BPF
from data_collection.bpf_instrumentation.bpf_hook import POLL_TIMEOUT_MS, BPFProgram
from data_schema import CollectionTable
from data_schema.generic_table import UnmapRangeDataTable


@dataclass(frozen=True)
class UnmapRangeStat:
  tgid: int
  ts_ns: int
  start: int
  end: int
  is_huge: bool

class UnmapRangeBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "unmap_range"

  def __init__(self):
    self.is_support_raw_tp = True #  BPF.support_raw_tracepoint()
    self.bpf_text = open(Path(__file__).parent / "bpf/unmap_range.bpf.c", "r").read()
    self.unmap_range_stat = list[UnmapRangeStat]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    self.bpf.attach_kprobe(event=b"unmap_page_range", fn_name=b"kprobe__unmap_page_range")
    self.bpf.attach_kprobe(event=b"__unmap_hugepage_range", fn_name=b"kprobe__unmap_hugepage_range")
    self.bpf["unmap_range_output"].open_perf_buffer(self._unmap_range_eh, page_cnt=64)

  def poll(self):
    self.bpf.perf_buffer_poll(timeout=POLL_TIMEOUT_MS)

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    return [
            UnmapRangeDataTable.from_df_id(
                pl.DataFrame(self.unmap_range_stat),
                collection_id=self.collection_id,
            ),
        ]

  def clear(self):
    self.unmap_range_stat.clear()

  def pop_data(self) -> list[CollectionTable]:
    tables = self.data()
    self.clear()
    return tables

  def _unmap_range_eh(self, cpu, unmap_range_struct, size):
      event = self.bpf["unmap_range_output"].event(unmap_range_struct)
      self.unmap_range_stat.append(
        UnmapRangeStat(
          tgid=event.tgid,
          ts_ns=event.ts_ns,
          start=event.start,
          end=event.end,
          is_huge=False if event.huge == 0 else True,
        )
      )
