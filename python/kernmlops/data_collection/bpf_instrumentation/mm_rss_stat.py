from dataclasses import dataclass
from pathlib import Path

import polars as pl
from bcc import BPF
from data_collection.bpf_instrumentation.bpf_hook import POLL_TIMEOUT_MS, BPFProgram
from data_schema import CollectionTable
from data_schema.rss_stat import TraceMMRSSStatDataTable


@dataclass(frozen=True)
class TraceRSSStat:
  pid: int
  tgid: int
  ts_ns: int
  file: int
  anon: int
  swap: int
  shmem: int

class TraceRSSStatBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "mm_rss_stat"

  def __init__(self):
    self.is_support_raw_tp = True #  BPF.support_raw_tracepoint()
    self.bpf_text = open(Path(__file__).parent / "bpf/mm_trace_rss_stat.bpf.c", "r").read()
    self.trace_rss_stat = list[TraceRSSStat]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    #self.bpf.attach_raw_tracepoint(tp=b"mm_trace_rss_stat", fn_name=b"mm_trace_rss_stat")
    self.bpf["rss_stat_output"].open_perf_buffer(self._mm_trace_rss_stat_eh, page_cnt=256)

  def poll(self):
    self.bpf.perf_buffer_poll(timeout=POLL_TIMEOUT_MS)

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    return [
            TraceMMRSSStatDataTable.from_df_id(
                pl.DataFrame(self.trace_rss_stat),
                collection_id=self.collection_id,
            ),
        ]

  def clear(self):
    self.trace_rss_stat.clear()

  def pop_data(self) -> list[CollectionTable]:
    quanta_tables = self.data()
    self.clear()
    return quanta_tables

  def _mm_trace_rss_stat_eh(self, cpu, rss_stat_struct, size):
      event = self.bpf["rss_stat_output"].event(rss_stat_struct)
      self.trace_rss_stat.append(
        TraceRSSStat(
          pid=event.pid,
          tgid=event.tgid,
          ts_ns=event.ts,
          file=event.file,
          anon=event.anon,
          swap=event.swap,
          shmem=event.shmem,
        )
      )
