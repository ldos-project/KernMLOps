from dataclasses import dataclass
from pathlib import Path

import polars as pl
from bcc import BPF
from data_collection.bpf_instrumentation.bpf_hook import POLL_TIMEOUT_MS, BPFProgram
from data_schema import CollectionTable
from data_schema.zswap_runtime import ZswapRuntimeDataTable


@dataclass(frozen=True)
class ZswapRuntimeStat:
  pid: int
  tgid: int
  start_ts: int
  end_ts: int
  name: str

class ZswapRuntimeBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "zswap_runtime"

  def __init__(self):
    self.bpf_text = open(Path(__file__).parent / "bpf/zswap_runtime.bpf.c", "r").read()
    self.trace_process = list[ZswapRuntimeStat]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    self.bpf.attach_kprobe(event=b"zswap_store", fn_name=b"trace_zswap_store_entry")
    self.bpf.attach_kretprobe(event=b"zswap_store", fn_name=b"trace_zswap_store_return")
    self.bpf.attach_kprobe(event=b"zswap_load", fn_name=b"trace_zswap_load_entry")
    self.bpf.attach_kretprobe(event=b"zswap_load", fn_name=b"trace_zswap_load_return")
    self.bpf.attach_kprobe(event=b"zswap_invalidate", fn_name=b"trace_zswap_invalidate_entry")
    self.bpf.attach_kretprobe(event=b"zswap_invalidate", fn_name=b"trace_zswap_invalidate_return")
    self.bpf["zswap_store_events"].open_perf_buffer(self._zswap_store_eh, page_cnt=128)
    self.bpf["zswap_load_events"].open_perf_buffer(self._zswap_load_eh, page_cnt=128)
    self.bpf["zswap_invalidate_events"].open_perf_buffer(self._zswap_invalidate_eh, page_cnt=128)

  def poll(self):
    self.bpf.perf_buffer_poll(timeout=POLL_TIMEOUT_MS)

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    return [
            ZswapRuntimeDataTable.from_df_id(
                pl.DataFrame(self.trace_process),
                collection_id=self.collection_id,
            ),
        ]

  def clear(self):
    self.trace_process.clear()

  def pop_data(self) -> list[CollectionTable]:
    tables = self.data()
    self.clear()
    return tables

  def _zswap_store_eh(self, cpu, start_data, size):
      event = self.bpf["zswap_store_events"].event(start_data)
      self.trace_process.append(
        ZswapRuntimeStat(
          pid=event.pid,
          tgid=event.tgid,
          start_ts=event.start_ts,
          end_ts=event.end_ts,
          name="zswap_store"
        )
      )

  def _zswap_load_eh(self, cpu, start_data, size):
      event = self.bpf["zswap_load_events"].event(start_data)
      self.trace_process.append(
        ZswapRuntimeStat(
          pid=event.pid,
          tgid=event.tgid,
          start_ts=event.start_ts,
          end_ts=event.end_ts,
          name="zswap_load"
        )
      )

  def _zswap_invalidate_eh(self, cpu, start_data, size):
      event = self.bpf["zswap_invalidate_events"].event(start_data)
      self.trace_process.append(
        ZswapRuntimeStat(
          pid=event.pid,
          tgid=event.tgid,
          start_ts=event.start_ts,
          end_ts=event.end_ts,
          name="zswap_invalidate"
        )
      )
