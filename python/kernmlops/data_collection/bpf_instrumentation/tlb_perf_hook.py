from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl
from bcc import BPF, PerfType
from data_schema import CollectionTable, TLBPerfTable

from data_collection.bpf_instrumentation.bpf_hook import BPFProgram


class PerfHWCacheConfig:
  # From perf_hw_cache_id in uapi/linux/perf_event.h
  class Cache(Enum):
    PERF_COUNT_HW_CACHE_L1D = 0
    PERF_COUNT_HW_CACHE_L1I = 1
    PERF_COUNT_HW_CACHE_LL = 2
    PERF_COUNT_HW_CACHE_DTLB = 3
    PERF_COUNT_HW_CACHE_ITLB = 4
    PERF_COUNT_HW_CACHE_BPU = 5
    PERF_COUNT_HW_CACHE_NODE = 6

  # From perf_hw_cache_op_id in uapi/linux/perf_event.h
  class Op(Enum):
    PERF_COUNT_HW_CACHE_OP_READ = 0
    PERF_COUNT_HW_CACHE_OP_WRITE = 1
    PERF_COUNT_HW_CACHE_OP_PREFETCH = 2

  # From perf_hw_cache_op_result_id  in uapi/linux/perf_event.h
  class Result(Enum):
    PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
    PERF_COUNT_HW_CACHE_RESULT_MISS = 1

  # From https://man7.org/linux/man-pages/man2/perf_event_open.2.html
  @classmethod
  def config(cls, cache: Cache, op: Op, result: Result) -> int:
    return (cache.value) | (op.value << 8) | (result.value << 16)



@dataclass(frozen=True)
class TLBPerfData:
  cpu: int
  ts_uptime_us: int
  cumulative_tlb_misses: int
  pmu_enabled_time_us: int
  pmu_running_time_us: int

  @classmethod
  def from_event(cls, cpu: int, event: Any):
    return TLBPerfData(
        cpu=cpu,
        ts_uptime_us=event.ts_uptime_us,
        cumulative_tlb_misses=event.count,
        pmu_enabled_time_us=event.enabled_time_us,
        pmu_running_time_us=event.running_time_us,
    )


class TLBPerfBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "tlb_perf"

  def __init__(self):
    self.bpf_text = open(Path(__file__).parent / "bpf/tlb_perf.bpf.c", "r").read()

    self.dtlb_perf_data = list[TLBPerfData]()
    self.itlb_perf_data = list[TLBPerfData]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    # sample frequency is in hertz
    self.bpf.attach_perf_event(
      ev_type=PerfType.HW_CACHE,
      ev_config=PerfHWCacheConfig.config(
        cache=PerfHWCacheConfig.Cache.PERF_COUNT_HW_CACHE_DTLB,
        op=PerfHWCacheConfig.Op.PERF_COUNT_HW_CACHE_OP_READ,
        result=PerfHWCacheConfig.Result.PERF_COUNT_HW_CACHE_RESULT_MISS,
      ),
      fn_name=b"on_dtlb_cache_miss",
      sample_freq=1000,
    )
    self.bpf.attach_perf_event(
      ev_type=PerfType.HW_CACHE,
      ev_config=PerfHWCacheConfig.config(
        cache=PerfHWCacheConfig.Cache.PERF_COUNT_HW_CACHE_ITLB,
        op=PerfHWCacheConfig.Op.PERF_COUNT_HW_CACHE_OP_READ,
        result=PerfHWCacheConfig.Result.PERF_COUNT_HW_CACHE_RESULT_MISS,
      ),
      fn_name=b"on_itlb_cache_miss",
      sample_freq=1000,
    )
    self.bpf["dtlb_misses"].open_perf_buffer(self._dtlb_misses_handler, page_cnt=64)
    self.bpf["itlb_misses"].open_perf_buffer(self._itlb_misses_handler, page_cnt=64)

  def poll(self):
    self.bpf.perf_buffer_poll()

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    dtlb_df = pl.DataFrame(self.dtlb_perf_data).with_columns(pl.lit(True).alias("dtlb_event"), pl.lit(False).alias("itlb_event"))
    itlb_df = pl.DataFrame(self.itlb_perf_data).with_columns(pl.lit(False).alias("dtlb_event"), pl.lit(True).alias("itlb_event"))
    return [
      TLBPerfTable.from_df_id(
        pl.concat([dtlb_df, itlb_df]),
        collection_id=self.collection_id,
      ),
    ]

  def clear(self):
    self.dtlb_perf_data.clear()
    self.itlb_perf_data.clear()

  def pop_data(self) -> list[CollectionTable]:
    tlb_miss_tables = self.data()
    self.clear()
    return tlb_miss_tables

  def _dtlb_misses_handler(self, cpu, tlb_perf_event, size):
    event = self.bpf["dtlb_misses"].event(tlb_perf_event)
    try:
        self.dtlb_perf_data.append(TLBPerfData.from_event(cpu, event))
    except Exception as _:
       pass

  def _itlb_misses_handler(self, cpu, tlb_perf_event, size):
    event = self.bpf["itlb_misses"].event(tlb_perf_event)
    try:
        self.itlb_perf_data.append(TLBPerfData.from_event(cpu, event))
    except Exception as _:
       pass
