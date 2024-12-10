from dataclasses import dataclass
from pathlib import Path

import polars as pl
from bcc import BPF
from data_collection.bpf_instrumentation.bpf_hook import POLL_TIMEOUT_MS, BPFProgram
from data_schema import CollectionTable
from data_schema.huge_pages import (
  CollapseHugePageDataTable,
  TraceMMCollapseHugePageDataTable,
  TraceMMKhugepagedScanPMDDataTable,
)


@dataclass(frozen=True)
class TraceMMKhugepagedScanPMDRuntimeData:
  pid: int
  tgid: int
  start_ts_ns: int
  end_ts_ns: int
  mm: str
  page: str
  writeable: int
  referenced: int
  none_or_zero: int
  status: int
  unmapped: bool

@dataclass(frozen=True)
class CollapseHugePageRuntimeData:
  pid: int
  tgid: int
  start_ts_ns: int
  end_ts_ns: int
  mm: str
  referenced: int
  address: str
  unmapped: int
  referenced: int
  cc: str

@dataclass(frozen=True)
class TraceMMCollapseHugePageRuntimeData:
  pid: int
  tgid: int
  start_ts_ns: int
  end_ts_ns: int
  mm: str
  isolated: bool
  status: int

class CollapseHugePageBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "collapse_huge_pages"

  def __init__(self):
    self.is_support_raw_tp = True #  BPF.support_raw_tracepoint()
    self.bpf_text = open(Path(__file__).parent / "bpf/collapse_huge_page.bpf.c", "r").read()
    self.collapse_huge_pages = list[CollapseHugePageRuntimeData]()
    self.trace_mm_collapse_huge_pages = list[TraceMMCollapseHugePageRuntimeData]()
    self.trace_mm_khugepaged_scan_pmds = list[TraceMMKhugepagedScanPMDRuntimeData]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    #self.bpf.attach_raw_tracepoint(tp=b"mm_collapse_huge_page", fn_name=b"mm_collapse_huge_page")
    self.bpf.attach_kprobe(event=b"collapse_huge_page", fn_name=b"kprobe_collapse_huge_page")
    self.bpf["collapse_huge_pages"].open_perf_buffer(self._collapse_huge_pages_eh, page_cnt=64)
    self.bpf["trace_mm_collapse_huge_pages"].open_perf_buffer(self._trace_huge_pages_eh, page_cnt=64)
    self.bpf["trace_mm_khugepaged_scan_pmds"].open_perf_buffer(self._trace_khugepaged_scan_eh, page_cnt=64)

  def poll(self):
    self.bpf.perf_buffer_poll(timeout=POLL_TIMEOUT_MS)

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    return [
            CollapseHugePageDataTable.from_df_id(
                pl.DataFrame(self.collapse_huge_pages),
                collection_id = self.collection_id,),
            TraceMMCollapseHugePageDataTable.from_df_id(
                pl.DataFrame(self.trace_mm_collapse_huge_pages),
                collection_id = self.collection_id,),
            TraceMMKhugepagedScanPMDDataTable.from_df_id(
                pl.DataFrame(self.trace_mm_khugepaged_scan_pmds),
                collection_id = self.collection_id,),
        ]

  def clear(self):
    self.collapse_huge_pages.clear()
    self.trace_mm_collapse_huge_pages.clear()
    self.trace_mm_khugepaged_scan_pmds.clear()

  def pop_data(self) -> list[CollectionTable]:
    quanta_tables = self.data()
    self.clear()
    return quanta_tables

  def _trace_khugepaged_scan_eh(self, cpu, trace_mm_khugepaged_scan_pmd_struct, size):
      event = self.bpf["trace_mm_khugepaged_scan_pmds"].event(trace_mm_khugepaged_scan_pmd_struct)
      self.trace_mm_khugepaged_scan_pmds.append(
        TraceMMKhugepagedScanPMDRuntimeData(
          pid=event.pid,
          tgid=event.tgid,
          start_ts_ns=event.start_ts_ns,
          end_ts_ns=event.end_ts_ns,
          mm=hex(event.mm),
          page=hex(event.page),
          writeable=event.writeable,
          referenced=event.referenced,
          none_or_zero=event.none_or_zero,
          status=event.status,
          unmapped=event.unmapped,
          )
        )


  def _trace_huge_pages_eh(self, cpu, trace_mm_collapse_huge_page_struct, size):
    event = self.bpf["trace_mm_collapse_huge_pages"].event(trace_mm_collapse_huge_page_struct)
    self.trace_mm_collapse_huge_pages.append(
      TraceMMCollapseHugePageRuntimeData(
          pid=event.pid,
          tgid=event.tgid,
          start_ts_ns=event.start_ts_ns,
          end_ts_ns=event.end_ts_ns,
          mm=hex(event.mm),
          isolated=event.isolated,
          status=event.status,
      )
    )

  def _collapse_huge_pages_eh(self, cpu, collapse_huge_page_struct, size):
    print("Collapse")
    event = self.bpf["collapse_huge_pages"].event(collapse_huge_page_struct)
    self.collapse_huge_pages.append(
      CollapseHugePageRuntimeData(
          pid=event.pid,
          tgid=event.tgid,
          start_ts_ns=event.start_ts_ns,
          end_ts_ns=event.end_ts_ns,
          mm=hex(event.mm),
          address=hex(event.address),
          referenced=event.referenced,
          unmapped=event.unmapped,
          cc=hex(event.cc),
      )
    )
