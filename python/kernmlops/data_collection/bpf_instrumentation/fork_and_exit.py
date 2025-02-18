from dataclasses import dataclass
from pathlib import Path

import polars as pl
from bcc import BPF
from data_collection.bpf_instrumentation.bpf_hook import POLL_TIMEOUT_MS, BPFProgram
from data_schema import CollectionTable
from data_schema.process_trace import ProcessTraceDataTable


@dataclass(frozen=True)
class TraceProcessStat:
  pid: int
  tgid: int
  ts_ns: int
  name: str
  cap_type: str

class TraceProcessHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "process_trace"

  def __init__(self):
    self.bpf_text = open(Path(__file__).parent / "bpf/fork_and_exit.bpf.c", "r").read()
    self.trace_process = list[TraceProcessStat]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    self.bpf.attach_kretprobe(event=b"copy_process", fn_name=b"kretprobe_copy_process")
    self.bpf.attach_kprobe(event=b"release_task", fn_name=b"kprobe_release_task")
    self.bpf.attach_kretprobe(event=b"__set_task_comm", fn_name=b"kretprobe_exec")
    self.bpf["copy_task_events"].open_perf_buffer(self._create_task_eh, page_cnt=128)
    self.bpf["release_task_events"].open_perf_buffer(self._release_task_eh, page_cnt=128)
    self.bpf["exec_events"].open_perf_buffer(self._exec_eh, page_cnt=128)

  def poll(self):
    self.bpf.perf_buffer_poll(timeout=POLL_TIMEOUT_MS)

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    return [
            ProcessTraceDataTable.from_df_id(
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

  def _create_task_eh(self, cpu, start_data, size):
      event = self.bpf["copy_task_events"].event(start_data)
      self.trace_process.append(
        TraceProcessStat(
          pid=event.pid,
          tgid=event.tgid,
          ts_ns=event.ts,
          name=event.buff.decode("ascii"),
          cap_type="start"
        )
      )

  def _release_task_eh(self, cpu, start_data, size):
      event = self.bpf["release_task_events"].event(start_data)
      self.trace_process.append(
        TraceProcessStat(
          pid=event.pid,
          tgid=event.tgid,
          ts_ns=event.ts,
          name="",
          cap_type="end"
        )
      )

  def _exec_eh(self, cpu, start_data, size):
      event = self.bpf["exec_events"].event(start_data)
      self.trace_process.append(
        TraceProcessStat(
          pid=event.pid,
          tgid=event.tgid,
          ts_ns=event.ts,
          name=event.buff.decode("ascii"),
          cap_type="exec"
        )
      )
