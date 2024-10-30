from dataclasses import dataclass
from pathlib import Path

import polars as pl
from bcc import BPF
from data_collection.bpf_instrumentation.bpf_hook import BPFProgram
from data_schema import UPTIME_TIMESTAMP, CollectionTable
from data_schema.quanta_runtime import QuantaQueuedTable, QuantaRuntimeTable

# Note: collecting blocked time is not useful since parent processes blocking on children
# obfuscates the meaning

@dataclass(frozen=True)
class QuantaRuntimeData:
  cpu: int
  pid: int
  tgid: int
  quanta_end_uptime_us: int
  quanta_run_length_us: int

class QuantaRuntimeBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "quanta_runtime"

  def __init__(self):
    self.is_support_raw_tp = False #  BPF.support_raw_tracepoint()
    bpf_text = open(Path(__file__).parent / "bpf/sched_quanta_runtime.bpf.c", "r").read()

    # code substitutions
    if BPF.kernel_struct_has_field(b'task_struct', b'__state') == 1:
        bpf_text = bpf_text.replace('STATE_FIELD', '__state')
    else:
        bpf_text = bpf_text.replace('STATE_FIELD', 'state')
    # pid from userspace point of view is thread group from kernel pov
    # bpf_text = bpf_text.replace('FILTER', 'tgid != %s' % args.pid)
    self.bpf_text = bpf_text.replace('FILTER', '0')
    if self.is_support_raw_tp:
        bpf_text = bpf_text.replace('USE_TRACEPOINT', '1')
    else:
        bpf_text = bpf_text.replace('USE_TRACEPOINT', '0')
    self.quanta_runtime_data = list[QuantaRuntimeData]()
    self.quanta_queue_data = list[QuantaRuntimeData]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    if not self.is_support_raw_tp:
      self.bpf.attach_kprobe(event=b"ttwu_do_activate", fn_name=b"trace_ttwu_do_wakeup")
      self.bpf.attach_kprobe(event=b"wake_up_new_task", fn_name=b"trace_wake_up_new_task")
      self.bpf.attach_kprobe(
        event_re=rb'^finish_task_switch$|^finish_task_switch\.isra\.\d$',
        fn_name=b"trace_run"
      )
    self.bpf["quanta_runtimes"].open_perf_buffer(self._runtime_event_handler, page_cnt=64)
    self.bpf["quanta_queue_times"].open_perf_buffer(self._queue_event_handler, page_cnt=64)

  def poll(self):
    self.bpf.perf_buffer_poll()

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    return [
      QuantaRuntimeTable.from_df_id(
        pl.DataFrame(self.quanta_runtime_data).rename({
          "quanta_end_uptime_us": UPTIME_TIMESTAMP,
        }),
        collection_id=self.collection_id,
      ),
      QuantaQueuedTable.from_df_id(
        pl.DataFrame(self.quanta_queue_data).rename({
          "quanta_end_uptime_us": UPTIME_TIMESTAMP,
          "quanta_run_length_us": "quanta_queued_time_us",
        }),
        collection_id=self.collection_id,
      )
    ]

  def clear(self):
    self.quanta_runtime_data.clear()
    self.quanta_queue_data.clear()

  def pop_data(self) -> list[CollectionTable]:
    quanta_tables = self.data()
    self.clear()
    return quanta_tables

  def _runtime_event_handler(self, cpu, quanta_runtime_perf_event, size):
    event = self.bpf["quanta_runtimes"].event(quanta_runtime_perf_event)
    self.quanta_runtime_data.append(
      QuantaRuntimeData(
        cpu=cpu,
        pid=event.pid,
        tgid=event.tgid,
        quanta_end_uptime_us=event.quanta_end_uptime_us,
        quanta_run_length_us=event.quanta_run_length_us,
      )
    )

  def _queue_event_handler(self, cpu, quanta_runtime_perf_event, size):
    event = self.bpf["quanta_queue_times"].event(quanta_runtime_perf_event)
    self.quanta_queue_data.append(
      QuantaRuntimeData(
        cpu=cpu,
        pid=event.pid,
        tgid=event.tgid,
        quanta_end_uptime_us=event.quanta_end_uptime_us,
        quanta_run_length_us=event.quanta_run_length_us,
      )
    )
