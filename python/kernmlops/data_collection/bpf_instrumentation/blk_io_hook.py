from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl
from bcc import BPF
from data_collection.bpf_instrumentation.bpf_hook import POLL_TIMEOUT_MS, BPFProgram
from data_schema import UPTIME_TIMESTAMP, CollectionTable
from data_schema.block_io import BlockIOLatencyTable, BlockIOQueueTable, BlockIOTable


@dataclass(frozen=True)
class BlockIOQueueData:
  cpu: int
  device: int
  sector: int
  segments: int
  block_io_bytes: int
  block_io_start_uptime_us: int
  block_io_flags: int
  queue_length_segment_ios: int
  queue_length_4k_ios: int

@dataclass(frozen=True)
class BlockIOLatencyData:
  cpu: int
  device: int
  sector: int
  segments: int
  block_io_bytes: int
  block_io_end_uptime_us: int
  block_latency_us: int
  block_io_latency_us: int
  block_io_flags: int

class BlockIOBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "block_io"

  def __init__(self):
    bpf_text = open(Path(__file__).parent / "bpf/blk_io.bpf.c", "r").read()

    # code substitutions
    if BPF.kernel_struct_has_field(b'request', b'rq_disk') == 1:
        bpf_text = bpf_text.replace('__RQ_DISK__', 'rq_disk')
    else:
        bpf_text = bpf_text.replace('__RQ_DISK__', 'q->disk')
    self.bpf_text = bpf_text
    self.block_io_queue_data = list[BlockIOQueueData]()
    self.block_io_latency_data = list[BlockIOLatencyData]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    self.bpf["block_io_starts"].open_perf_buffer(self._queue_event_handler, page_cnt=64)
    self.bpf["block_io_ends"].open_perf_buffer(self._latency_event_handler, page_cnt=64)

  def poll(self):
    self.bpf.perf_buffer_poll(timeout=POLL_TIMEOUT_MS)

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    return list[CollectionTable]([
      BlockIOTable.from_tables(
        latency_table=cast(
          BlockIOLatencyTable,
          BlockIOLatencyTable.from_df_id(
            pl.DataFrame(self.block_io_latency_data).rename({
              "block_io_end_uptime_us": UPTIME_TIMESTAMP,
            }),
            collection_id=self.collection_id,
          ),
        ),
        queue_table=cast(
          BlockIOQueueTable,
          BlockIOQueueTable.from_df_id(
            pl.DataFrame(self.block_io_queue_data).rename({
              "block_io_start_uptime_us": UPTIME_TIMESTAMP,
            }),
            collection_id=self.collection_id,
          ),
        ),
      ),
    ])

  def clear(self):
    self.block_io_queue_data.clear()
    self.block_io_latency_data.clear()

  def pop_data(self) -> list[CollectionTable]:
    block_io_tables = self.data()
    self.clear()
    return block_io_tables

  def _queue_event_handler(self, cpu, block_io_start_perf_event, size):
    event = self.bpf["block_io_starts"].event(block_io_start_perf_event)
    sector = event.sector
    if sector == 18446744073709551615: # this is -1 in a u64
      sector = -1
    self.block_io_queue_data.append(
      BlockIOQueueData(
        cpu=cpu,
        device=event.device,
        sector=sector,
        segments=event.segments,
        block_io_bytes=event.block_io_bytes,
        block_io_start_uptime_us=event.block_io_start_uptime_us,
        block_io_flags=event.block_io_flags,
        queue_length_segment_ios=event.queue_length_segments,
        queue_length_4k_ios=event.queue_length_4ks,
      )
    )

  def _latency_event_handler(self, cpu, block_io_end_perf_event, size):
    event = self.bpf["block_io_ends"].event(block_io_end_perf_event)
    sector = event.sector
    if sector == 18446744073709551615: # this is -1 in a u64
      sector = -1
    self.block_io_latency_data.append(
      BlockIOLatencyData(
        cpu=cpu,
        device=event.device,
        sector=sector,
        segments=event.segments,
        block_io_bytes=event.block_io_bytes,
        block_io_end_uptime_us=event.block_io_end_uptime_us,
        block_latency_us=event.block_latency_us,
        block_io_latency_us=event.block_io_latency_us,
        block_io_flags=event.block_io_flags,
      )
    )
