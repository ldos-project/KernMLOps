from dataclasses import dataclass
from pathlib import Path

import polars as pl
from bcc import BPF
from data_collection.bpf_instrumentation.bpf_hook import BPFProgram
from data_schema import CollectionTable, FileDataTable


@dataclass(frozen=True)
class FileOpenData:
  cpu: int
  pid: int
  tgid: int
  ts_uptime_us: int
  file_inode: int
  file_size_bytes: int
  file_name: str


class FileDataBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "file_data"

  def __init__(self):
    bpf_text = open(Path(__file__).parent / "bpf/file_data.bpf.c", "r").read()

    # code substitutions
    if BPF.kernel_struct_has_field(b'renamedata', b'new_mnt_idmap') == 1:
        bpf_text = bpf_text.replace('TRACE_CREATE_1', '0')
        bpf_text = bpf_text.replace('TRACE_CREATE_2', '0')
        bpf_text = bpf_text.replace('TRACE_CREATE_3', '1')
    elif BPF.kernel_struct_has_field(b'renamedata', b'old_mnt_userns') == 1:
        bpf_text = bpf_text.replace('TRACE_CREATE_1', '0')
        bpf_text = bpf_text.replace('TRACE_CREATE_2', '1')
        bpf_text = bpf_text.replace('TRACE_CREATE_3', '0')
    else:
        bpf_text = bpf_text.replace('TRACE_CREATE_1', '1')
        bpf_text = bpf_text.replace('TRACE_CREATE_2', '0')
        bpf_text = bpf_text.replace('TRACE_CREATE_3', '0')
    # pid from userspace point of view is thread group from kernel pov
    # bpf_text = bpf_text.replace('FILTER', 'tgid != %s' % args.pid)
    self.bpf_text = bpf_text.replace('FILTER', '0')
    self.file_open_data = list[FileOpenData]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    self.bpf.attach_kprobe(event=b"vfs_create", fn_name=b"trace_create")
    self.bpf.attach_kprobe(event=b"vfs_open", fn_name=b"trace_open")
    if BPF.get_kprobe_functions(b"security_inode_create"):
        self.bpf.attach_kprobe(event=b"security_inode_create", fn_name=b"trace_security_inode_create")
    self.bpf["file_open_events"].open_perf_buffer(self._file_open_event_handler, page_cnt=64)

  def poll(self):
    self.bpf.perf_buffer_poll()

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    return [
      FileDataTable.from_df_id(
        pl.DataFrame(self.file_open_data),
        collection_id=self.collection_id,
      ),
    ]

  def clear(self):
    self.file_open_data.clear()

  def pop_data(self) -> list[CollectionTable]:
    file_tables = self.data()
    self.clear()
    return file_tables

  def _file_open_event_handler(self, cpu, file_open_perf_event, size):
    event = self.bpf["file_open_events"].event(file_open_perf_event)
    try:
        data = FileOpenData(
            cpu=cpu,
            pid=event.pid,
            tgid=event.tgid,
            ts_uptime_us=event.ts_uptime_us,
            file_inode=event.file_inode,
            file_size_bytes=event.file_size_bytes,
            file_name=event.file_name.decode('utf-8'),
        )
        self.file_open_data.append(data)
    except Exception as _:
       pass
