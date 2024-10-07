import os
import signal
from dataclasses import dataclass, fields
from functools import cache
from typing import Any, Mapping

# TODO(Patrick): experiment without osquery
import osquery
import osquery.extensions
import polars as pl
from data_schema import CollectionTable
from data_schema.process_metadata import ProcessMetadataTable
from osquery.extensions.ttypes import ExtensionStatus

from data_collection.bpf_instrumentation.bpf_hook import BPFProgram


@dataclass(frozen=True)
class ProcessMetadata:
  pid: int
  name: str
  cmdline: str
  start_time: int
  parent: int
  nice: int
  cgroup_path: str

class ProcessMetadataHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "process_metadata"

  @classmethod
  @cache
  def _select_columns(cls) -> list[str]:
    return [field.name for field in fields(ProcessMetadata)]

  @classmethod
  @cache
  def _query_select_columns(cls) -> str:
    return ", ".join(cls._select_columns())

  def __init__(self):
    self.collector_pid = os.getpid()
    self.process_metadata = list[Mapping[str, Any]]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.osquery_instance = osquery.SpawnInstance()
    self.osquery_instance.open()
    self.osquery_client = self.osquery_instance.client

    initial_processes_query = self.osquery_client.query(
      f"SELECT {self._query_select_columns()} FROM processes"
    )
    # TODO(Patrick): Add error handling
    assert isinstance(initial_processes_query.status, ExtensionStatus)
    assert initial_processes_query.status.code == 0
    assert isinstance(initial_processes_query.response, list)
    self.process_metadata = initial_processes_query.response

  def poll(self):
    new_processes_query = self.osquery_client.query(
      f"SELECT {self._query_select_columns()} FROM processes WHERE pid > {self.collector_pid}"
    )
    # TODO(Patrick): Add error handling
    assert isinstance(new_processes_query.status, ExtensionStatus)
    assert new_processes_query.status.code == 0
    assert isinstance(new_processes_query.response, list)
    self.process_metadata.extend(new_processes_query.response)

  def close(self):
    self.osquery_instance.instance.send_signal(signal.SIGINT)  # pyright: ignore [reportOptionalMemberAccess]
    self.osquery_instance.instance.wait()  # pyright: ignore [reportOptionalMemberAccess]

  def data(self) -> list[CollectionTable]:
    return [
      ProcessMetadataTable.from_df_id(
        pl.DataFrame(
          self.process_metadata
        ).unique(
          "pid"
        ).cast({
          "pid": pl.Int64(),
          "start_time": pl.Int64(),
          "parent": pl.Int64(),
          "nice": pl.Int64(),
        }).rename({
          "parent": "parent_pid",
          "start_time": "start_time_unix_sec",
        }),
        collection_id=self.collection_id,
      )
    ]

  def clear(self):
    self.process_metadata.clear()

  def pop_data(self) -> list[CollectionTable]:
    process_table = self.data()
    self.clear()
    return process_table
