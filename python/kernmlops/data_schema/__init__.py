"""Library for maintaining, manipulating, and using schemas."""

import os
from pwd import getpwnam
from typing import Callable

from data_schema.process_metadata import ProcessMetadataTable
from data_schema.quanta_runtime import QuantaQueuedTable, QuantaRuntimeTable
from data_schema.schema import (
    CollectionData,
    CollectionGraph,
    CollectionTable,
    SystemInfoTable,
    collection_id_column,
)

table_types = [
    SystemInfoTable,
    QuantaRuntimeTable,
    QuantaQueuedTable,
    ProcessMetadataTable,
]

def demote(user_id: int | None = None, group_id: int | None = None) -> Callable[[], None]:
    def no_op():
        pass
    if user_id is None:
        # if the user id is unspecified and the account is not privileged do nothing
        if os.getuid() >= 1000:
            return no_op
        if "UNAME" in os.environ:
            user_id = getpwnam(os.environ["UNAME"]).pw_uid
        else:
            raise Exception("not enough information to demote user")
    if group_id is None:
        group_id = int(os.environ.get("GID", user_id))

    def do_demote():
        os.setgid(group_id)
        os.setuid(user_id)
    return do_demote

__all__ = [
    "collection_id_column",
    "demote",
    "table_types",
    "CollectionTable",
    "CollectionData",
    "CollectionGraph",
    "SystemInfoTable",
]
