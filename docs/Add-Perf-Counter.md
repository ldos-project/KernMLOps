# Add Perf Counter

## Search for Desired Counter

Enter the docker container and ask libperf to give information on available counters:

```shell
make docker
python python/kernmlops collect perf-list
```

Sample output:

```shell
...
#-----------------------------
IDX : 650117146
PMU name : bdx_unc_r3qpi1 (Intel BroadwellX R3QPI1 uncore)
Name : UNC_R3_VNA_CREDITS_ACQUIRED
Equiv : None
Flags : None
Desc : Number of QPI VNA Credit acquisitions. This event <...>
Code : 51
Umask-00 : 1 : PMU : AD : None : VNA credit Acquisitions -- HOM Message Class
Umask-01 : 4 : PMU : BL : None : VNA credit Acquisitions -- HOM Message Class
Modif-00 : 0x00 : PMU : [e] : edge detect (boolean)
Modif-01 : 0x01 : PMU : [i] : invert (boolean)
Modif-02 : 0x02 : PMU : [t] : threshold in range [0-255] (integer)

#-----------------------------
IDX : 650117147
PMU name : bdx_unc_r3qpi1 (Intel BroadwellX R3QPI1 uncore)
Name : UNC_R3_VNA_CREDITS_REJECT
Equiv : None
Flags : None
Desc : Number of attempted VNA credit acquisitions <...>
Code : 52
Umask-00 : 8 : PMU : DRS : None : VNA Credit Reject -- DRS Message Class
Umask-01 : 1 : PMU : HOM : None : VNA Credit Reject -- HOM Message Class
Umask-02 : 16 : PMU : NCB : None : VNA Credit Reject -- NCB Message Class
Umask-03 : 32 : PMU : NCS : None : VNA Credit Reject -- NCS Message Class
Umask-04 : 4 : PMU : NDR : None : VNA Credit Reject -- NDR Message Class
Umask-05 : 2 : PMU : SNP : None : VNA Credit Reject -- SNP Message Class
Modif-00 : 0x00 : PMU : [e] : edge detect (boolean)
...

```

## Adding your own perf counter to capture

Let's go through the steps of adding counters for resource stall information.
The metadata for related counters as found through the above process is:

```shell
IDX : 421527602
PMU name : bdw_ep (Intel Broadwell EP)
Name : RESOURCE_STALLS
Equiv : None
Flags : None
Desc : Cycles Allocation is stalled due to Resource Related reason
Code : 162
Umask-00 : 1 : PMU : ANY : [default] : Cycles Allocation is stalled due to Resource Related reason
Umask-01 : 1 : PMU : ALL : None : Alias to ANY
Umask-02 : 4 : PMU : RS : None : Stall cycles caused by absence of eligible entries in Reservation Station (RS)
Umask-03 : 8 : PMU : SB : None : Cycles Allocator is stalled due to Store Buffer full (not including draining from synch)
Umask-04 : 16 : PMU : ROB : None : ROB full stall cycles
Modif-00 : 0x00 : PMU : [k] : monitor at priv level 0 (boolean)
Modif-01 : 0x01 : PMU : [u] : monitor at priv level 1, 2, 3 (boolean)
Modif-02 : 0x02 : PMU : [e] : edge level (may require counter-mask >= 1) (boolean)
Modif-03 : 0x03 : PMU : [i] : invert (boolean)
Modif-04 : 0x04 : PMU : [c] : counter-mask in range [0-255] (integer)
Modif-05 : 0x05 : PMU : [t] : measure any thread (boolean)
Modif-06 : 0x07 : PMU : [intx] : monitor only inside transactional memory region (boolean)
Modif-07 : 0x08 : PMU : [intxcp] : do not count occurrences inside aborted transactional memory region (boolean)
```

So create your own file in `python/kernmlops/data_schema/perf/`
Here we will make one called `resource_stalls.py`

```shell
vi python/kernmlops/data_schema/perf/resource_stalls.py
```

Copy in the following and change anything in angled brackets
to what you want it to be in your code:

```python3
import polars as pl
from bcc import PerfType
from data_schema.perf.perf_schema import (
    CustomHWEventID,
    PerfCollectionTable,
    PerfHWCacheConfig,
)
from data_schema.schema import CollectionGraph

class <New Class Prefix>PerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "<Intended Column Name>" # cosmetic

    @classmethod
    def ev_type(cls) -> int:
        return PerfType.RAW

    @classmethod
    def ev_config(cls) -> int:
        return 0

    @classmethod
    def hw_ids(cls) -> list[CustomHWEventID]:
        return [
            CustomHWEventID(
                name="<'Name' in above perf list entry>",
                umask="<UMask description in above perf list entry ex: ANY/ALL>"
            ),
        ]

    @classmethod
    def component_name(cls) -> str:
        return "<Desired Component Name>" # cosmetic

    @classmethod
    def measured_event_name(cls) -> str:
        return "<Desired Measured Event Name>" # cosmetic

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "<New Class Prefix>PerfTable":
        return <New Class Prefix>PerfTable(
            table=table.cast(cls.schema(), strict=True)
        )  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []
```

For the example we have been following we used:

| String to Replace | Replacement |
| ------ | -----|
| `<New Class Prefix>` | `ResourceStall` |
| `<New File Name>` | `resource_stalls` |
| `<Intended Column Name>` | `resource_stalls` |
| `<'Name' in above perf list entry>` | `RESOURCE_STALLS` |
| `<UMask description in above perf list entry ex: ANY/ALL>` | `ANY`|
| `<Desired Component Name>` | `Re-Order Buffer` |
| `<Desired Measured Event Name` | `Stalls` |

Finally, add the new perf collection table to `perf_table_types`
in `python/kernmlops/data_schema/perf/__init__.py`:

```python3
from data_schema.perf.<New File Name> import <New Class Prefix>PerfTable

perf_table_types: Mapping[str, type[PerfCollectionTable]] = {
    DTLBPerfTable.name(): DTLBPerfTable,
    ITLBPerfTable.name(): ITLBPerfTable,
    TLBFlushPerfTable.name(): TLBFlushPerfTable,
    <New Class Prefix>PerfTable.name(): <New Class Prefix>PerfTable,
}
```

## Check to see if it worked

Go through the [../README.md](Readme's) section on data collection to python.
See if a key on the dictionary of dataframes exists
and has the same name as your `<Intended Column Name>`
