# Contributing to KernMLOps

The KernMLOps collector is designed to make
adding new instrumentation and to decide if
the new data collected is actually useful as easy as possible.

To do this, there are 4 main base classes of the collector:
`BPFProgram`, `CollectionTable`, `CollectionGraph`, and `Benchmark`.

The definitions can be found respectively in:
`python/kernmlops/data_collection/bpf_instrumentation/bpf_hook.py`,
`python/kernmlops/data_schema/schema.py`,
`python/kernmlops/data_schema/schema.py`,
`python/kernmlops/kernmlops_benchmark/benchmark.py`.

## Adding Instrumentation

There are 2 primary steps when adding new instrumentation,
first add a new hook implementing the `BPFProgram` Protocol and then
adding a new data schema implementing the `CollectionTable` Protocol.

### Adding a Hook

Add a new hook implementing the `BPFProgram` Protocol
under `python/kernmlops/data_collection/bpf_instrumentation`
in the style already present.

The definition of the `BPFProgram` Protocol can be found in
`python/kernmlops/data_collection/bpf_instrumentation/bpf_hook.py`.

This likely entails adding a new BPF hook, it is recommended to
put as much C code as possible under
`python/kernmlops/data_collection/bpf_instrumentation/bpf`
so that it is still readable without any modifications made to it
by the python code.

Before this new `BPFProgram` can return any data, a new data schema
implementing the `CollectionTable` Protocol must be created.

Once the hook has been created, it must be added to the public list of
hooks under
`python/kernmlops/data_collection/bpf_instrumentation/__init__.py`
in the style there.

#### Adding a Perf Hook

There is a special perf hook already defined that is used to gather
all perf counters.

This `TLBPerfBPFHook` class handles most of the boilerplate for adding
new perf counters.

Here, all perf counters are enabled on all cpus and each cpu has its own
group_fd for all of its perf counters, allowing an easy interface for
disabling/enabling the counters.

First, run `python python/kernmlops collect perf-list` to get a list of
all a machine's local perf hardware counters.

Then identify the desired counter and note its name and umask that
correspond to the data required.  These will differ from machine to
machine and this process may need to be repeated if a new machine
does not have any of the counters already identified.

Once the name and umask are in hand, add a simple metadata class to
`python/kernmlops/data_collection/bpf_instrumentation/tlb_perf_hook.py`
implementing the `CustomHWEvent` Protocol.  Multiple `CustomHWEventID`
can be returned as a list to support multiple machines.

### Adding a Data Schema

Add a new data schema by implementing the `CollectionTable` Protocol.

The definition of the `CollectionTable` Protocol can be found in
`python/kernmlops/data_schema/schema.py`.

The Protocol is very simple and primarily needs the final DataFrame
schema specified.  Other parts of the Protocol can be copied from
existing examples and are not complicated.

It should be noted that most `CollectionTable` instances should contain
a timestamp in microseconds maintaining the time since system uptime.
The constant for the name of this field is `UPTIME_TIMESTAMP`, also
found in `python/kernmlops/data_schema/schema.py`.

If data does not contain a timestamp since it is collected only once,
it may make more sense to add it to the `SystemInfoTable`.

Once the schema has been created, it must be added to the public list of
schemas under `python/kernmlops/data_schema/__init__.py` in the
style there.

#### Adding a Perf Hook Data Schema

Like in instrumentation, there is a class `PerfCollectionTable` that handles
most of the boilerplate for adding new perf counter schemas.

The definition of the `PerfCollectionTable` can be found in
`python/kernmlops/data_schema/schema.py`.

Note that these schemas should be added to a second list of public schemas
under `python/kernmlops/data_schema/__init__.py` as well.

## Adding Graphs

The primary reason behind adding graphs to KernMLOps is for testing
whether or not new instrumentation data is useful.

That is, if a graph can show some pattern or interesting characteristic
in one or more benchmarks, it may be useful for machine learning.

These graphs can also be used to sanity check and verify that instrumentation
is doing what it is expected to.

In the `CollectionTable` Protocol, there is a function that returns a list
of graphs that will use and plot its data.

A graph can be made for a `CollectionTable` by implementing the `CollectionGraph`
found in `python/kernmlops/data_schema/schema.py`.

### Adding Graphs for Perf Counters

Rate and CDF graphs are already written for perf counter data and can be
trivially enabled by implementing and setting proper names.

See `RatePerfGraph`, `CumulativePerfGraph` in
`python/kernmlops/data_schema/schema.py`.

## Adding Benchmarks

The main components to adding a benchmark are: defining the configuration
for a benchmark, setting up the benchmark, launching it, and checking for
termination.

Add a new benchmark by implementing `Benchmark` in
`python/kernmlops/kernmlops_benchmark/benchmark.py`.

Once the benchmark has been created, it must be added to the public list of
benchmarks under `python/kernmlops/kernmlops_benchmark/__init__.py` in the
style there.
