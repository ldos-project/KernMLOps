# KernMLOps

This repository serves as the mono-repo for the KernMLOps research project.

Currently, it only contains scripts for data collection of kernel performance.

See CONTRIBUTING.md for an overview of how to expand this tool.

Quick Setup:

```shell
pip install -r requirements.txt

make hooks

make docker-image

# Installs gap benchmark (default)
bash scripts/setup-benchmarks/setup-gap.sh

# Ensure you have installed your kernel's development headers
# On ubuntu: apt install linux-headers-$(uname -r)
# On redhat: dnf install kernel-devel kernel-headers

# Run default data collection inside docker until manually terminated via Ctrl+C
make docker
make collect-raw
# Run gap benchmark inside docker
make docker
make benchmark-gap
# Run yaml configured data collection inside docker
make collect
```

## Capturing Data -> Processing in Python

For this example you need to open two terminals.

In Terminal 1 navigate to your cloned version of `KernMLOps`.

```shell
make docker
make collect-raw
```

You are looking for output that looks like this:

```shell
Started benchmark faux
```

This tells you that the probes have been inserted and data collection has begun.

In Terminal 2, start the application.
You will eventually need the pid,
the terminal can get you that as shown below.

```shell
./app arg1 arg2 arg3 &
echo $!
wait
```

The result of the command should be a pid.
The pid can be used later to filter results.
When the wait call finishes the program `app` has exited.
Then in Terminal 1 press `CTRL+C`.
Now the data should be collected in ...
Now in Terminal 1 you can exit the docker container,
enter python and import the last data collection.

The data is now under `data/curated` in the `KernMLOps` directory.

You can import that to python by doing the following:
Note that you need at least Python 3.12 (as is provided in the container)
Change to the `python/kernmlops/` directory

```python3
cd python/kernmlops/
python3
>>> import data_import as di
>>> r = di.read_parquet_dir("<path-to-data-curated>")
>>> r.keys()
```

In this case `r` is a dictionary containing a dataframe per key.
If we want to explore for example the `dtlb_misses` for our program
we can do the following:

```python3
>>> dtlb_data = r['dtlb_misses']
>>> import polars as pl
>>> dtlb_data.filter(pl.col("tgid") == <pid for program>)
```

## Tools

### Python-3.12

This is here to make the minimum python version blatant.

### [pre-commit](https://pre-commit.com)

A left shifting tool to consistently run a set of checks on the code repo.
Our checks enforce syntax validations and formatting.
We encourage contributors to use pre-commit hooks.

```shell
# install all pre-commit hooks
make hooks

# run pre-commit on repo once
make pre-commit
```

### [perf](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)

Perf counters are used for low-level insights into performance.

When using a new machine it is likely the counters used will be different
from those already explicitly supported.  Developers can run
`python python/kernmlops collect perf-list` to get the names, descriptions,
and umasks of the various hardware events on the new machine. From there
developers can add the correct `name, umask` pair to an existing counter
config that already exists.

It is simplest to run the above command inside a container.

## Dependencies

### Python

Python is required, at least version `3.12` is required for its generic typing support.
This is the default version on Ubuntu 24.04.

Python package dependencies are listed in `requirements.txt` and can be
installed via:

```shell
# On some systems like Ubuntu 24.04 without a virtual environment
# `--break-system-packages` may be necessary
pip install [--break-system-packages] -r requirements.txt
```

## Contributing

Developers should verify their code passes basic standards by running:

```shell
make lint
```

Developers can automatically fix many common styling issues with:

```shell
make format
```

## Usage

Users can run data collection with:

```shell
make collect-raw
```

## Configuration

All default configuration options are shown in `defaults.yaml`, this can be generated
via `make defaults`.

To configure collection, users can create and modify a `overrides.yaml` file with
just the overrides they wish to set, i.e.:

```yaml
---
benchmark_config:
  generic:
    benchmark: gap
  gap:
    trials: 7
```

Then `make collect` or `make collect-data` will use the overrides set.

If an unknown configuration parameter is set (i.e. `benchmark_cfg`) and
error will be thrown before collection begins.

## Troubleshooting: Or How I Learned to Shoot My Foot

### eBPF Programs

eBPF Programs are statically verified when the python scripts attempt
to load them to into the kernel and that is where errors will manifest.
When a program fails to compile the error
will be the usual sort of C-compiler error. i.e.

```shell
/virtual/main.c:53:3: error: call to undeclared function '__bpf_builtin_memset';
    ISO C99 and later do not support implicit function declarations
    [-Wimplicit-function-declaration]
   53 |   __bpf_builtin_memset(&data, 0, sizeof(data));
      |   ^
1 error generated.
```

For verification errors the entire compiled bytecode will be printed,
look for something along the lines of:

```shell
invalid indirect read from stack R4 off -32+20 size 24
processed 59 insns (limit 1000000) max_states_per_insn 0
    total_states 3 peak_states 3 mark_read 3
```

#### eBPF Padding

The error:

```shell
invalid indirect read from stack R4 off -32+20 size 24
processed 59 insns (limit 1000000) max_states_per_insn 0
    total_states 3 peak_states 3 mark_read 3
```

Indicates that a read in the program is reading uninitialized memory.

That error came from:

```c
struct quanta_runtime_perf_event data;
data.pid = pid;
data.tgid = tgid;
data.quanta_end_uptime_us = ts / 1000;
data.quanta_run_length_us = delta / 1000;
quanta_runtimes.perf_submit(ctx, &data, sizeof(data));
```

The invalid read was `perf_submit` since there was extra padding in the `data` struct
that was not formally initialized.  To be as robust as possible this should be handled
with an explicit `__builtin_memset` as in:

```c
struct quanta_runtime_perf_event data;
__builtin_memset(&data, 0, sizeof(data));
data.pid = pid;
data.tgid = tgid;
data.quanta_end_uptime_us = ts / 1000;
data.quanta_run_length_us = delta / 1000;
quanta_runtimes.perf_submit(ctx, &data, sizeof(data));
```

This gives the most robust handling for multiple systems,
see [here](https://github.com/iovisor/bcc/issues/2623#issuecomment-560214481).
