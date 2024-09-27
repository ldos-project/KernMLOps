# KernMLOps

This repository serves as the mono-repo for the KernMLOps research project.

Currently, it only contains scripts for data collection of kernel performance.

Quick Setup:

```shell
make dependencies
make hooks

make docker-image

pip install -r requirements.txt
# Installs benchmarks without root privileges
make provision-benchmarks
# If the above fails dependencies may need to be installed with
make provision-benchmarks-admin

# Ensure you have installed your kernel's development headers
# On ubuntu: apt install linux-headers-$(uname -r)
# On redhat: dnf install kernel-devel kernel-headers
# Or
# For first time installation of tooling like kernel-headers, bcc-tools, osquery
# This requires admin permissions
make provision-development

make collect
```

## Tools

### [asdf](https://asdf-vm.com)

Provides a declarative set of tools pinned to
specific versions for environmental consistency.

These tools are defined in `.tool-versions`.
Run `make dependencies` to initialize a new environment.

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

### [bcc](https://github.com/iovisor/bcc)

A toolset for building eBPF programs including python bindings.

It is highly recommended to build this from scratch so that its python package
can be pip installed to your local environment.  By default system packages will
only install it for the system python, which may complicate development for user
level package management.

Building instructions are provided [here](https://github.com/iovisor/bcc/blob/master/INSTALL.md#source)
for most major distributions.

After installing, the python package can be installed to your local environment via:

```shell
pip install -e BCC_BUILD_DIR/src/python/bcc-python3
```

Verify proper installation with:

```python
import bcc
```

### [ansible](https://www.ansible.com/)

Ansible is an automation tool for configuring VMs and baremetal machines.

### [vagrant](https://developer.hashicorp.com/vagrant/docs/installation)

Vagrant is a tool for managing VMs simply across different backend like VirtualBox
and libvirt, we recommend using libvirt.

## Dependencies

### Python

Python is required, version `3.12` is recommended for best compatibility.

Python package dependencies are listed in `requirements.txt` and can be
installed via `pip`.

They can also be install via [conda](https://docs.anaconda.com/miniconda/miniconda-install/).
If `conda` is used then it is recommended to then install `mamba` and use
that as a drop in replacement.
This can be done with `conda install -conda-forge mamba`.

Or by [poetry](https://python-poetry.org/docs/).

Then the python packages can be installed via:

```shell
conda install -c conda-forge --file requirements.txt
mamba install -c conda-forge --file requirements.txt
```

### Creating VMs

For ubuntu the requirements can be installed with:

```shell
sudo apt install -y bc flex bison gcc make libelf-dev libssl-dev \
    squashfs-tools busybox-static tree cpio curl
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
make collect-data
```

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
