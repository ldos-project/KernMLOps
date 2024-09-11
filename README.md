# KernMLOps

This repository serves as the mono-repo for the KernMLOps research project.

Currently, it only contains scripts for data collection of kernel performance.

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
