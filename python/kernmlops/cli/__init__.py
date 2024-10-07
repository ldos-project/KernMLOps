import sys
from pathlib import Path

import click
import data_collection
import data_import
import data_schema
from click_default_group import DefaultGroup
from kernmlops_benchmark import FauxBenchmark, benchmarks

from cli import collect


@click.group()
def cli():
    """Run kernmlops operations."""


@cli.group("collect", cls=DefaultGroup, default="data", default_if_no_args=True)
def cli_collect():
    """Collect things."""


@cli_collect.command("data")
@click.option(
    "-p",
    "--poll-rate",
    "poll_rate",
    default=.5,
    required=False,
    type=float,
)
@click.option(
    "-b",
    "--benchmark",
    "benchmark_name",
    default=FauxBenchmark.name(),
    type=click.Choice(list(benchmarks.keys())),
)
@click.option(
    "--cpus",
    "cpus",
    default=None,
    required=False,
    type=int,
    help="Used to scale workloads, defaults to number physical cores",
)
@click.option(
    "-v",
    "--verbose",
    "verbose",
    default=False,
    is_flag=True,
    type=bool,
)
@click.option(
    "--no-hooks",
    "no_hooks",
    default=False,
    is_flag=True,
    type=bool,
    help="Used as baseline for overhead of instrumentation hooks",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    default=Path("data"),
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-d",
    "--benchmark-dir",
    "benchmark_dir",
    default=Path.home() / "kernmlops-benchmark",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def cli_collect_data(
    output_dir: Path,
    benchmark_name: str,
    benchmark_dir: Path,
    cpus: int | None,
    poll_rate: float,
    no_hooks: bool,
    verbose: bool,
):
    """Run data collection tooling."""
    bpf_programs = [] if no_hooks else data_collection.bpf.hooks()
    benchmark_args = {
        "benchmark_dir": benchmark_dir,
        "cpus": cpus,
    }
    benchmark = benchmarks[benchmark_name](**benchmark_args)  # pyright: ignore [reportCallIssue]
    collect.run_collect(
        data_dir=output_dir,
        benchmark=benchmark,
        bpf_programs=bpf_programs,
        poll_rate=poll_rate,
        verbose=verbose,
    )


@cli_collect.command("dump")
@click.option(
    "-d",
    "--input-dir",
    "input_dir",
    default=Path("data/curated"),
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-b",
    "--benchmark",
    "benchmark_name",
    default=None,
    required=False,
    help="Benchmark to filter by, default is to dump all data",
    type=click.Choice(list(benchmarks.keys())),
)
def cli_collect_dump(input_dir: Path, benchmark_name: str | None):
    """Debug tool to dump collected data."""
    kernmlops_dfs = data_import.read_parquet_dir(input_dir, benchmark_name=benchmark_name)
    for name, kernmlops_df in kernmlops_dfs.items():
        print(f"{name}: {kernmlops_df}")


@cli_collect.command("graph")
@click.option(
    "-d",
    "--input-dir",
    "input_dir",
    default=Path("data/curated"),
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-c",
    "--collection-id",
    "collection_id",
    default=None,
    required=True,
    help="Collection id to filter by, can be a unique prefix",
    type=str,
)
@click.option(
    "--no-trends",
    "no_trends",
    default=False,
    is_flag=True,
    type=bool,
    help="Omit trend lines from graphs",
)
def cli_collect_graph(input_dir: Path, collection_id: str, no_trends: bool):
    """Debug tool to graph collected data."""
    collection_data = data_schema.CollectionData.from_data(
        data_dir=input_dir,
        collection_id=collection_id,
        table_types=data_schema.table_types,
    )
    collection_data.dump(no_trends=no_trends)


def main():
    try:
        # TODO(Patrick): use logging
        cli.main(prog_name="kernmlops")
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)
