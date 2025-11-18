import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import click
import data_collection
import data_import
import data_schema
import yaml
from cli import collect
from cli.config import KernmlopsConfig
from click_default_group import DefaultGroup
from data_collection import GenericCollectorConfig
from kernmlops_benchmark import benchmarks
from kernmlops_config import DEFAULT_CONFIG_FILE


@click.group()
def cli():
    """Run kernmlops operations."""


@cli.group("collect", cls=DefaultGroup, default="data", default_if_no_args=True)
def cli_collect():
    """Collect things."""


@cli_collect.command("data")
@click.option(
    "-c",
    "--config-file",
    "config_file",
    default=DEFAULT_CONFIG_FILE,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-b",
    "--benchmark",
    "benchmark_name",
    default=None,
    type=click.Choice(list(benchmarks.keys())),
    help="Used to override benchmark from config file.",
)
@click.option(
    "-v",
    "--verbose",
    "verbose",
    default=False,
    is_flag=True,
    type=bool,
)
def cli_collect_data(
    config_file: Path,
    benchmark_name: str | None,
    verbose: bool,
):
    """Run data collection tooling."""
    config_overrides = yaml.safe_load(config_file.read_text())
    config = KernmlopsConfig().merge(config_overrides)
    collector_config: GenericCollectorConfig = config.collector_config
    name = benchmark_name if benchmark_name else str(config.benchmark_config.generic.benchmark)
    benchmark = benchmarks[name].from_config(config.benchmark_config)
    collect.run_collect(
        collector_config=collector_config,
        benchmark=benchmark,
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
    "-o",
    "--output-dir",
    "output_dir",
    default=None,
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
@click.option(
    "-m",
    "--matplot",
    "use_matplot",
    default=False,
    is_flag=True,
    type=bool,
    help="Use matplotlib to graph data",
)
def cli_collect_graph(input_dir: Path, output_dir: Path | None, collection_id: str, no_trends: bool, use_matplot: bool):
    """Debug tool to graph collected data."""
    collection_data = data_schema.CollectionData.from_data(
        data_dir=input_dir,
        collection_id=collection_id,
        table_types=data_schema.table_types,
    )
    collection_data.dump(output_dir=output_dir, no_trends=no_trends, use_matplot=use_matplot)


@cli_collect.command("perf-list")
def cli_collect_perf():
    """Lists perf counter names for use in supporting new computers."""
    for _, data in data_collection.bpf.CustomHWConfigManager.hw_event_map().items():
        print(data.dump())


@cli_collect.command("defaults")
def cli_collect_defaults():
    """Output default collection config into yaml file defaults.yaml."""

    # From https://github.com/yaml/pyyaml/issues/234
    class Dumper(yaml.Dumper):
        def increase_indent(self, flow=False, *args, **kwargs):
            return super().increase_indent(flow=flow, indentless=False)

    DEFAULT_CONFIG_FILE.write_text(
        yaml.dump(
            asdict(KernmlopsConfig()),
            Dumper=Dumper,
            indent=2,
            sort_keys=False,
            explicit_start=True,
        )
    )


def main():
    try:
        # TODO(Patrick): use logging
        cli.main(prog_name="kernmlops")
    except Exception:
        print(traceback.format_exc())
        sys.exit(1)
