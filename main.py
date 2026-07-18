"""Main entry point for the algorithmic-trading project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from tabulate import tabulate

from src.data_loader import download_market_data
from src.indicators import prepare_indicator_data
from src.strategies import prepare_strategy_data


def load_config(
    config_path: str = "config.yaml",
) -> dict[str, Any]:
    """Read and validate the YAML configuration file."""

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml must contain a YAML dictionary."
        )

    return config


def parse_arguments() -> argparse.Namespace:
    """Read optional commands entered after python main.py."""

    parser = argparse.ArgumentParser(
        description=(
            "Download data, calculate indicators and "
            "generate strategy signals."
        )
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Download fresh Yahoo data instead of "
            "using cached CSV files."
        ),
    )

    return parser.parse_args()


def print_table(
    title: str,
    table,
) -> None:
    """Print a DataFrame as a readable terminal table."""

    print(f"\n{title}")
    print("-" * len(title))

    print(
        tabulate(
            table,
            headers="keys",
            tablefmt="github",
            showindex=False,
        )
    )


def main() -> None:
    """Run the data, indicator and signal pipeline."""

    arguments = parse_arguments()
    config = load_config()

    market_frames, download_summary = download_market_data(
        config=config,
        force_refresh=arguments.refresh,
    )

    print_table(
        title="Market-data summary",
        table=download_summary,
    )

    indicator_frames, indicator_summary = (
        prepare_indicator_data(
            frames=market_frames,
            config=config,
        )
    )

    print_table(
        title="Indicator summary",
        table=indicator_summary,
    )

    _, strategy_summary = prepare_strategy_data(
        frames=indicator_frames,
        config=config,
    )

    print_table(
        title="Strategy-signal summary",
        table=strategy_summary,
    )

    print(
        "\nPipeline completed successfully."
        "\nRaw data:              data/raw/"
        "\nIndicators and signals: data/processed/"
    )


if __name__ == "__main__":
    main()