"""Main entry point for the algorithmic-trading project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from tabulate import tabulate

from src.data_loader import download_market_data


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Read and validate the YAML configuration file."""

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml must contain a YAML dictionary."
        )

    return config


def parse_arguments() -> argparse.Namespace:
    """Read optional commands entered after 'python main.py'."""

    parser = argparse.ArgumentParser(
        description="Download and prepare ETF market data."
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download fresh files instead of using cached CSV files.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the market-data preparation stage."""

    arguments = parse_arguments()
    config = load_config()

    _, summary = download_market_data(
        config=config,
        force_refresh=arguments.refresh,
    )

    print("\nDownload summary")
    print("----------------")

    print(
        tabulate(
            summary,
            headers="keys",
            tablefmt="github",
            showindex=False,
        )
    )

    print("\nMarket data is ready inside data/raw/.")


if __name__ == "__main__":
    main()