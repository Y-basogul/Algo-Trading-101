"""Main entry point for the algorithmic-trading project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from tabulate import tabulate

from src.backtester import run_all_backtests
from src.data_loader import download_market_data
from src.indicators import prepare_indicator_data
from src.metrics import build_performance_report
from src.plots import create_all_plots
from src.risk_free import prepare_risk_free_data
from src.strategies import prepare_strategy_data


def load_config(
    config_path: str = "config.yaml",
) -> dict[str, Any]:
    """Read and validate the YAML configuration."""

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
    """Read optional terminal arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete ETF trading backtest."
        )
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Download fresh Yahoo and FRED data "
            "instead of using cached CSV files."
        ),
    )

    return parser.parse_args()


def print_table(
    title: str,
    table: Any,
) -> None:
    """Print a readable terminal table."""

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


def format_basic_backtest_table(
    table,
):
    """Format the simple backtest results."""

    display = table.copy()

    display["Total return"] = (
        display["Total return"]
        .map(lambda value: f"{value:.2%}")
    )

    display["Win rate"] = (
        display["Win rate"]
        .map(lambda value: f"{value:.2%}")
    )

    for column in [
        "Initial capital",
        "Final value",
        "Cash interest earned",
    ]:
        display[column] = (
            display[column]
            .map(lambda value: f"${value:,.2f}")
        )

    return display


def format_metrics_table(
    table,
):
    """Format percentages, ratios and currency."""

    display = table.copy()

    currency_columns = [
        "Initial value",
        "Final value",
    ]

    percentage_columns = [
        "Total return",
        "Annualised return",
        "Annualised volatility",
        "Maximum drawdown",
        "Positive day rate",
        "Trade win rate",
    ]

    ratio_columns = [
        "Sharpe ratio",
        "Sortino ratio",
    ]

    for column in currency_columns:
        display[column] = display[column].map(
            lambda value: f"${value:,.2f}"
        )

    for column in percentage_columns:
        display[column] = display[column].map(
            lambda value: (
                "N/A"
                if value != value
                else f"{value:.2%}"
            )
        )

    for column in ratio_columns:
        display[column] = display[column].map(
            lambda value: (
                "N/A"
                if value != value
                else f"{value:.2f}"
            )
        )

    display["Trades"] = display["Trades"].map(
        lambda value: (
            "N/A"
            if value != value
            else str(int(value))
        )
    )

    return display


def format_risk_free_table(
    table,
):
    """Format the risk-free-data summary."""

    display = table.copy()

    display["Average annual yield"] = (
        display["Average annual yield"]
        .map(lambda value: f"{value:.2%}")
    )

    return display


def main() -> None:
    """Run the complete project pipeline."""

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

    strategy_frames, strategy_summary = (
        prepare_strategy_data(
            frames=indicator_frames,
            config=config,
        )
    )

    print_table(
        title="Strategy-signal summary",
        table=strategy_summary,
    )

    risk_free_returns, risk_free_summary = (
        prepare_risk_free_data(
            config=config,
            frames=strategy_frames,
            force_refresh=arguments.refresh,
        )
    )

    print_table(
        title="Risk-free-rate summary",
        table=format_risk_free_table(
            risk_free_summary
        ),
    )

    backtest_summary = run_all_backtests(
        frames=strategy_frames,
        config=config,
        risk_free_returns=risk_free_returns,
    )

    print_table(
        title="Basic backtest summary",
        table=format_basic_backtest_table(
            backtest_summary
        ),
    )

    comparison, performance_metrics = (
        build_performance_report(
            frames=strategy_frames,
            config=config,
            risk_free_returns=risk_free_returns,
        )
    )

    print_table(
        title="Complete performance comparison",
        table=format_metrics_table(
            performance_metrics
        ),
    )

    create_all_plots(
        comparison=comparison,
    )

    print(
        "\nPipeline completed successfully."
        "\nRaw data:          data/raw/"
        "\nProcessed data:    data/processed/"
        "\nResults and plots: outputs/"
    )


if __name__ == "__main__":
    main()