"""Charts for comparing strategies with the SPY benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_equity_curve_plot(
    comparison: pd.DataFrame,
    output_directory: str | Path = "outputs",
) -> Path:
    """Plot all portfolio values on one chart."""

    output_path = Path(output_directory)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    for column in comparison.columns:
        axis.plot(
            comparison.index,
            comparison[column],
            label=column,
            linewidth=1.7,
        )

    axis.set_title(
        "Strategy Portfolios vs SPY Buy-and-Hold"
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio Value ($)")
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()

    file_path = (
        output_path
        / "equity_curve_comparison.png"
    )

    figure.savefig(
        file_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return file_path


def create_drawdown_plot(
    comparison: pd.DataFrame,
    output_directory: str | Path = "outputs",
) -> Path:
    """Plot each portfolio's peak-to-trough drawdown."""

    output_path = Path(output_directory)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    drawdowns = (
        comparison
        / comparison.cummax()
        - 1
    )

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    for column in drawdowns.columns:
        axis.plot(
            drawdowns.index,
            drawdowns[column],
            label=column,
            linewidth=1.5,
        )

    axis.set_title(
        "Portfolio Drawdowns"
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.yaxis.set_major_formatter(
        lambda value, position: f"{value:.0%}"
    )

    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()

    file_path = (
        output_path
        / "drawdown_comparison.png"
    )

    figure.savefig(
        file_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return file_path


def create_all_plots(
    comparison: pd.DataFrame,
) -> list[Path]:
    """Generate every current project chart."""

    equity_path = create_equity_curve_plot(
        comparison
    )

    drawdown_path = create_drawdown_plot(
        comparison
    )

    print(
        f"Created graph: {equity_path}"
    )

    print(
        f"Created graph: {drawdown_path}"
    )

    return [
        equity_path,
        drawdown_path,
    ]