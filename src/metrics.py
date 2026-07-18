"""Performance metrics, benchmark and combined-portfolio calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def load_portfolio_history(
    file_path: str | Path,
) -> pd.DataFrame:
    """Load one portfolio-history CSV."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Portfolio history was not found: {path}"
        )

    history = pd.read_csv(
        path,
        parse_dates=["Date"],
        index_col="Date",
    )

    if "Portfolio value" not in history.columns:
        raise ValueError(
            f"{path} does not contain a "
            "'Portfolio value' column."
        )

    return history.sort_index()


def calculate_order_commission(
    shares: int,
    config: dict[str, Any],
) -> float:
    """Calculate the benchmark's opening commission."""

    if shares <= 0:
        return 0.0

    per_share = float(
        config["costs"]["commission_per_share"]
    )

    minimum = float(
        config["costs"]["minimum_commission"]
    )

    return max(
        minimum,
        shares * per_share,
    )


def create_spy_benchmark(
    spy_data: pd.DataFrame,
    comparison_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    risk_free_returns: pd.Series,
) -> pd.Series:
    """
    Simulate buying SPY and holding it.

    Any cash remaining after purchasing whole shares earns the
    same risk-free return used by the strategies.
    """

    initial_capital = float(
        config["portfolio"]["initial_capital"]
    )

    slippage_rate = (
        float(config["costs"]["slippage_bps"])
        / 10_000
    )

    aligned_spy = spy_data.reindex(
        comparison_dates
    ).dropna(
        subset=["Open", "Close"]
    )

    if aligned_spy.empty:
        raise ValueError(
            "SPY data could not be aligned "
            "with the backtest dates."
        )

    first_date = aligned_spy.index[0]

    theoretical_entry_price = float(
        aligned_spy.at[first_date, "Open"]
    )

    entry_price = (
        theoretical_entry_price
        * (1 + slippage_rate)
    )

    shares = int(
        initial_capital
        // entry_price
    )

    while shares > 0:
        opening_commission = (
            calculate_order_commission(
                shares,
                config,
            )
        )

        total_cost = (
            shares * entry_price
            + opening_commission
        )

        if total_cost <= initial_capital:
            break

        shares -= 1

    if shares <= 0:
        raise ValueError(
            "Initial capital is insufficient to buy SPY."
        )

    remaining_cash = (
        initial_capital
        - total_cost
    )

    aligned_risk_free = (
        risk_free_returns
        .reindex(aligned_spy.index)
        .fillna(0.0)
    )

    cash_path = (
        remaining_cash
        * (1 + aligned_risk_free).cumprod()
    )

    benchmark = (
        cash_path
        + shares * aligned_spy["Close"]
    )

    benchmark.name = "SPY Buy & Hold"

    return benchmark


def create_combined_portfolio(
    momentum: pd.Series,
    mean_reversion: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """Combine both strategies using the configured weights."""

    momentum_weight = float(
        config["combination"]["momentum_weight"]
    )

    mean_reversion_weight = float(
        config["combination"][
            "mean_reversion_weight"
        ]
    )

    total_weight = (
        momentum_weight
        + mean_reversion_weight
    )

    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            "The strategy weights must add to 1."
        )

    initial_capital = float(
        config["portfolio"]["initial_capital"]
    )

    momentum_normalised = (
        momentum
        / float(momentum.iloc[0])
    )

    mean_reversion_normalised = (
        mean_reversion
        / float(mean_reversion.iloc[0])
    )

    combined = initial_capital * (
        momentum_weight * momentum_normalised
        + mean_reversion_weight
        * mean_reversion_normalised
    )

    combined.name = "50/50 Combined"

    return combined


def calculate_drawdown(
    portfolio_values: pd.Series,
) -> pd.Series:
    """Calculate percentage declines from previous peaks."""

    running_peak = portfolio_values.cummax()

    return (
        portfolio_values
        / running_peak
        - 1
    )


def calculate_performance_metrics(
    portfolio_values: pd.Series,
    risk_free_returns: pd.Series,
) -> dict[str, float]:
    """Calculate return and risk metrics."""

    values = portfolio_values.dropna()

    if len(values) < 2:
        raise ValueError(
            "At least two portfolio observations are required."
        )

    daily_returns = (
        values
        .pct_change()
        .dropna()
    )

    aligned_risk_free = (
        risk_free_returns
        .reindex(daily_returns.index)
        .fillna(0.0)
    )

    excess_returns = (
        daily_returns
        - aligned_risk_free
    )

    initial_value = float(
        values.iloc[0]
    )

    final_value = float(
        values.iloc[-1]
    )

    total_return = (
        final_value
        / initial_value
        - 1
    )

    number_of_days = (
        values.index[-1]
        - values.index[0]
    ).days

    years = (
        number_of_days
        / 365.25
    )

    if years > 0 and final_value > 0:
        annualised_return = (
            final_value
            / initial_value
        ) ** (1 / years) - 1
    else:
        annualised_return = np.nan

    daily_volatility = (
        daily_returns.std(ddof=1)
    )

    annualised_volatility = (
        daily_volatility
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    excess_volatility = (
        excess_returns.std(ddof=1)
    )

    if excess_volatility > 0:
        sharpe_ratio = (
            excess_returns.mean()
            / excess_volatility
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe_ratio = np.nan

    negative_returns = daily_returns[
        daily_returns < 0
    ]

    downside_deviation = (
        negative_returns.std(ddof=1)
    )

    if (
        len(negative_returns) > 1
        and downside_deviation > 0
    ):
        sortino_ratio = (
            daily_returns.mean()
            / downside_deviation
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sortino_ratio = np.nan

    maximum_drawdown = float(
        calculate_drawdown(values).min()
    )

    positive_day_rate = float(
        (daily_returns > 0).mean()
    )

    return {
        "Initial value": initial_value,
        "Final value": final_value,
        "Total return": total_return,
        "Annualised return": annualised_return,
        "Annualised volatility":
            annualised_volatility,
        "Sharpe ratio": sharpe_ratio,
        "Sortino ratio": sortino_ratio,
        "Maximum drawdown": maximum_drawdown,
        "Positive day rate": positive_day_rate,
    }


def load_trade_win_rate(
    trade_file: str | Path,
) -> tuple[int, float]:
    """Return trade count and profitable-trade percentage."""

    path = Path(trade_file)

    if not path.exists():
        return 0, np.nan

    trades = pd.read_csv(path)

    if trades.empty or "Net PnL" not in trades.columns:
        return 0, np.nan

    trade_count = len(trades)

    win_rate = float(
        (trades["Net PnL"] > 0).mean()
    )

    return trade_count, win_rate


def build_performance_report(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    risk_free_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the portfolio comparison and metrics table."""

    momentum_history = load_portfolio_history(
        "outputs/momentum_portfolio.csv"
    )

    mean_reversion_history = load_portfolio_history(
        "outputs/mean_reversion_portfolio.csv"
    )

    comparison_dates = (
        momentum_history.index.intersection(
            mean_reversion_history.index
        )
    )

    momentum_values = momentum_history.loc[
        comparison_dates,
        "Portfolio value",
    ]

    mean_reversion_values = (
        mean_reversion_history.loc[
            comparison_dates,
            "Portfolio value",
        ]
    )

    combined_values = create_combined_portfolio(
        momentum=momentum_values,
        mean_reversion=mean_reversion_values,
        config=config,
    )

    benchmark_values = create_spy_benchmark(
        spy_data=frames[
            config["data"]["benchmark"]
        ],
        comparison_dates=comparison_dates,
        config=config,
        risk_free_returns=risk_free_returns,
    )

    comparison = pd.concat(
        [
            momentum_values.rename("Momentum"),
            mean_reversion_values.rename(
                "Mean Reversion"
            ),
            combined_values,
            benchmark_values,
        ],
        axis=1,
        join="inner",
    ).dropna()

    comparison.to_csv(
        "outputs/comparison_portfolios.csv"
    )

    momentum_trades, momentum_win_rate = (
        load_trade_win_rate(
            "outputs/momentum_trades.csv"
        )
    )

    mean_reversion_trades, mean_reversion_win_rate = (
        load_trade_win_rate(
            "outputs/mean_reversion_trades.csv"
        )
    )

    trade_information = {
        "Momentum": (
            momentum_trades,
            momentum_win_rate,
        ),
        "Mean Reversion": (
            mean_reversion_trades,
            mean_reversion_win_rate,
        ),
        "50/50 Combined": (
            np.nan,
            np.nan,
        ),
        "SPY Buy & Hold": (
            1,
            np.nan,
        ),
    }

    metric_rows = []

    for name in comparison.columns:
        row = {
            "Portfolio": name,
            **calculate_performance_metrics(
                portfolio_values=comparison[name],
                risk_free_returns=risk_free_returns,
            ),
        }

        (
            row["Trades"],
            row["Trade win rate"],
        ) = trade_information[name]

        metric_rows.append(row)

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics.to_csv(
        "outputs/performance_metrics.csv",
        index=False,
    )

    return comparison, metrics