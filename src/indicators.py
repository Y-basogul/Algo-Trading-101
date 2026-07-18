"""Technical-indicator calculations used by the trading strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

INDICATOR_COLUMNS = [
    "Momentum_Fast_MA",
    "Momentum_Slow_MA",
    "Trend_MA",
    "RSI",
    "ZScore",
    "ATR",
]


def _validate_price_data(data: pd.DataFrame) -> None:
    """Check that the required price columns exist."""

    missing_columns = [
        column for column in PRICE_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Price data is missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("Price data is empty.")


def moving_average(
    prices: pd.Series,
    period: int,
) -> pd.Series:
    """Calculate a simple moving average."""

    if period <= 0:
        raise ValueError("Moving-average period must be positive.")

    return prices.rolling(
        window=period,
        min_periods=period,
    ).mean()


def relative_strength_index(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate RSI using Wilder-style exponential smoothing.

    RSI varies between 0 and 100 and compares recent gains with
    recent losses.
    """

    if period <= 0:
        raise ValueError("RSI period must be positive.")

    price_change = prices.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))

    # Deal with periods containing only gains or only losses.
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100.0,
    )

    rsi = rsi.mask(
        (average_gain == 0) & (average_loss > 0),
        0.0,
    )

    return rsi


def true_range(data: pd.DataFrame) -> pd.Series:
    """
    Calculate each day's true range.

    True range considers:
    1. Today's high minus today's low.
    2. Today's high versus yesterday's close.
    3. Today's low versus yesterday's close.
    """

    previous_close = data["Close"].shift(1)

    high_low = data["High"] - data["Low"]
    high_previous_close = (data["High"] - previous_close).abs()
    low_previous_close = (data["Low"] - previous_close).abs()

    ranges = pd.concat(
        [
            high_low,
            high_previous_close,
            low_previous_close,
        ],
        axis=1,
    )

    return ranges.max(axis=1)


def average_true_range(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range using Wilder-style smoothing.

    ATR measures volatility, not price direction.
    """

    if period <= 0:
        raise ValueError("ATR period must be positive.")

    daily_true_range = true_range(data)

    return daily_true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def rolling_zscore(
    prices: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the rolling z-score, mean and standard deviation.

    Z-score measures how far the current price is from its recent
    average in standard-deviation units.
    """

    if period <= 1:
        raise ValueError("Z-score period must be greater than one.")

    rolling_mean = prices.rolling(
        window=period,
        min_periods=period,
    ).mean()

    rolling_standard_deviation = prices.rolling(
        window=period,
        min_periods=period,
    ).std(ddof=0)

    usable_standard_deviation = rolling_standard_deviation.replace(
        0,
        np.nan,
    )

    zscore = (
        prices - rolling_mean
    ) / usable_standard_deviation

    return zscore, rolling_mean, rolling_standard_deviation


def add_indicators(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Add every required indicator to one price DataFrame."""

    _validate_price_data(data)

    result = data.copy()

    momentum_config = config["momentum"]
    mean_reversion_config = config["mean_reversion"]
    risk_config = config["risk"]

    fast_period = int(
        momentum_config["fast_moving_average"]
    )

    slow_period = int(
        momentum_config["slow_moving_average"]
    )

    trend_period = int(
        mean_reversion_config["trend_moving_average"]
    )

    rsi_period = int(
        mean_reversion_config["rsi_period"]
    )

    zscore_period = int(
        mean_reversion_config["zscore_period"]
    )

    atr_period = int(
        risk_config["atr_period"]
    )

    result["Momentum_Fast_MA"] = moving_average(
        result["Close"],
        fast_period,
    )

    result["Momentum_Slow_MA"] = moving_average(
        result["Close"],
        slow_period,
    )

    result["Trend_MA"] = moving_average(
        result["Close"],
        trend_period,
    )

    result["RSI"] = relative_strength_index(
        result["Close"],
        rsi_period,
    )

    (
        result["ZScore"],
        result["Bollinger_Middle"],
        rolling_standard_deviation,
    ) = rolling_zscore(
        result["Close"],
        zscore_period,
    )

    result["Bollinger_Upper"] = (
        result["Bollinger_Middle"]
        + 2 * rolling_standard_deviation
    )

    result["Bollinger_Lower"] = (
        result["Bollinger_Middle"]
        - 2 * rolling_standard_deviation
    )

    result["ATR"] = average_true_range(
        result,
        atr_period,
    )

    return result


def prepare_indicator_data(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Calculate indicators for every symbol and save processed CSVs.

    Returns:
        processed_frames:
            Dictionary containing the enriched DataFrames.

        summary:
            Table showing when indicators become available.
    """

    processed_directory = Path(
        config["data"]["processed_directory"]
    )

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for symbol, frame in frames.items():
        processed_frame = add_indicators(
            data=frame,
            config=config,
        )

        output_path = processed_directory / f"{symbol}.csv"
        processed_frame.to_csv(output_path)

        processed_frames[symbol] = processed_frame

        complete_indicator_rows = processed_frame.dropna(
            subset=INDICATOR_COLUMNS
        )

        if complete_indicator_rows.empty:
            first_ready_date = None
            ready_rows = 0
        else:
            first_ready_date = (
                complete_indicator_rows.index.min().date()
            )
            ready_rows = len(complete_indicator_rows)

        summary_rows.append(
            {
                "Symbol": symbol,
                "Total rows": len(processed_frame),
                "Indicator-ready rows": ready_rows,
                "First ready date": first_ready_date,
                "Saved file": str(output_path),
            }
        )

        print(f"Calculated indicators for {symbol}.")
        print(f"  Saved processed data to {output_path}")

    summary = pd.DataFrame(summary_rows)

    return processed_frames, summary