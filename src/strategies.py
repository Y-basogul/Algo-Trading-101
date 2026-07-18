"""Trading-signal rules for momentum and mean-reversion strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_INDICATOR_COLUMNS = [
    "Close",
    "Momentum_Fast_MA",
    "Momentum_Slow_MA",
    "Trend_MA",
    "RSI",
    "ZScore",
    "ATR",
]


def _validate_indicator_data(data: pd.DataFrame) -> None:
    """Confirm that all indicators needed by the strategies exist."""

    missing_columns = [
        column
        for column in REQUIRED_INDICATOR_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Indicator data is missing required columns: "
            f"{missing_columns}"
        )

    if data.empty:
        raise ValueError("Indicator data is empty.")


def add_momentum_signals(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Add momentum regime, buy and exit signals.

    Long regime:
        Fast moving average is above the slow moving average.

    Buy:
        The strategy moves from outside the long regime into it.

    Exit:
        The strategy moves from the long regime back outside it.
    """

    result = data.copy()

    valid_indicators = (
        result["Momentum_Fast_MA"].notna()
        & result["Momentum_Slow_MA"].notna()
    )

    long_regime = (
        valid_indicators
        & (
            result["Momentum_Fast_MA"]
            > result["Momentum_Slow_MA"]
        )
    )

    previous_long_regime = long_regime.shift(
        periods=1,
        fill_value=False,
    )

    result["Momentum_Regime"] = long_regime.astype(int)

    result["Momentum_Buy_Signal"] = (
        long_regime
        & ~previous_long_regime
    )

    result["Momentum_Exit_Signal"] = (
        ~long_regime
        & previous_long_regime
    )

    return result


def add_mean_reversion_signals(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Add long-only mean-reversion buy and exit signals.

    Buy condition:
        Price is above its long-term trend average.
        RSI is below the configured entry threshold.
        Z-score is below the configured entry threshold.

    Exit condition:
        Z-score returns to its recent average,
        or RSI rises above its configured exit threshold.

    ATR stops and maximum holding days are applied later by the
    backtester because they depend on the actual trade entry.
    """

    result = data.copy()

    mean_reversion_config = config["mean_reversion"]

    rsi_entry = float(
        mean_reversion_config["rsi_entry"]
    )

    rsi_exit = float(
        mean_reversion_config["rsi_exit"]
    )

    zscore_entry = float(
        mean_reversion_config["zscore_entry"]
    )

    zscore_exit = float(
        mean_reversion_config["zscore_exit"]
    )

    valid_indicators = (
        result["Trend_MA"].notna()
        & result["RSI"].notna()
        & result["ZScore"].notna()
    )

    trend_filter = (
        valid_indicators
        & (result["Close"] > result["Trend_MA"])
    )

    entry_condition = (
        trend_filter
        & (result["RSI"] < rsi_entry)
        & (result["ZScore"] < zscore_entry)
    )

    previous_entry_condition = entry_condition.shift(
        periods=1,
        fill_value=False,
    )

    # A buy signal appears when the complete entry condition
    # becomes true after previously being false.
    buy_signal = (
        entry_condition
        & ~previous_entry_condition
    )

    exit_condition = (
        valid_indicators
        & (
            (result["ZScore"] >= zscore_exit)
            | (result["RSI"] >= rsi_exit)
        )
    )

    result["MeanReversion_Trend_Filter"] = (
        trend_filter.astype(int)
    )

    result["MeanReversion_Entry_Condition"] = (
        entry_condition
    )

    result["MeanReversion_Buy_Signal"] = buy_signal

    result["MeanReversion_Exit_Signal"] = (
        exit_condition
    )

    return result


def add_strategy_signals(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Add momentum and mean-reversion signals to one DataFrame."""

    _validate_indicator_data(data)

    result = add_momentum_signals(
        data=data,
        config=config,
    )

    result = add_mean_reversion_signals(
        data=result,
        config=config,
    )

    return result


def _first_signal_date(
    data: pd.DataFrame,
    column: str,
):
    """Return the first date on which a Boolean signal is true."""

    signal_rows = data.index[data[column].fillna(False)]

    if len(signal_rows) == 0:
        return None

    return signal_rows.min().date()


def prepare_strategy_data(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Calculate strategy signals for every symbol.

    The processed CSV files are overwritten with new versions that
    now include both indicators and strategy signals.
    """

    processed_directory = Path(
        config["data"]["processed_directory"]
    )

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    strategy_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for symbol, frame in frames.items():
        strategy_frame = add_strategy_signals(
            data=frame,
            config=config,
        )

        output_path = processed_directory / f"{symbol}.csv"

        strategy_frame.to_csv(output_path)

        strategy_frames[symbol] = strategy_frame

        summary_rows.append(
            {
                "Symbol": symbol,
                "Momentum buys": int(
                    strategy_frame[
                        "Momentum_Buy_Signal"
                    ].sum()
                ),
                "Momentum exits": int(
                    strategy_frame[
                        "Momentum_Exit_Signal"
                    ].sum()
                ),
                "Mean-reversion buys": int(
                    strategy_frame[
                        "MeanReversion_Buy_Signal"
                    ].sum()
                ),
                "First momentum buy": _first_signal_date(
                    strategy_frame,
                    "Momentum_Buy_Signal",
                ),
                "First mean-reversion buy": _first_signal_date(
                    strategy_frame,
                    "MeanReversion_Buy_Signal",
                ),
            }
        )

        print(f"Generated strategy signals for {symbol}.")
        print(f"  Updated processed file: {output_path}")

    summary = pd.DataFrame(summary_rows)

    return strategy_frames, summary