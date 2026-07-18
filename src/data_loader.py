"""Download and validate historical market data from Yahoo Finance."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataDownloadError(RuntimeError):
    """Raised when market data cannot be downloaded or validated."""


def _resolve_end_date(configured_end_date: str | None) -> str:
    """
    Return the configured end date.

    If no end date was entered, use today's New York date.
    Yahoo treats the end date as exclusive, so today's unfinished
    daily candle will not enter the backtest.
    """
    if configured_end_date:
        return configured_end_date

    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _clean_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Clean and validate one downloaded price DataFrame."""

    if frame.empty:
        raise DataDownloadError(f"{symbol}: Yahoo returned no data.")

    # Some yfinance versions can return two-level column names.
    if isinstance(frame.columns, pd.MultiIndex):
        column_found = False

        for level in range(frame.columns.nlevels):
            level_values = frame.columns.get_level_values(level)

            if set(REQUIRED_COLUMNS).issubset(set(level_values)):
                frame.columns = level_values
                column_found = True
                break

        if not column_found:
            raise DataDownloadError(
                f"{symbol}: Could not understand Yahoo's column structure."
            )

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    ]

    if missing_columns:
        raise DataDownloadError(
            f"{symbol}: Missing required columns: {missing_columns}"
        )

    # Keep only the fields our backtester needs.
    frame = frame[REQUIRED_COLUMNS].copy()

    # Ensure all price and volume values are numeric.
    frame = frame.apply(pd.to_numeric, errors="coerce")

    # Standardise and clean the date index.
    frame.index = pd.to_datetime(frame.index)

    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)

    frame.index.name = "Date"

    # Remove duplicate dates and incomplete rows.
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.dropna(subset=REQUIRED_COLUMNS)
    frame = frame.sort_index()

    if frame.empty:
        raise DataDownloadError(
            f"{symbol}: No valid rows remained after cleaning."
        )

    return frame


def _download_from_yahoo(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    auto_adjust: bool,
    maximum_attempts: int = 3,
) -> pd.DataFrame:
    """Download one symbol, retrying temporary failures."""

    last_error: Exception | None = None

    for attempt in range(1, maximum_attempts + 1):
        try:
            print(
                f"Downloading {symbol} "
                f"(attempt {attempt}/{maximum_attempts})..."
            )

            frame = yf.download(
                tickers=symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
                multi_level_index=False,
                timeout=30,
            )

            return _clean_frame(frame, symbol)

        except Exception as error:
            last_error = error

            if attempt < maximum_attempts:
                wait_seconds = 2**attempt
                print(
                    f"  {symbol} failed temporarily. "
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)

    raise DataDownloadError(
        f"{symbol}: Download failed after {maximum_attempts} attempts. "
        f"Last error: {last_error}"
    )


def _load_cached_file(file_path: Path, symbol: str) -> pd.DataFrame:
    """Load and validate a previously downloaded CSV file."""

    frame = pd.read_csv(
        file_path,
        index_col="Date",
        parse_dates=["Date"],
    )

    return _clean_frame(frame, symbol)


def download_market_data(
    config: dict[str, Any],
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Download the benchmark and ETF universe.

    Returns:
        frames:
            Dictionary such as {"SPY": DataFrame, "XLK": DataFrame}.

        summary:
            Table describing the data downloaded for each symbol.
    """

    data_config = config["data"]

    benchmark = data_config["benchmark"]
    tickers = data_config["tickers"]

    # Add SPY first and remove any accidental duplicates.
    symbols = list(dict.fromkeys([benchmark, *tickers]))

    start_date = data_config["start_date"]
    end_date = _resolve_end_date(data_config.get("end_date"))
    interval = data_config["interval"]
    auto_adjust = bool(data_config["auto_adjust"])

    raw_directory = Path(data_config["raw_directory"])
    raw_directory.mkdir(parents=True, exist_ok=True)

    print("\nMarket-data settings")
    print("--------------------")
    print(f"Symbols:       {', '.join(symbols)}")
    print(f"Start date:    {start_date}")
    print(f"End date:      {end_date} (exclusive)")
    print(f"Interval:      {interval}")
    print(f"Auto-adjusted: {auto_adjust}")
    print()

    frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}

    for symbol in symbols:
        file_path = raw_directory / f"{symbol}.csv"

        try:
            if file_path.exists() and not force_refresh:
                print(f"Loading cached file for {symbol}: {file_path}")
                frame = _load_cached_file(file_path, symbol)
                source = "Cached CSV"

            else:
                frame = _download_from_yahoo(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    auto_adjust=auto_adjust,
                )

                frame.to_csv(file_path)
                source = "Yahoo Finance"
                print(f"  Saved {symbol} to {file_path}")

            frames[symbol] = frame

            summary_rows.append(
                {
                    "Symbol": symbol,
                    "Rows": len(frame),
                    "First date": frame.index.min().date(),
                    "Last date": frame.index.max().date(),
                    "Source": source,
                }
            )

        except Exception as error:
            failures[symbol] = str(error)
            print(f"ERROR: {error}")

    if failures:
        failure_text = "\n".join(
            f"- {symbol}: {message}"
            for symbol, message in failures.items()
        )

        raise DataDownloadError(
            "One or more symbols could not be prepared:\n"
            f"{failure_text}"
        )

    summary = pd.DataFrame(summary_rows)

    return frames, summary