"""Download and prepare a risk-free-rate proxy from FRED."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd


class RiskFreeDataError(RuntimeError):
    """Raised when the risk-free series cannot be prepared."""


def _clean_fred_data(
    frame: pd.DataFrame,
    series_id: str,
) -> pd.Series:
    """Clean a FRED CSV into a dated numeric yield series."""

    date_candidates = [
        "observation_date",
        "DATE",
        "Date",
    ]

    date_column = next(
        (
            column
            for column in date_candidates
            if column in frame.columns
        ),
        None,
    )

    if date_column is None:
        raise RiskFreeDataError(
            "Could not find the date column in the FRED data."
        )

    value_candidates = [
        series_id,
        "Annual yield percent",
    ]

    value_column = next(
        (
            column
            for column in value_candidates
            if column in frame.columns
        ),
        None,
    )

    if value_column is None:
        raise RiskFreeDataError(
            f"Could not find the {series_id} values "
            "in the FRED data."
        )

    cleaned = frame[
        [date_column, value_column]
    ].copy()

    cleaned[date_column] = pd.to_datetime(
        cleaned[date_column],
        errors="coerce",
    )

    cleaned[value_column] = pd.to_numeric(
        cleaned[value_column],
        errors="coerce",
    )

    cleaned = cleaned.dropna(
        subset=[date_column, value_column]
    )

    cleaned = cleaned.drop_duplicates(
        subset=[date_column],
        keep="last",
    )

    cleaned = cleaned.sort_values(
        date_column
    )

    series = cleaned.set_index(
        date_column
    )[value_column]

    series.index.name = "Date"
    series.name = "Annual yield percent"

    if series.empty:
        raise RiskFreeDataError(
            "No valid risk-free observations remained "
            "after cleaning."
        )

    return series


def download_risk_free_yield(
    config: dict[str, Any],
    force_refresh: bool = False,
) -> tuple[pd.Series, str]:
    """
    Download or load the annualised Treasury yield.

    Returns:
        yield_series:
            Annual yield values expressed in percent.

        source:
            Either FRED or Cached CSV.
    """

    risk_free_config = config["risk_free"]

    series_id = risk_free_config[
        "fred_series"
    ]

    cache_path = Path(
        risk_free_config["cache_file"]
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if cache_path.exists() and not force_refresh:
        print(
            f"Loading cached risk-free data: {cache_path}"
        )

        cached_frame = pd.read_csv(
            cache_path
        )

        yield_series = _clean_fred_data(
            cached_frame,
            series_id,
        )

        return yield_series, "Cached CSV"

    url = (
        "https://fred.stlouisfed.org/"
        f"graph/fredgraph.csv?id={series_id}"
    )

    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            print(
                f"Downloading {series_id} from FRED "
                f"(attempt {attempt}/3)..."
            )

            downloaded_frame = pd.read_csv(
                url,
                na_values=["."],
            )

            yield_series = _clean_fred_data(
                downloaded_frame,
                series_id,
            )

            yield_series.to_frame().to_csv(
                cache_path,
                index_label="Date",
            )

            print(
                f"  Saved risk-free data to {cache_path}"
            )

            return yield_series, "FRED"

        except Exception as error:
            last_error = error

            if attempt < 3:
                wait_seconds = 2**attempt

                print(
                    "  FRED download failed temporarily. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    raise RiskFreeDataError(
        "FRED risk-free data could not be downloaded. "
        f"Last error: {last_error}"
    )


def common_trading_dates(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
) -> pd.DatetimeIndex:
    """Return the dates available for every trading ETF."""

    dates = pd.DatetimeIndex(
        frames[symbols[0]].index
    )

    for symbol in symbols[1:]:
        dates = dates.intersection(
            frames[symbol].index
        )

    dates = dates.sort_values()

    if dates.empty:
        raise RiskFreeDataError(
            "No common ETF dates were available "
            "for risk-free alignment."
        )

    return dates


def build_daily_risk_free_table(
    annual_yield_percent: pd.Series,
    target_dates: pd.DatetimeIndex,
    day_count_basis: float,
) -> pd.DataFrame:
    """
    Convert annual percentage yields into period returns.

    Each trading date uses the most recently available yield from
    no later than the preceding calendar day. This prevents the
    backtest from using future or same-day unpublished information.
    """

    if day_count_basis <= 0:
        raise ValueError(
            "The day-count basis must be positive."
        )

    target_dates = pd.DatetimeIndex(
        target_dates
    ).sort_values().unique()

    lookup_dates = (
        target_dates
        - pd.Timedelta(days=1)
    )

    combined_index = (
        annual_yield_percent.index
        .union(lookup_dates)
        .sort_values()
    )

    filled_yields = (
        annual_yield_percent
        .reindex(combined_index)
        .ffill()
    )

    prior_available_yield = (
        filled_yields
        .reindex(lookup_dates)
    )

    prior_available_yield.index = target_dates

    if prior_available_yield.isna().any():
        first_missing_date = (
            prior_available_yield[
                prior_available_yield.isna()
            ].index[0]
        )

        raise RiskFreeDataError(
            "No earlier risk-free observation was "
            f"available for {first_missing_date.date()}."
        )

    date_series = pd.Series(
        target_dates,
        index=target_dates,
    )

    calendar_days = (
        date_series
        .diff()
        .dt.days
        .fillna(0.0)
        .astype(float)
    )

    annual_decimal_yield = (
        prior_available_yield
        / 100.0
    )

    period_return = (
        annual_decimal_yield
        * calendar_days
        / day_count_basis
    )

    period_return.iloc[0] = 0.0

    result = pd.DataFrame(
        {
            "Annual yield percent used":
                prior_available_yield,
            "Calendar days":
                calendar_days,
            "Risk-free return":
                period_return,
        },
        index=target_dates,
    )

    result.index.name = "Date"

    return result


def prepare_risk_free_data(
    config: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    force_refresh: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """Download, align and save the risk-free return series."""

    symbols = list(
        config["data"]["tickers"]
    )

    target_dates = common_trading_dates(
        frames=frames,
        symbols=symbols,
    )

    annual_yield, source = (
        download_risk_free_yield(
            config=config,
            force_refresh=force_refresh,
        )
    )

    risk_free_config = config["risk_free"]

    daily_table = build_daily_risk_free_table(
        annual_yield_percent=annual_yield,
        target_dates=target_dates,
        day_count_basis=float(
            risk_free_config["day_count_basis"]
        ),
    )

    processed_path = Path(
        risk_free_config["processed_file"]
    )

    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_table.to_csv(
        processed_path
    )

    print(
        f"Prepared daily risk-free returns: {processed_path}"
    )

    summary = pd.DataFrame(
        [
            {
                "Series": risk_free_config[
                    "fred_series"
                ],
                "Source": source,
                "First trading date":
                    daily_table.index.min().date(),
                "Last trading date":
                    daily_table.index.max().date(),
                "Average annual yield":
                    daily_table[
                        "Annual yield percent used"
                    ].mean()
                    / 100.0,
            }
        ]
    )

    risk_free_returns = daily_table[
        "Risk-free return"
    ].copy()

    return risk_free_returns, summary