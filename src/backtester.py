"""Long-only portfolio backtester for the two ETF strategies."""

from pathlib import Path
import math

import pandas as pd


def commission(shares, config):
    """Commission for one order."""

    per_share = float(
        config["costs"]["commission_per_share"]
    )

    minimum = float(
        config["costs"]["minimum_commission"]
    )

    if shares <= 0:
        return 0.0

    return max(
        minimum,
        shares * per_share,
    )


def fill_price(
    price,
    side,
    config,
):
    """Apply adverse slippage to a theoretical price."""

    rate = (
        float(config["costs"]["slippage_bps"])
        / 10_000
    )

    if side == "buy":
        return price * (1 + rate)

    return price * (1 - rate)


def portfolio_value(
    cash,
    positions,
    frames,
    date,
    price_column,
):
    """Cash plus the value of all open positions."""

    value = cash

    for symbol, position in positions.items():
        price = float(
            frames[symbol].at[
                date,
                price_column,
            ]
        )

        value += (
            position["shares"]
            * price
        )

    return value


def common_dates(
    frames,
    symbols,
):
    """Return dates available for every ETF."""

    dates = pd.DatetimeIndex(
        frames[symbols[0]].index
    )

    for symbol in symbols[1:]:
        dates = dates.intersection(
            frames[symbol].index
        )

    dates = dates.sort_values()

    if dates.empty:
        raise ValueError(
            "No common ETF trading dates were found."
        )

    return dates


def signal_priority(
    strategy,
    row,
):
    """Rank simultaneous entry signals."""

    if strategy == "momentum":
        return (
            float(row["Momentum_Fast_MA"])
            / float(row["Momentum_Slow_MA"])
        ) - 1

    # A more negative z-score represents a stronger
    # mean-reversion opportunity.
    return -float(row["ZScore"])


def backtest_strategy(
    frames,
    config,
    strategy,
):
    """
    Run one strategy.

    Signals observed at today's close execute at the
    next trading day's open.
    """

    if strategy not in {
        "momentum",
        "mean_reversion",
    }:
        raise ValueError(
            "Unknown strategy."
        )

    symbols = list(
        config["data"]["tickers"]
    )

    dates = common_dates(
        frames,
        symbols,
    )

    initial_capital = float(
        config["portfolio"]["initial_capital"]
    )

    max_allocation = float(
        config["portfolio"][
            "maximum_position_allocation"
        ]
    )

    risk_per_trade = float(
        config["portfolio"]["risk_per_trade"]
    )

    atr_multiple = float(
        config["risk"]["atr_stop_multiple"]
    )

    trailing = config["risk"][
        "trailing_stop"
    ]

    trailing_enabled = bool(
        trailing["enabled"]
    )

    trailing_multiple = float(
        trailing["atr_multiple"]
    )

    circuit = config["risk"][
        "circuit_breaker"
    ]

    circuit_enabled = bool(
        circuit["enabled"]
    )

    drawdown_limit = float(
        circuit["maximum_drawdown"]
    )

    cooldown_days = int(
        circuit["cooldown_days"]
    )

    max_holding_days = int(
        config["mean_reversion"][
            "maximum_holding_days"
        ]
    )

    if strategy == "momentum":
        buy_column = (
            "Momentum_Buy_Signal"
        )

        exit_column = (
            "Momentum_Exit_Signal"
        )

    else:
        buy_column = (
            "MeanReversion_Buy_Signal"
        )

        exit_column = (
            "MeanReversion_Exit_Signal"
        )

    cash = initial_capital

    positions = {}

    pending_entries = []

    pending_exits = {}

    peak_value = initial_capital

    cooldown_until = -1

    circuit_breakers = 0

    history_rows = []

    trade_rows = []

    def sell(
        symbol,
        date,
        market_price,
        reason,
    ):
        """
        Close one position and update cash and
        trade history.
        """

        nonlocal cash

        position = positions.pop(symbol)

        exit_price = fill_price(
            market_price,
            "sell",
            config,
        )

        exit_commission = commission(
            position["shares"],
            config,
        )

        cash += (
            position["shares"]
            * exit_price
            - exit_commission
        )

        net_pnl = (
            position["shares"]
            * (
                exit_price
                - position["entry_price"]
            )
            - position["entry_commission"]
            - exit_commission
        )

        trade_rows.append(
            {
                "Strategy": strategy,
                "Symbol": symbol,
                "Entry date": (
                    position["entry_date"]
                ),
                "Exit date": date,
                "Shares": (
                    position["shares"]
                ),
                "Entry price": (
                    position["entry_price"]
                ),
                "Exit price": exit_price,
                "Net PnL": net_pnl,
                "Trade return": (
                    exit_price
                    / position["entry_price"]
                ) - 1,
                "Holding days": (
                    position["holding_days"]
                ),
                "Exit reason": reason,
            }
        )

    for day_number, date in enumerate(
        dates
    ):
        # ----------------------------------------
        # 1. Yesterday's exit signals execute at
        # today's open.
        # ----------------------------------------

        for symbol, reason in list(
            pending_exits.items()
        ):
            if symbol in positions:
                sell(
                    symbol=symbol,
                    date=date,
                    market_price=float(
                        frames[symbol].at[
                            date,
                            "Open",
                        ]
                    ),
                    reason=reason,
                )

        pending_exits.clear()

        # ----------------------------------------
        # 2. Yesterday's buy signals execute at
        # today's open.
        # ----------------------------------------

        if (
            day_number > cooldown_until
            and pending_entries
        ):
            signal_date = dates[
                day_number - 1
            ]

            equity_at_open = portfolio_value(
                cash=cash,
                positions=positions,
                frames=frames,
                date=date,
                price_column="Open",
            )

            ranked_symbols = sorted(
                pending_entries,
                key=lambda symbol: (
                    signal_priority(
                        strategy,
                        frames[symbol].loc[
                            signal_date
                        ],
                    )
                ),
                reverse=True,
            )

            for symbol in ranked_symbols:
                if symbol in positions:
                    continue

                atr = float(
                    frames[symbol].at[
                        signal_date,
                        "ATR",
                    ]
                )

                if (
                    not math.isfinite(atr)
                    or atr <= 0
                ):
                    continue

                entry_price = fill_price(
                    float(
                        frames[symbol].at[
                            date,
                            "Open",
                        ]
                    ),
                    "buy",
                    config,
                )

                stop_distance = (
                    atr_multiple
                    * atr
                )

                risk_budget = (
                    equity_at_open
                    * risk_per_trade
                )

                allocation_budget = (
                    equity_at_open
                    * max_allocation
                )

                shares_by_risk = math.floor(
                    risk_budget
                    / stop_distance
                )

                shares_by_allocation = (
                    math.floor(
                        allocation_budget
                        / entry_price
                    )
                )

                shares = min(
                    shares_by_risk,
                    shares_by_allocation,
                )

                # Reduce the order until there is
                # enough cash to afford it.
                while shares > 0:
                    entry_commission = (
                        commission(
                            shares,
                            config,
                        )
                    )

                    total_cost = (
                        shares
                        * entry_price
                        + entry_commission
                    )

                    if total_cost <= cash:
                        break

                    shares -= 1

                if shares <= 0:
                    continue

                cash -= total_cost

                positions[symbol] = {
                    "shares": shares,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "entry_commission": (
                        entry_commission
                    ),
                    "stop_price": (
                        entry_price
                        - stop_distance
                    ),
                    "highest_close": float(
                        frames[symbol].at[
                            date,
                            "Close",
                        ]
                    ),
                    "holding_days": 0,
                }

        pending_entries.clear()

        # ----------------------------------------
        # 3. Check today's low against every
        # position's ATR stop.
        # ----------------------------------------

        for symbol in list(positions):
            position = positions[symbol]

            open_price = float(
                frames[symbol].at[
                    date,
                    "Open",
                ]
            )

            low_price = float(
                frames[symbol].at[
                    date,
                    "Low",
                ]
            )

            if (
                low_price
                <= position["stop_price"]
            ):
                # When price gaps below the stop,
                # use the worse opening price.
                if (
                    open_price
                    <= position["stop_price"]
                ):
                    stop_market_price = (
                        open_price
                    )

                else:
                    stop_market_price = (
                        position["stop_price"]
                    )

                sell(
                    symbol=symbol,
                    date=date,
                    market_price=(
                        stop_market_price
                    ),
                    reason="atr_stop",
                )

        # ----------------------------------------
        # 4. Update open positions and schedule
        # tomorrow's exits.
        # ----------------------------------------

        for symbol, position in (
            positions.items()
        ):
            row = frames[symbol].loc[
                date
            ]

            close_price = float(
                row["Close"]
            )

            position["holding_days"] += 1

            position["highest_close"] = max(
                position["highest_close"],
                close_price,
            )

            if trailing_enabled:
                atr = float(row["ATR"])

                if (
                    math.isfinite(atr)
                    and atr > 0
                ):
                    new_stop = (
                        position[
                            "highest_close"
                        ]
                        - trailing_multiple
                        * atr
                    )

                    position["stop_price"] = (
                        max(
                            position[
                                "stop_price"
                            ],
                            new_stop,
                        )
                    )

            if bool(row[exit_column]):
                pending_exits[symbol] = (
                    "strategy_signal"
                )

            elif (
                strategy
                == "mean_reversion"
                and position[
                    "holding_days"
                ]
                >= max_holding_days
            ):
                pending_exits[symbol] = (
                    "maximum_holding_days"
                )

        # ----------------------------------------
        # 5. Calculate the closing portfolio value
        # and check the circuit breaker.
        # ----------------------------------------

        closing_value = portfolio_value(
            cash=cash,
            positions=positions,
            frames=frames,
            date=date,
            price_column="Close",
        )

        peak_value = max(
            peak_value,
            closing_value,
        )

        drawdown = (
            closing_value
            / peak_value
        ) - 1

        circuit_active = (
            day_number
            <= cooldown_until
        )

        if (
            circuit_enabled
            and not circuit_active
            and drawdown
            <= -drawdown_limit
        ):
            circuit_breakers += 1

            for symbol in positions:
                pending_exits[symbol] = (
                    "circuit_breaker"
                )

            pending_entries.clear()

            cooldown_until = (
                day_number
                + cooldown_days
            )

            # Future drawdown measurement begins
            # from the new capital level.
            peak_value = closing_value

            circuit_active = True

        # ----------------------------------------
        # 6. Today's buy signals become tomorrow's
        # pending entries.
        # ----------------------------------------

        if day_number < len(dates) - 1:
            entries_allowed_tomorrow = (
                day_number + 1
                > cooldown_until
            )

            if entries_allowed_tomorrow:
                pending_entries = [
                    symbol
                    for symbol in symbols
                    if (
                        symbol
                        not in positions
                    )
                    and (
                        symbol
                        not in pending_exits
                    )
                    and bool(
                        frames[symbol].at[
                            date,
                            buy_column,
                        ]
                    )
                ]

            else:
                pending_entries = []

        history_rows.append(
            {
                "Date": date,
                "Strategy": strategy,
                "Cash": cash,
                "Positions value": (
                    closing_value
                    - cash
                ),
                "Portfolio value": (
                    closing_value
                ),
                "Drawdown": drawdown,
                "Open positions": len(
                    positions
                ),
                "Circuit breaker active": (
                    circuit_active
                ),
            }
        )

    # Close any positions still open at the
    # final available closing price.

    final_date = dates[-1]

    for symbol in list(positions):
        sell(
            symbol=symbol,
            date=final_date,
            market_price=float(
                frames[symbol].at[
                    final_date,
                    "Close",
                ]
            ),
            reason="end_of_backtest",
        )

    history = pd.DataFrame(
        history_rows
    ).set_index("Date")

    trades = pd.DataFrame(
        trade_rows
    )

    # Update the last portfolio row after all final
    # positions have been sold.

    history.at[
        final_date,
        "Cash",
    ] = cash

    history.at[
        final_date,
        "Positions value",
    ] = 0.0

    history.at[
        final_date,
        "Portfolio value",
    ] = cash

    history.at[
        final_date,
        "Open positions",
    ] = 0

    trade_count = len(trades)

    if trade_count > 0:
        winning_trades = int(
            (
                trades["Net PnL"]
                > 0
            ).sum()
        )

        win_rate = (
            winning_trades
            / trade_count
        )

    else:
        win_rate = 0.0

    summary = {
        "Strategy": strategy,
        "Initial capital": (
            initial_capital
        ),
        "Final value": cash,
        "Total return": (
            cash
            / initial_capital
        ) - 1,
        "Trades": trade_count,
        "Win rate": win_rate,
        "Circuit breakers": (
            circuit_breakers
        ),
    }

    return (
        history,
        trades,
        summary,
    )


def run_all_backtests(
    frames,
    config,
):
    """Run both strategies and save their result files."""

    output_directory = Path(
        "outputs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for strategy in [
        "momentum",
        "mean_reversion",
    ]:
        (
            history,
            trades,
            summary,
        ) = backtest_strategy(
            frames=frames,
            config=config,
            strategy=strategy,
        )

        history.to_csv(
            output_directory
            / f"{strategy}_portfolio.csv"
        )

        trades.to_csv(
            output_directory
            / f"{strategy}_trades.csv",
            index=False,
        )

        summaries.append(
            summary
        )

        print(
            f"Completed {strategy} backtest."
        )

        print(
            f"  outputs/"
            f"{strategy}_portfolio.csv"
        )

        print(
            f"  outputs/"
            f"{strategy}_trades.csv"
        )

    summary_table = pd.DataFrame(
        summaries
    )

    summary_table.to_csv(
        output_directory
        / "backtest_summary.csv",
        index=False,
    )

    return summary_table