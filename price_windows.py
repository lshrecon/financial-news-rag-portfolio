"""Offline event windows for daily US ETF closing prices.

Input dates are exchange-session labels, even when the legacy column is named
``date_kst``. News/as-of timestamps must have a timezone. A date-only event is
conservatively treated as the end of that exchange day. Regular-session closes
are assumed to occur at 16:00 America/New_York; early closes are not modeled.
"""
from __future__ import annotations

from datetime import date, datetime
import math
import re

import pandas as pd


def market_timestamp(value, *, market_tz="America/New_York"):
    """Keep real timestamps; explicitly interpret date-only values as day-end."""
    date_only = (
        isinstance(value, date) and not isinstance(value, datetime)
    ) or (isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None)
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("A valid event/as-of timestamp is required")
    if date_only:
        local_day_end = stamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return local_day_end.tz_localize(market_tz)
    if stamp.tzinfo is None:
        raise ValueError("Timestamp must include a timezone, or use an explicit YYYY-MM-DD date")
    return stamp.tz_convert(market_tz)


def observable_price_rows(prices, as_of, *, date_column="date_kst", market_tz="America/New_York"):
    """Use the same regular-session visibility rule for every chart as for returns."""
    cutoff = market_timestamp(as_of, market_tz=market_tz)
    sessions = pd.to_datetime(prices[date_column], errors="raise")
    if sessions.dt.tz is not None or sessions.isna().any() or not sessions.eq(sessions.dt.normalize()).all():
        raise ValueError("Chart prices require timezone-free daily exchange-session labels")
    close_at = (sessions + pd.Timedelta(hours=16)).dt.tz_localize(market_tz)
    result = prices.loc[close_at <= cutoff].copy()
    result[date_column] = sessions.loc[result.index]
    return result


def event_price_window(
    prices: pd.DataFrame,
    ticker: str,
    event_at,
    *,
    as_of,
    horizon_sessions: int = 3,
    date_column: str = "date_kst",
    price_column: str = "adj_close",
    market_tz: str = "America/New_York",
) -> dict:
    """Return the change from the last pre-event close to N post-event closes.

    No price later than ``as_of`` is observable. No out-of-range dates, incomplete
    windows or missing baseline prices are replaced with other dates. An
    unavailable window has ``status != 'complete'`` and no ``pct_change``.
    The post-event close count uses rows supplied by the caller, not a trading
    calendar; missing daily input rows must be checked upstream.
    """
    if isinstance(horizon_sessions, bool) or not isinstance(horizon_sessions, int) or horizon_sessions < 1:
        raise ValueError("horizon_sessions must be a positive integer")
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("A ticker is required")
    required = {"ticker", date_column, price_column}
    if not required.issubset(prices.columns):
        raise ValueError("Missing required price columns: " + ", ".join(sorted(required - set(prices.columns))))

    event = market_timestamp(event_at, market_tz=market_tz)
    cutoff = market_timestamp(as_of, market_tz=market_tz)
    ticker = ticker.strip().upper()
    result = {
        "ticker": ticker,
        "event_at": event.isoformat(),
        "as_of": cutoff.isoformat(),
        "horizon_sessions": horizon_sessions,
        "market_timezone": market_tz,
        "close_time_assumption": "16:00 regular session; early closes not modeled",
    }
    if event > cutoff:
        return {**result, "status": "event_after_as_of"}

    rows = prices.loc[prices["ticker"].astype(str).str.upper() == ticker,
                      [date_column, price_column]].copy()
    if rows.empty:
        return {**result, "status": "no_prices"}
    sessions = pd.to_datetime(rows[date_column], errors="raise")
    if sessions.dt.tz is not None:
        raise ValueError("Price dates must be timezone-free exchange-session labels")
    if sessions.isna().any() or not sessions.eq(sessions.dt.normalize()).all():
        raise ValueError("Price dates must be valid daily session labels at midnight")
    if sessions.duplicated().any():
        raise ValueError("Duplicate session dates for the selected ticker")
    rows["_session"] = sessions
    rows["_close_at"] = (sessions + pd.Timedelta(hours=16)).dt.tz_localize(market_tz)
    rows = rows.loc[rows["_close_at"] <= cutoff].sort_values("_close_at")
    values = pd.to_numeric(rows[price_column], errors="raise")
    if not values.map(lambda v: math.isfinite(float(v)) and float(v) > 0).all():
        raise ValueError("Observed prices must be finite and strictly positive")
    rows["_value"] = values.astype(float)

    before = rows.loc[rows["_close_at"] <= event]
    if before.empty:
        return {**result, "status": "no_prior_close"}
    after = rows.loc[rows["_close_at"] > event]
    if len(after) < horizon_sessions:
        return {**result, "status": "awaiting_observations",
                "available_post_event_closes": int(len(after))}

    start = before.iloc[-1]
    end = after.iloc[horizon_sessions - 1]
    start_price = float(start["_value"])
    end_price = float(end["_value"])
    return {
        **result,
        "status": "complete",
        "start_date": start["_session"].date().isoformat(),
        "end_date": end["_session"].date().isoformat(),
        "start_close_at": start["_close_at"].isoformat(),
        "end_close_at": end["_close_at"].isoformat(),
        "start_price": start_price,
        "end_price": end_price,
        "pct_change": (end_price / start_price - 1.0) * 100.0,
        "n_points": horizon_sessions + 1,
    }
