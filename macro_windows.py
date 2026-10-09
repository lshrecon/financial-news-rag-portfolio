"""Daily macro comparisons with exact endpoints and explicit rate units.

Daily labels alone do not establish each provider's publication time. Callers
must verify that separately before treating these observations as point-in-time.
"""
import math
import pandas as pd


def macro_window_summary(frame, start_date, end_date):
    """Never substitute an interior row for a missing requested endpoint."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start.tzinfo is not None or end.tzinfo is not None or start >= end:
        raise ValueError("Macro window requires ordered, timezone-free daily labels")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.dt.tz is not None or dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Macro dates must be timezone-free daily labels")
    if dates.duplicated().any():
        raise ValueError("Duplicate macro daily labels must be resolved upstream")
    result = {"window_start": start.date().isoformat(), "window_end": end.date().isoformat(),
              "observation_timing": "daily labels only; provider publication times remain unverified"}
    units = frame.attrs.get("units", {})
    for name in ("SPX", "VIX", "DXY", "UST10Y", "GOLD", "BTC"):
        column = f"{name}_close"
        value = {"status": "missing_column", "start": None, "end": None,
                 "chg_abs": None, "chg_pct": None}
        if name == "UST10Y":
            value.update({"input_unit": units.get(column, "unverified"),
                          "unit_status": "verified_input_contract" if units.get(column) == "percent" else "unverified",
                          "chg_percentage_points": None, "chg_bp": None})
        if column not in frame.columns:
            result[name] = value
            continue
        series = frame[column]
        if isinstance(series, pd.DataFrame):
            raise ValueError("Macro columns must be unique, flat scalar series")
        first, last = series.loc[dates.eq(start)], series.loc[dates.eq(end)]
        if first.empty or last.empty or pd.isna(first.iloc[0]) or pd.isna(last.iloc[0]):
            value["status"] = "missing_endpoint"
            result[name] = value
            continue
        try:
            a, b = float(first.iloc[0]), float(last.iloc[0])
        except (ValueError, TypeError):
            value["status"] = "invalid_endpoint"
            result[name] = value
            continue
        if not math.isfinite(a) or not math.isfinite(b) or (name != "UST10Y" and (a <= 0 or b <= 0)):
            value["status"] = "invalid_endpoint"
            result[name] = value
            continue
        difference = b - a
        value.update({"status": "complete", "start": a, "end": b, "chg_abs": difference})
        if name == "UST10Y":
            if units.get(column) == "percent":
                value.update({"chg_percentage_points": difference, "chg_bp": difference * 100.0})
            # Unknown source units stay raw; never label an unverified difference bp.
        else:
            value["chg_pct"] = (b / a - 1.0) * 100.0
        result[name] = value
    return result
