"""Synthetic regression fixtures. Baseline mode executes one legacy function only."""
from pathlib import Path
import ast
import json
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from price_windows import event_price_window

BASELINE = "--baseline" in sys.argv
if BASELINE:
    sys.argv.remove("--baseline")


def sample_prices():
    # Friday, Monday, Tuesday, Wednesday: the data includes future rows on purpose.
    return pd.DataFrame({
        "date_kst": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]),
        "ticker": ["SPY"] * 4,
        "adj_close": [100.0, 90.0, 99.0, 108.0],
    })


def subject(prices, ticker, event_at, *, as_of, horizon_sessions=1):
    if not BASELINE:
        return event_price_window(prices, ticker, event_at, as_of=as_of,
                                  horizon_sessions=horizon_sessions)
    # Execute the extracted original function, never the notebook or its imports.
    namespace = {"pd": pd, "price_df": prices}
    code = (ROOT / "tests" / "legacy_price_window.py").read_text(encoding="utf-8")
    exec(compile(code, "legacy_price_window.py", "exec"), namespace)
    old = namespace["get_price_window"](ticker, event_at, window=horizon_sessions)
    # Only normalize the return shape; the legacy logic receives no added filter.
    return {"status": "complete", **old} if old else {"status": "unavailable"}


class EventWindowRegression(unittest.TestCase):
    def test_article_after_close_measures_subsequent_move(self):
        result = subject(sample_prices(), "SPY", "2025-01-06T22:00:00Z",
                         as_of="2025-01-08T22:00:00Z")
        self.assertEqual(result["status"], "complete")
        # Jan 6 closed at 90 before the news; the next observed close is 99.
        self.assertAlmostEqual(result["pct_change"], 10.0)
        self.assertEqual((result["start_date"], result["end_date"]), ("2025-01-06", "2025-01-07"))

    def test_event_older_than_price_history_has_no_invented_return(self):
        result = subject(sample_prices(), "SPY", "2024-12-01T22:00:00Z",
                         as_of="2025-01-08T22:00:00Z")
        self.assertNotEqual(result["status"], "complete")
        self.assertNotIn("pct_change", result)

    def test_future_price_row_is_not_observed_before_market_close(self):
        result = subject(sample_prices(), "SPY", "2025-01-06T22:00:00Z",
                         as_of="2025-01-07T20:00:00Z")
        self.assertNotEqual(result["status"], "complete")
        self.assertNotIn("pct_change", result)

    def test_future_article_is_not_analyzed_at_earlier_as_of(self):
        result = subject(sample_prices(), "SPY", "2025-01-08T22:00:00Z",
                         as_of="2025-01-07T22:00:00Z")
        self.assertNotEqual(result["status"], "complete")
        self.assertNotIn("pct_change", result)

    def test_after_last_price_waits_instead_of_clamping_to_last_day(self):
        result = subject(sample_prices(), "SPY", "2025-02-01T22:00:00Z",
                         as_of="2025-02-03T22:00:00Z")
        self.assertNotEqual(result["status"], "complete")
        self.assertNotIn("pct_change", result)


@unittest.skipIf(BASELINE, "Additional contract checks apply to the repaired function")
class PriceWindowContract(unittest.TestCase):
    def test_intraday_article_uses_previous_close_and_today_close(self):
        result = event_price_window(sample_prices(), "SPY", "2025-01-06T18:00:00Z",
                                    as_of="2025-01-06T22:00:00Z", horizon_sessions=1)
        self.assertEqual(result["start_date"], "2025-01-03")
        self.assertEqual(result["end_date"], "2025-01-06")
        self.assertAlmostEqual(result["pct_change"], -10.0)

    def test_two_session_horizon_is_not_two_calendar_days(self):
        result = event_price_window(sample_prices(), "SPY", "2025-01-03T22:00:00Z",
                                    as_of="2025-01-08T22:00:00Z", horizon_sessions=2)
        self.assertEqual(result["end_date"], "2025-01-07")
        self.assertAlmostEqual(result["pct_change"], -1.0)

    def test_incomplete_horizon_is_not_published_as_partial_return(self):
        result = event_price_window(sample_prices(), "SPY", "2025-01-06T22:00:00Z",
                                    as_of="2025-01-07T22:00:00Z", horizon_sessions=2)
        self.assertEqual(result["status"], "awaiting_observations")
        self.assertEqual(result["available_post_event_closes"], 1)
        self.assertNotIn("pct_change", result)

    def test_dst_uses_new_york_close_instead_of_fixed_utc_offset(self):
        prices = pd.DataFrame({"date_kst": pd.to_datetime(["2025-03-07", "2025-03-10"]),
                               "ticker": ["SPY", "SPY"], "adj_close": [100.0, 110.0]})
        result = event_price_window(prices, "SPY", "2025-03-07T21:30:00Z",
                                    as_of="2025-03-10T20:30:00Z", horizon_sessions=1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(pd.Timestamp(result["end_close_at"]).tz_convert("UTC").hour, 20)
        self.assertAlmostEqual(result["pct_change"], 10.0)

    def test_date_only_news_uses_conservative_end_of_day(self):
        result = event_price_window(sample_prices(), "SPY", "2025-01-06",
                                    as_of="2025-01-08", horizon_sessions=1)
        self.assertEqual(result["end_date"], "2025-01-07")

    def test_timezone_naive_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            event_price_window(sample_prices(), "SPY", "2025-01-06 17:00:00",
                               as_of="2025-01-08", horizon_sessions=1)

    def test_duplicate_daily_rows_are_rejected(self):
        prices = pd.concat([sample_prices(), sample_prices().iloc[[1]]], ignore_index=True)
        with self.assertRaises(ValueError):
            event_price_window(prices, "SPY", "2025-01-06", as_of="2025-01-08", horizon_sessions=1)

    def test_zero_observed_price_is_rejected(self):
        prices = sample_prices()
        prices.loc[1, "adj_close"] = 0
        with self.assertRaises(ValueError):
            event_price_window(prices, "SPY", "2025-01-06", as_of="2025-01-08", horizon_sessions=1)

    def test_other_ticker_and_future_rows_do_not_change_observed_return(self):
        prices = pd.concat([sample_prices(), pd.DataFrame({
            "date_kst": pd.to_datetime(["2025-01-07", "2025-01-09"]),
            "ticker": ["QQQ", "SPY"], "adj_close": [99999.0, 88888.0],
        })], ignore_index=True)
        result = event_price_window(prices, "SPY", "2025-01-06T22:00:00Z",
                                    as_of="2025-01-07T22:00:00Z", horizon_sessions=1)
        self.assertAlmostEqual(result["pct_change"], 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
