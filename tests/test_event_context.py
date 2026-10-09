"""Regressions for ETF-specific event dates, macro endpoints and rate units."""
from pathlib import Path
import io
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import langgraph
    import faiss
    import matplotlib
except ModuleNotFoundError:
    raise unittest.SkipTest("Use the existing torch312 environment for these integration checks")

import pandas as pd
from langchain_core.documents import Document
from offline_runtime import load_notebook, block_network, invoke_graph

REQUEST = {"question": "SPY QQQ policy", "as_of": "2025-01-08T22:00:00Z", "horizon_sessions": 1}


def multi_etf_namespace(path=None):
    docs = [Document(page_content="SPY CURRENT_EVENT policy", metadata={
                "title": "SPY synthetic event", "date": "2025-01-06T22:00:00Z",
                "url": "https://example.invalid/current"}),
            Document(page_content="QQQ CURRENT_EVENT policy", metadata={
                "title": "QQQ synthetic event", "date": "2025-01-07T22:00:00Z",
                "url": "https://example.invalid/qqq-current"})]
    docs += [Document(page_content=f"weather unrelated{i}", metadata={
                "title": f"weather{i}", "date": "2025-01-02T22:00:00Z",
                "url": f"https://example.invalid/noise-{i}"}) for i in range(12)]
    ns = load_notebook(path, documents=docs)
    ns["KNOWN_TICKERS"] = {"SPY", "QQQ"}
    ns["price_df"] = pd.DataFrame({"ticker": ["SPY"] * 4 + ["QQQ"] * 4,
        "date_kst": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"] * 2),
        "adj_close": [100., 90., 99., 49.5, 200., 200., 220., 242.]})
    ns["etf_profile_table"] = {"SPY": {"ticker": "SPY", "name": "Synthetic SPY profile"},
                               "QQQ": {"ticker": "QQQ", "name": "Synthetic QQQ profile"}}
    ns["macro_df"] = pd.DataFrame({"date": pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08"]),
                                    "SPX_close": [100., 110., 99.], "UST10Y_close": [4.25, 4.50, 4.40]})
    ns["macro_df"].attrs["units"] = {"UST10Y_close": "percent"}
    return ns


def macro_for(state, ticker):
    context = state["macro_context"]
    return context.get("by_ticker", {}).get(ticker, context)


class EventContextRegression(unittest.TestCase):
    def setUp(self):
        self.guard = block_network()
        self.attempts = self.guard.__enter__()
        self.ns = multi_etf_namespace()

    def tearDown(self):
        self.guard.__exit__(None, None, None)
        self.assertEqual(self.attempts, [])

    def test_actual_graph_keeps_each_etf_event_return_and_source_together(self):
        state, trace = invoke_graph(self.ns, REQUEST)
        self.assertAlmostEqual(state["etf_stats"]["SPY"]["current_change"], 10.)
        self.assertAlmostEqual(state["etf_stats"]["QQQ"]["current_change"], 10.)
        # MemorySaver serializes the tuple-valued window as a JSON-style list.
        self.assertEqual(list(state["etf_stats"]["SPY"]["current_window"]), ["2025-01-06", "2025-01-07"])
        self.assertEqual(list(state["etf_stats"]["QQQ"]["current_window"]), ["2025-01-07", "2025-01-08"])
        self.assertEqual(state["price_window_status"]["SPY"]["event_sources"][0]["url"], "https://example.invalid/current")
        for prompt in self.ns["llm"].prompts[1:]:
            self.assertIn("Synthetic SPY profile", prompt)
            self.assertIn("Synthetic QQQ profile", prompt)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            report = Path(self.ns["save_report_md"](state, out_dir=tmp)).read_text(encoding="utf-8")
            self.assertIn("SPY 기준 기사", report)
            self.assertIn("QQQ 기준 기사", report)

    def test_unmapped_later_article_cannot_make_a_complete_etf_window_wait(self):
        hits = [{"title": "SPY", "text": "SPY CURRENT_EVENT", "date": "2025-01-06T22:00:00Z"},
                {"title": "weather", "text": "unrelated weather", "date": "2025-01-08T22:00:00Z"}]
        state = self.ns["attach_price_context"]({**REQUEST, "reranked_hits": hits})
        self.assertEqual(state["price_window_status"]["SPY"]["status"], "complete")
        self.assertAlmostEqual(state["etf_stats"]["SPY"]["current_change"], 10.)

    def test_macro_windows_follow_each_etf_instead_of_the_last_reaction(self):
        state = {"as_of": REQUEST["as_of"], "price_reactions": [
            {"ticker": "SPY", "start_date": "2025-01-06", "end_date": "2025-01-07"},
            {"ticker": "QQQ", "start_date": "2025-01-07", "end_date": "2025-01-08"}]}
        state.update(self.ns["attach_macro_context"](state))
        self.assertAlmostEqual(macro_for(state, "SPY")["SPX"]["chg_pct"], 10.)
        self.assertAlmostEqual(macro_for(state, "QQQ")["SPX"]["chg_pct"], -10.)

    def test_missing_macro_endpoint_is_not_presented_as_zero_change(self):
        self.ns["macro_df"] = self.ns["macro_df"].iloc[[0]].copy()
        state = {"as_of": REQUEST["as_of"], "price_reactions": [
            {"ticker": "SPY", "start_date": "2025-01-06", "end_date": "2025-01-07"}]}
        state.update(self.ns["attach_macro_context"](state))
        result = macro_for(state, "SPY")["SPX"]
        self.assertIsNone(result["chg_pct"])
        self.assertEqual(result["status"], "missing_endpoint")

    def test_basis_points_require_an_explicit_yield_unit(self):
        state = {"as_of": REQUEST["as_of"], "price_reactions": [
            {"ticker": "SPY", "start_date": "2025-01-06", "end_date": "2025-01-07"}]}
        state.update(self.ns["attach_macro_context"](state))
        result = macro_for(state, "SPY")["UST10Y"]
        self.assertAlmostEqual(result["chg_bp"], 25.)
        self.assertAlmostEqual(result["chg_percentage_points"], .25)
        self.ns["macro_df"].attrs.clear()
        state.update(self.ns["attach_macro_context"](state))
        result = macro_for(state, "SPY")["UST10Y"]
        self.assertIsNone(result["chg_bp"])
        self.assertEqual(result["unit_status"], "unverified")


if __name__ == "__main__":
    unittest.main(verbosity=2)
