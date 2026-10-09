"""A reported comparison must preserve its own article/price evidence."""
from pathlib import Path
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
    raise unittest.SkipTest("Use the existing torch312 environment")

from langchain_core.documents import Document
from offline_runtime import block_network, fixtures, load_notebook


class HistoricalReportSources(unittest.TestCase):
    def setUp(self):
        self.guard = block_network()
        self.attempts = self.guard.__enter__()

    def tearDown(self):
        self.guard.__exit__(None, None, None)
        self.assertEqual(self.attempts, [])

    def context(self, ns, *, url="https://example.invalid/current"):
        state = ns["init_today"]({"question": "SPY", "as_of": "2025-01-07T22:00:00Z", "horizon_sessions": 1})
        state["reranked_hits"] = [{"title": "SPY current only", "text": "SPY CURRENT_EVENT",
                                   "date": "2025-01-06T22:00:00Z", "url": url}]
        state.update(ns["attach_price_context"](state))
        state["final_answer"] = "Source/price check only. No generation model was called."
        return state

    def report(self, ns, state):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = ns["save_report_md"](state, out_dir=tmp)
            return Path(path).read_text(encoding="utf-8")

    def test_article_found_only_in_historical_search_reaches_the_report(self):
        ns = load_notebook()
        state = self.context(ns)
        self.assertAlmostEqual(state["etf_stats"]["SPY"]["hist_changes"][0], 2.)
        self.assertIn("https://example.invalid/past", self.report(ns, state))
        evidence = state["etf_stats"]["SPY"]["historical_evidence"][0]
        self.assertEqual(evidence["start_date"], "2024-12-23")
        self.assertEqual(evidence["end_date"], "2024-12-24")
        self.assertEqual((evidence["start_price"], evidence["end_price"]), (100., 102.))
        self.assertIn(evidence["sources"][0]["source_id"], self.report(ns, state))
        self.assertEqual(len(ns["llm"].prompts), 0)

    def test_shared_price_interval_keeps_both_article_sources_without_double_counting(self):
        docs, _, _ = fixtures()
        docs.append(Document(page_content="SPY PAST_EVENT second source", metadata={
            "date": "2024-12-23T23:00:00Z", "url": "https://example.invalid/past-second",
            "title": "Second historical article"}))
        ns = load_notebook(documents=docs)
        state = self.context(ns)
        stats = state["etf_stats"]["SPY"]
        self.assertEqual(len(stats["hist_changes"]), 1)
        evidence = stats.get("historical_evidence", [])
        self.assertEqual(len(evidence), 1)
        self.assertEqual({s["url"] for s in evidence[0]["sources"]},
                         {"https://example.invalid/past", "https://example.invalid/past-second"})
        report = self.report(ns, state)
        self.assertIn("https://example.invalid/past-second", report)

    def test_missing_url_is_labeled_unknown_instead_of_citing_nan(self):
        ns = load_notebook()
        state = self.context(ns, url=float("nan"))
        report = self.report(ns, state)
        self.assertNotIn("| nan", report)
        self.assertIn("URL 미확인", report)
        source = state["price_window_status"]["SPY"]["event_sources"][0]
        self.assertIsNone(source["url"])
        json.dumps(source, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
