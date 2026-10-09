"""Run only selected notebook functions with synthetic data, never imports/API cells."""
from pathlib import Path
import ast
from datetime import datetime, timezone, timedelta
import json
import re
from types import SimpleNamespace
import sys
from typing import Any, Dict, List, Optional, TypedDict
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from price_windows import event_price_window, market_timestamp
from retrieval_contracts import visible_news, retrieval_fetch_k
from macro_windows import macro_window_summary
from source_evidence import article_source, document_source
from test_price_windows import sample_prices


class ForbiddenLLM:
    def invoke(self, *args, **kwargs):
        raise AssertionError("Unobservable price windows must not request an explanation")


def namespace():
    wanted = {
        "GraphState", "get_price_window", "init_today", "find_similar_news",
        "summarize_historical_etf_reactions", "attach_price_context", "extract_etfs_from_text",
        "attach_historical_price_context", "_passthrough", "analyze_price_reaction",
        "attach_etf_profile", "attach_macro_context", "explain_news_impact", "final_answer",
        "format_etf_profiles",
    }
    nb = json.loads((ROOT / "LangGraph.ipynb").read_text(encoding="utf-8"))
    definitions = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
                definitions.append(node)
    ns = {"pd": pd, "re": re, "datetime": datetime, "timezone": timezone, "timedelta": timedelta,
          "Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "TypedDict": TypedDict,
          "event_price_window": event_price_window, "market_timestamp": market_timestamp,
          "visible_news": visible_news, "retrieval_fetch_k": retrieval_fetch_k,
          "macro_window_summary": macro_window_summary,
          "article_source": article_source, "document_source": document_source,
          "price_df": sample_prices(), "KNOWN_TICKERS": {"SPY"}, "KEYWORD_TO_TICKERS": {},
          "HIST_TOPK": 3, "HIST_MIN_DAYS_GAP": 7, "llm": ForbiddenLLM(),
          "etf_profile_table": {"SPY": {"ticker": "SPY", "name": "Synthetic ETF profile"}},
          "vectorstore": SimpleNamespace(index=SimpleNamespace(ntotal=0), similarity_search=lambda *args, **kwargs: []),
          "macro_df": pd.DataFrame({"date": pd.to_datetime(["2025-01-06", "2025-01-07"]),
                                     "SPX_close": [100.0, 110.0]})}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "selected_notebook_functions", "exec"), ns)
    return ns


class NotebookIntegration(unittest.TestCase):
    def test_complete_window_is_the_same_in_context_stats_and_reactions(self):
        ns = namespace()
        state = {"question": "SPY", "as_of": "2025-01-08T22:00:00Z", "horizon_sessions": 1,
                 "reranked_hits": [{"title": "SPY synthetic event", "text": "SPY", "date": "2025-01-06T22:00:00Z"}]}
        # Seed retrieval output after initialization; initialization clears checkpoint evidence.
        hits = state["reranked_hits"]
        state.update(ns["init_today"](state))
        state["reranked_hits"] = hits
        for name in ("attach_price_context", "attach_etf_profile", "attach_historical_price_context",
                     "analyze_price_reaction", "attach_macro_context"):
            writes = ns[name](state)
            self.assertTrue(set(writes).issubset(ns["GraphState"].__annotations__), name)
            state.update(writes)
        self.assertEqual(state["matched_etfs"], ["SPY"])
        self.assertAlmostEqual(state["etf_stats"]["SPY"]["current_change"], 10.0)
        self.assertAlmostEqual(state["price_reactions"][0]["return_pct"], 10.0)
        self.assertIn("2025-01-06~2025-01-07 +10.00%", state["price_context"])
        self.assertEqual(state["etf_profile"]["ticker"], "SPY")
        self.assertEqual(state["macro_context"]["by_ticker"]["SPY"]["window_start"], "2025-01-06")
        self.assertEqual(state["macro_context"]["by_ticker"]["SPY"]["window_end"], "2025-01-07")

    def test_unfinished_window_does_not_request_llm_analysis_or_reuse_old_stats(self):
        ns = namespace()
        state = {"as_of": "2025-01-07T20:00:00Z", "horizon_sessions": 1,
                 "etf_stats": {"SPY": {"current_change": 999, "current_window": ("1999-01-01", "1999-01-02")}},
                 "reranked_hits": [{"title": "SPY", "text": "SPY", "date": "2025-01-06T22:00:00Z"}]}
        hits = state["reranked_hits"]
        state.update(ns["init_today"](state))
        state["reranked_hits"] = hits
        for name in ("attach_price_context", "analyze_price_reaction", "attach_macro_context", "explain_news_impact", "final_answer"):
            state.update(ns[name](state))
        self.assertEqual(state["etf_stats"], {})
        self.assertEqual(state["price_reactions"], [])
        self.assertEqual(state["macro_context"], {})
        self.assertEqual(state["price_window_status"]["SPY"]["status"], "awaiting_observations")
        self.assertIn("관측 대기", state["final_answer"])
        self.assertNotIn("999", state["final_answer"])

    def test_article_published_after_as_of_is_removed_from_explanation_input(self):
        ns = namespace()
        state = {"as_of": "2025-01-07T22:00:00Z", "horizon_sessions": 1,
                 "reranked_hits": [
                     {"title": "known", "text": "SPY", "date": "2025-01-06T22:00:00Z"},
                     {"title": "future", "text": "SPY", "date": "2025-01-08T22:00:00Z"}]}
        result = ns["attach_price_context"](state)
        self.assertEqual([h["title"] for h in result["reranked_hits"]], ["known"])
        self.assertAlmostEqual(result["etf_stats"]["SPY"]["current_change"], 10.0)

    def test_historical_windows_do_not_use_missing_early_prices_or_future_articles(self):
        ns = namespace()
        docs = [SimpleNamespace(metadata={"date": "2024-12-01T22:00:00Z", "url": "synthetic-old"}, page_content="SPY"),
                SimpleNamespace(metadata={"date": "2025-01-09T22:00:00Z", "url": "synthetic-future"}, page_content="SPY")]
        ns["vectorstore"] = SimpleNamespace(index=SimpleNamespace(ntotal=len(docs)), similarity_search=lambda *args, **kwargs: docs)
        formatted, changes = ns["summarize_historical_etf_reactions"](
            "SPY", "SPY", "2025-01-06T22:00:00Z", as_of="2025-01-08T22:00:00Z", horizon_sessions=1)
        self.assertEqual((formatted, changes), ([], []))

    def test_one_price_window_definition_remains(self):
        nb = json.loads((ROOT / "LangGraph.ipynb").read_text(encoding="utf-8"))
        count = sum(isinstance(n, ast.FunctionDef) and n.name == "get_price_window"
                    for c in nb["cells"] if c["cell_type"] == "code"
                    for n in ast.parse("".join(c["source"])).body)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
