"""Actual notebook graph/runtime integration; external model behavior is a double."""
from pathlib import Path
import ast
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import langgraph
    import faiss
    import matplotlib
except ModuleNotFoundError:
    raise unittest.SkipTest("Use the existing torch312 environment for graph/FAISS/chart integration")

import pandas as pd
from matplotlib.axes import Axes
from offline_runtime import load_notebook, block_network, invoke_graph, fixtures, FakeBGE
from retrieval_contracts import bge_score_list, faiss_l2_relevance

ASOF = "2025-01-07T22:00:00Z"
WAITING = "2025-01-07T20:00:00Z"
REQUEST = {"question": "SPY policy", "as_of": ASOF, "horizon_sessions": 1}
BASELINE = ROOT / "tests/fixtures/pre_graph_repair.ipynb"


class ActualGraphRuntime(unittest.TestCase):
    def setUp(self):
        self.guard = block_network()
        self.attempts = self.guard.__enter__()
        self.ns = load_notebook()

    def tearDown(self):
        self.guard.__exit__(None, None, None)
        self.assertEqual(self.attempts, [], "An offline fixture unexpectedly attempted network access")

    def test_complete_real_graph_report_and_source_provenance(self):
        state, trace = invoke_graph(self.ns, REQUEST)
        for name in ("retrieve_bm25", "retrieve_dense", "retrieve_web"):
            self.assertLess(trace.index(name), trace.index("join_retrieval"))
        for name in ("join_retrieval", "fuse_results", "rerank_results", "attach_price_context", "final_answer"):
            self.assertEqual(trace.count(name), 1)
        self.assertAlmostEqual(state["etf_stats"]["SPY"]["current_change"], 10.)
        self.assertAlmostEqual(state["price_reactions"][0]["return_pct"], 10.)
        self.assertEqual(state["price_window_status"]["SPY"]["end_date"], "2025-01-07")
        self.assertTrue(state["final_answer"].startswith("[OFFLINE_STUB]"))
        self.assertEqual(len(self.ns["llm"].prompts), 3)
        self.assertNotIn("FUTURE_DO_NOT_USE", "\n".join(self.ns["llm"].prompts))
        self.assertIsNone(state["consistency_score"])
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            report = Path(self.ns["save_report_md"](state, out_dir=tmp))
            text = report.read_text(encoding="utf-8")
            self.assertIn(state["as_of"], text)
            self.assertIn("https://example.invalid/current", text)
            self.assertIn("[OFFLINE_STUB]", text)
            logs = self.ns["save_run_logs"](state, out_dir=tmp)
            metadata = json.loads(Path(logs["meta_json"]).read_text(encoding="utf-8"))
            self.assertEqual(metadata["as_of"], state["as_of"])
            self.assertEqual(metadata["config"]["fusion_method"], "weighted_sum")

    def test_incomplete_window_real_graph_abstains_without_analysis_model_calls(self):
        state, trace = invoke_graph(self.ns, {**REQUEST, "as_of": WAITING})
        self.assertIn("final_answer", trace)
        self.assertEqual(state["etf_stats"], {})
        self.assertEqual(state["price_window_status"]["SPY"]["status"], "awaiting_observations")
        self.assertIn("관측 대기", state["final_answer"])
        self.assertEqual(len(self.ns["llm"].prompts), 1, "Only the query rewrite is allowed")

    def test_same_checkpoint_empty_query_cannot_reuse_previous_report_evidence(self):
        first, _ = invoke_graph(self.ns, REQUEST, thread_id="reused")
        self.assertTrue(first["etf_stats"])
        count = len(self.ns["llm"].prompts)
        state, trace = invoke_graph(self.ns, {**REQUEST, "question": ""}, thread_id="reused")
        self.assertIn("join_retrieval", trace)
        self.assertNotIn("fuse_results", trace)
        self.assertNotIn("rerank_results", trace)
        for key in ("etf_stats", "macro_context", "etf_profile", "price_window_status"):
            self.assertEqual(state[key], {})
        self.assertEqual(state["reranked_hits"], [])
        self.assertEqual(len(self.ns["llm"].prompts), count)
        self.assertNotIn("[OFFLINE_STUB]", state["final_answer"])

    def test_no_eligible_news_branches_to_abstention_after_all_retrievers(self):
        ns = load_notebook(documents=[])
        state, trace = invoke_graph(ns, REQUEST)
        self.assertEqual(trace[-2:], ["join_retrieval", "final_answer"])
        self.assertNotIn("fuse_results", trace)
        self.assertEqual(state["reranked_hits"], [])
        self.assertEqual(len(ns["llm"].prompts), 1)

    def test_real_faiss_future_top_k_starvation_is_reproduced_and_repaired(self):
        before = load_notebook(BASELINE)
        old = before["retrieve_dense"](REQUEST)["dense_hits"]
        new = self.ns["retrieve_dense"](REQUEST)["dense_hits"]
        self.assertEqual(len(old), 20)
        self.assertTrue(all("FUTURE_DO_NOT_USE" in hit["text"] for hit in old))
        self.assertFalse(any("FUTURE_DO_NOT_USE" in hit["text"] for hit in new))
        self.assertTrue(any("CURRENT_EVENT" in hit["text"] for hit in new))
        docs, _, _ = fixtures()
        only_past = load_notebook(documents=[d for d in docs if "FUTURE_DO_NOT_USE" not in d.page_content])
        past_hits = only_past["retrieve_dense"](REQUEST)["dense_hits"]
        # IndexFlatL2 uses independent point distances, not a trained corpus statistic.
        self.assertEqual([(h["url"], h["score"]) for h in new],
                         [(h["url"], h["score"]) for h in past_hits])
        old_hist = before["find_similar_news"]("SPY", "2025-01-06T22:00:00Z", as_of=ASOF)
        new_hist = self.ns["find_similar_news"]("SPY", "2025-01-06T22:00:00Z", as_of=ASOF)
        self.assertEqual(old_hist, [])
        self.assertTrue(any("PAST_EVENT" in doc.page_content for doc in new_hist))

    def test_bm25_idf_does_not_change_when_future_documents_are_added(self):
        docs, _, _ = fixtures()
        past_only = [doc for doc in docs if "FUTURE_DO_NOT_USE" not in doc.page_content]
        past_ns = load_notebook(documents=past_only)
        with_future = self.ns["retrieve_bm25"](REQUEST)["bm25_hits"]
        without_future = past_ns["retrieve_bm25"](REQUEST)["bm25_hits"]
        self.assertTrue(with_future)
        self.assertEqual([(h["url"], h["score"]) for h in with_future],
                         [(h["url"], h["score"]) for h in without_future])

    def test_web_uses_request_time_and_preserves_deduplication(self):
        before = load_notebook(BASELINE)
        self.assertEqual(before["retrieve_web"](REQUEST)["web_hits"], [])
        hits = self.ns["retrieve_web"](REQUEST)["web_hits"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["url"], "https://example.invalid/current")

    def test_reranker_real_return_shape_preserves_count_and_numeric_order(self):
        hits = [{"title": str(i), "text": f"SPY item-{i}", "url": str(i), "score": 1., "date": ASOF} for i in range(12)]
        hits[10]["text"] = "SPY CURRENT_EVENT"
        state = {**REQUEST, "fused_hits": hits}
        before = load_notebook(BASELINE)
        old = before["rerank_results"](copy.deepcopy(state))["reranked_hits"]
        new = self.ns["rerank_results"](copy.deepcopy(state))["reranked_hits"]
        self.assertEqual(len(old), 5)
        self.assertEqual(len(new), 12)
        self.assertEqual(new[0]["url"], "10")
        self.assertEqual(new[0]["rerank_score"], .99)
        self.assertEqual(hits[10]["score"], 1., "Reranking must not mutate upstream evidence")

    def test_distance_ranking_and_bad_reranker_results_fail_visibly(self):
        before = load_notebook(BASELINE)
        self.assertLess(before["_norm_dense_score"](.1), before["_norm_dense_score"](.9))
        self.assertGreater(faiss_l2_relevance(.1), faiss_l2_relevance(.9))
        self.assertEqual(bge_score_list({"colbert+sparse+dense": .8}, 1), [.8])
        for malformed in ({"dense": [.9]}, {"colbert+sparse+dense": [.9]}, {"colbert+sparse+dense": [.9, float("nan")]}):
            with self.assertRaises(ValueError):
                bge_score_list(malformed, 2)

    def test_single_fusion_definition_and_duplicate_source_weights(self):
        nb = json.loads((ROOT / "LangGraph.ipynb").read_text(encoding="utf-8"))
        definitions = [node for cell in nb["cells"] if cell["cell_type"] == "code"
                       for node in ast.parse("".join(cell["source"])).body
                       if isinstance(node, ast.FunctionDef) and node.name == "fuse_results"]
        self.assertEqual(len(definitions), 1)
        def h(origin, score):
            return {"title": "same", "url": "same", "origin": origin, "score": score}
        result = self.ns["fuse_results"]({"dense_hits": [h("dense", .8), h("dense", .7)],
                                          "bm25_hits": [h("bm25", 1.)], "web_hits": [h("web", .5)]})
        self.assertEqual(len(result["fused_hits"]), 1)
        self.assertAlmostEqual(result["fused_hits"][0]["score"], .825)

    def test_all_chart_date_axes_are_bounded_by_observable_close(self):
        state, _ = invoke_graph(self.ns, {**REQUEST, "as_of": WAITING})
        def plotted_max(ns):
            dates = []
            original = Axes.plot
            def recording(ax, *args, **kwargs):
                if args and pd.api.types.is_datetime64_any_dtype(getattr(args[0], "dtype", None)):
                    dates.extend(pd.to_datetime(args[0]).tolist())
                return original(ax, *args, **kwargs)
            with tempfile.TemporaryDirectory(dir=ROOT) as tmp, patch.object(Axes, "plot", recording), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                files = ns["save_visuals"](state, ns["price_df"], out_dir=tmp)
                self.assertTrue(files)
                self.assertTrue(all(Path(p).read_bytes().startswith(b"\x89PNG") for p in files))
            self.assertTrue(dates)
            return max(dates).date().isoformat()
        before = load_notebook(BASELINE)
        self.assertEqual(plotted_max(before), "2025-01-08")
        self.assertEqual(plotted_max(self.ns), "2025-01-06")


if __name__ == "__main__":
    unittest.main(verbosity=2)
