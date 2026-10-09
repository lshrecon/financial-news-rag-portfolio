"""Run the notebook's real graph with synthetic data and deterministic model doubles.

No notebook setup/import/sample-execution cell is executed. This intentionally
does not load credentials, existing datasets, embedding weights, or an API client.
Run with the original torch312 environment; see INTEGRATION-NOTES.md.
"""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import datetime as datetime_module
import ast
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
from types import SimpleNamespace
import typing
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
for name in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
    os.environ[name] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
# Allow an isolated cache when the existing OneDrive cache is slow/unavailable.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import numpy as np
import pandas as pd
import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties, fontManager
_korean_font = Path("C:/Windows/Fonts/malgun.ttf")
if _korean_font.exists():
    fontManager.addfont(str(_korean_font))
    plt.rcParams["font.family"] = FontProperties(fname=str(_korean_font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from price_windows import event_price_window, market_timestamp, observable_price_rows
from retrieval_contracts import visible_news, faiss_l2_relevance, bge_score_list, retrieval_fetch_k
from macro_windows import macro_window_summary
from source_evidence import article_source, document_source, report_source_line


@contextmanager
def block_network():
    """Fail even an accidental network attempt during the experiment."""
    attempts = []
    def denied(*args, **kwargs):
        attempts.append("blocked socket operation")
        raise AssertionError("Network is disabled in the offline integration run")
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), \
         patch.object(socket, "create_connection", denied), patch.object(socket, "getaddrinfo", denied):
        yield attempts


class SyntheticEmbeddings(Embeddings):
    """Fixed geometry exercises actual FAISS L2 filtering; no semantic model."""
    def embed_documents(self, texts):
        return [[0.01 if "FUTURE_DO_NOT_USE" in t else 0.60 if "CURRENT_EVENT" in t
                 else 0.80 if "PAST_EVENT" in t else 2.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [0.0, 0.0]


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(str(prompt))
        if "재작성(키워드)" in prompt:
            return SimpleNamespace(content="SPY policy")
        return SimpleNamespace(content="[OFFLINE_STUB] Synthetic report text. Model quality was not evaluated.")


class FakeBGE:
    """Match FlagEmbedding 1.3.5's documented/installed mapping return shape."""
    def __init__(self):
        self.calls = []

    def compute_score(self, pairs, **kwargs):
        self.calls.append(pairs)
        scores = [0.99 if "CURRENT_EVENT" in text else 0.80 if "PAST_EVENT" in text
                  else 0.999 if "FUTURE_DO_NOT_USE" in text else 0.05 for _, text in pairs]
        return {mode: scores[:] for mode in
                ("colbert", "sparse", "dense", "sparse+dense", "colbert+sparse+dense")}


def fixtures():
    def doc(title, text, date, key):
        return Document(page_content=text, metadata={"title": title, "date": date,
                         "url": f"https://example.invalid/{key}", "source": "SYNTHETIC"})
    docs = [doc(f"Synthetic future {i}", "SPY FUTURE_DO_NOT_USE policy", "2025-01-08T22:00:00Z", f"future-{i}") for i in range(25)]
    docs += [doc("Synthetic current SPY news", "SPY CURRENT_EVENT policy", "2025-01-06T22:00:00Z", "current"),
             doc("Synthetic historical SPY news", "SPY PAST_EVENT policy", "2024-12-23T22:00:00Z", "past")]
    docs += [doc(f"Synthetic unrelated {i}", f"weather topic{i} unrelated", "2025-01-02T12:00:00Z", f"noise-{i}") for i in range(15)]
    prices = pd.DataFrame({"ticker": ["SPY"] * 8,
                           "date_kst": pd.to_datetime(["2024-12-20", "2024-12-23", "2024-12-24", "2024-12-26",
                                                       "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]),
                           "adj_close": [100., 100., 102., 101., 100., 90., 99., 108.]})
    web = {"title": "Synthetic current SPY news", "description": "SPY CURRENT_EVENT policy",
           "pubDate": "Mon, 06 Jan 2025 22:00:00 +0000", "link": "https://example.invalid/current",
           "originallink": "https://example.invalid/current"}
    return docs, prices, [web, dict(web)]


def load_notebook(path=None, *, documents=None):
    """Select only definitions, then execute the notebook's graph-building cell."""
    path = Path(path or ROOT / "LangGraph.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    docs, prices, web_items = fixtures()
    if documents is not None:
        docs = documents
    # An empty-store scenario still uses the real store; filter removes its sole undated dummy.
    index_docs = docs or [Document(page_content="empty", metadata={})]
    store = FAISS.from_documents(index_docs, SyntheticEmbeddings())
    llm, bge = FakeLLM(), FakeBGE()
    cfg = SimpleNamespace(use_naver=True, naver_max_pages=1, naver_page_size=20, ensemble_topN=80,
                          rerank_topk=12, rrf_k=75, fusion_method="weighted_sum",
                          reranker_mode="colbert+sparse+dense", thread_id="offline",
                          embedding_backend="synthetic_fixed_vectors", reranker_backend="fake_bge_mapping")
    ns = {**typing.__dict__, "__name__": "offline_notebook", "os": os, "json": json,
          "pd": pd, "np": np, "re": re, "html": html, "math": math, "pytz": pytz,
          "datetime": datetime, "timezone": timezone, "timedelta": timedelta, "_dt": datetime_module,
          "plt": plt, "mdates": mdates, "time": SimpleNamespace(sleep=lambda seconds: None),
          "StateGraph": StateGraph, "START": START, "END": END, "MemorySaver": MemorySaver,
          "Document": Document, "BM25Okapi": BM25Okapi, "CFG": cfg,
          "event_price_window": event_price_window, "market_timestamp": market_timestamp,
          "observable_price_rows": observable_price_rows,
          "visible_news": visible_news, "faiss_l2_relevance": faiss_l2_relevance,
          "bge_score_list": bge_score_list, "retrieval_fetch_k": retrieval_fetch_k,
          "macro_window_summary": macro_window_summary,
          "article_source": article_source, "document_source": document_source,
          "report_source_line": report_source_line,
          "llm": llm, "bge_m3": bge, "vectorstore": store, "LOCAL_DOCS": docs,
          "price_df": prices, "KNOWN_TICKERS": {"SPY"}, "KEYWORD_TO_TICKERS": {},
          "HIST_TOPK": 3, "HIST_MIN_DAYS_GAP": 7, "_TAG_RE": re.compile(r"<[^>]+>"),
          "_token_pat": re.compile(r"[0-9A-Za-z가-힣]+"),
          "etf_profile_table": {"SPY": {"ticker": "SPY", "name": "Synthetic profile"}},
          "macro_df": pd.DataFrame({"date": pd.to_datetime(["2025-01-06", "2025-01-07"]),
                                     "SPX_close": [100., 110.]})}
    definitions, graph_source = [], None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef) and node.name == "GraphState":
                definitions.append(node)
        if source.startswith("builder = StateGraph(GraphState)"):
            graph_source = source
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), ns)
    ns["_BM25"] = BM25Okapi([ns["tokenize_ko"](d.page_content) for d in docs]) if docs else None
    calls = []
    def fake_naver(**kwargs):
        calls.append(kwargs)
        return {"items": web_items if docs else []}
    ns["_naver_news_search"] = fake_naver
    ns["fake_web_calls"] = calls
    if graph_source is None:
        raise ValueError("Notebook graph-building cell missing")
    exec(compile(graph_source, str(path) + ":graph", "exec"), ns)
    ns["graph"] = ns["builder"].compile(checkpointer=ns["memory"])
    return ns


def invoke_graph(ns, request, *, thread_id="offline"):
    config = {"configurable": {"thread_id": thread_id}}
    updates = list(ns["graph"].stream(request, config=config, stream_mode="updates"))
    final_state = dict(ns["graph"].get_state(config).values)
    return final_state, [name for update in updates for name in update]


def environment_versions():
    names = ["langgraph", "langgraph-checkpoint", "langchain-core", "langchain-community",
             "FlagEmbedding", "rank-bm25", "faiss-cpu", "pandas", "numpy", "matplotlib"]
    return {name: importlib.metadata.version(name) for name in names}


if __name__ == "__main__":
    output = ROOT / "offline-evidence"
    output.mkdir(exist_ok=True)
    with block_network() as attempts:
        ns = load_notebook()
        state, trace = invoke_graph(ns, {"question": "SPY policy", "as_of": "2025-01-07T22:00:00Z", "horizon_sessions": 1})
        figures = ns["save_visuals"](state, ns["price_df"], out_dir=str(output / "figs"))
        report = ns["save_report_md"](state, out_dir=str(output), prefix="SYNTHETIC_ONLY", fig_paths=figures)
        logs = ns["save_run_logs"](state, out_dir=str(output / "logs"), prefix="SYNTHETIC_ONLY")
        waiting, waiting_trace = invoke_graph(ns, {"question": "SPY policy", "as_of": "2025-01-07T20:00:00Z", "horizon_sessions": 1}, thread_id="waiting")
        empty, empty_trace = invoke_graph(ns, {"question": "", "as_of": "2025-01-07T22:00:00Z", "horizon_sessions": 1})
    evidence = {"validation_scope": "real LangGraph/BM25/FAISS; synthetic embeddings, news, prices and fake LLM/BGE/web",
                "versions": environment_versions(), "trace_complete": trace, "trace_waiting": waiting_trace,
                "trace_empty_same_thread": empty_trace, "complete_stats": state.get("etf_stats"),
                "waiting_status": waiting.get("price_window_status"), "empty_stats": empty.get("etf_stats"),
                "report": str(Path(report).relative_to(ROOT)), "figure_count": len(figures),
                "llm_fake_calls": len(ns["llm"].prompts), "web_fake_calls": len(ns["fake_web_calls"]),
                "external_calls": 0, "blocked_network_attempts": len(attempts)}
    (output / "integration-run.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
