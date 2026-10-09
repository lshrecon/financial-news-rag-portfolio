"""Small, explicit contracts shared by notebook retrieval and offline checks."""
import math
from numbers import Real

from price_windows import market_timestamp


def visible_news(metadata, as_of, *, before=None):
    """Undated/ambiguous/future publication timestamps are not usable evidence."""
    try:
        published = market_timestamp(metadata.get("date"))
        cutoff = market_timestamp(as_of)
        if before is not None:
            cutoff = min(cutoff, market_timestamp(before))
        return published <= cutoff
    except (ValueError, TypeError):
        return False


def faiss_l2_relevance(distance):
    """The notebook uses FAISS IndexFlatL2: smaller squared distance is better.

    This monotonic transform is a ranking heuristic, not a probability.
    """
    distance = float(distance)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("Expected a finite non-negative FAISS L2 distance")
    return 1.0 / (1.0 + distance)


def bge_score_list(result, expected_count, mode="colbert+sparse+dense"):
    """FlagEmbedding 1.3.5 returns a mode->scores mapping, not a score list."""
    if not isinstance(result, dict) or mode not in result:
        raise ValueError("BGE score mapping is missing the selected mode")
    values = result[mode]
    if isinstance(values, Real):
        values = [values]
    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("BGE scores must be numeric") from exc
    if len(values) != expected_count or not all(math.isfinite(v) for v in values):
        raise ValueError("BGE scores must match the candidate count and be finite")
    return values


def retrieval_fetch_k(store, minimum):
    """Scan this small local FAISS index before metadata filtering/top-k.

    FAISS filtering is post-vector-search. Fetching only k can lose all valid
    historical documents when future documents occupy the nearest k slots.
    """
    return max(minimum, int(store.index.ntotal))
