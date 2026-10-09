"""Article provenance carried with numeric comparisons, without network lookups."""
import hashlib
import json
from urllib.parse import urlsplit

from price_windows import market_timestamp


def text_value(value):
    """Do not turn missing CSV values into the visible source label 'nan'."""
    return value.strip() if isinstance(value, str) else ""


def article_source(hit):
    text = text_value(hit.get("text")) or text_value(hit.get("title"))
    title = text_value(hit.get("title")) or (text.splitlines()[0] if text else "제목 미확인")
    raw_url = text_value(hit.get("url"))
    url = None
    try:
        parts = urlsplit(raw_url)
        if (parts.scheme.lower() in ("http", "https") and parts.netloc and not parts.username
                and not parts.password and not any(ch.isspace() for ch in raw_url)):
            url = raw_url
    except ValueError:
        pass
    try:
        date = market_timestamp(hit.get("date")).tz_convert("UTC").isoformat()
    except (TypeError, ValueError):
        date = None
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = json.dumps({"date": date, "url": url, "text_sha256": text_hash}, sort_keys=True, ensure_ascii=False)
    return {"source_id": "src-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
            "date": date, "title": title, "url": url,
            "url_status": "recorded_not_fetched" if url else "missing_or_invalid",
            "summary": text[:400], "text_sha256": text_hash}


def document_source(document):
    return article_source({**document.metadata, "text": document.page_content})


def report_source_line(source):
    title = source["title"].replace("\n", " ").replace("|", "\\|")
    return f"[{source['source_id']}] {source['date'] or '시각 미확인'} | {title} | {source['url'] or 'URL 미확인'}"
