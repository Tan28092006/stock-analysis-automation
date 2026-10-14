"""Agent-supplied, cited facts for bottom-fish checks (T12 spec §7). The desk never fetches news.

`swing_desk/events/bf_news.json` (git-ignored): {TICKER: {checked_session: record}}; a re-mark
replaces only that ticker/session. Every record carries `written_at` and `source_sha` (sha256 of
the canonical record). Items without an http(s) URL are rejected. Event resolution (EGM result,
final issue terms) is recorded in `events.json` under `resolved` by `mark_resolved`.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import re
from urllib.parse import urlparse

from . import reclog
from .config import DeskPaths, canonical, sha256_bytes

EVIDENCE = {"FACT_PRIMARY", "FACT_REPORTED", "OPINION", "RUMOR"}
CLASSES = {"fact_negative", "none_found", "opinion_only"}
PRIMARY_HOSTS = ("hsx.vn", "hose.vn", "hnx.vn", "ssc.gov.vn", "vsd.vn", "vsdc.com.vn")
MAX_QUOTE = 200


def _date(text: str, name: str) -> str:
    try:
        return str(date.fromisoformat(text.strip()))
    except ValueError:
        raise ValueError(f"{name}: invalid date {text!r} (use YYYY-MM-DD)") from None


def _url(text: str, name: str = "url") -> str:
    text = (text or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name}: an opened http(s) source URL is required, got {text!r}")
    return text


def parse_item(text: str) -> dict:
    """`date|evidence_type|impact|type|url|quote<=200`."""
    parts = [p.strip() for p in text.split("|")]
    if len(parts) < 5:
        raise ValueError(f"item needs 6 '|'-separated fields (date|evidence_type|impact|type|url|quote), got {len(parts)}")
    parts += [""] * (6 - len(parts))
    day, ev, impact, kind, url, quote = parts[0], parts[1], parts[2], parts[3], parts[4], "|".join(parts[5:])
    return validate_item({"date": day, "evidence_type": ev, "impact": impact, "type": kind, "url": url, "quote": quote})


def validate_item(item: dict) -> dict:
    ev = str(item.get("evidence_type", "")).upper()
    if ev not in EVIDENCE:
        raise ValueError(f"evidence_type {item.get('evidence_type')!r}: expected one of {sorted(EVIDENCE)}")
    try:
        impact = int(str(item.get("impact")).strip())
    except ValueError:
        raise ValueError(f"impact {item.get('impact')!r}: expected an integer -2..+2") from None
    if not -2 <= impact <= 2:
        raise ValueError(f"impact {impact}: expected -2..+2")
    kind = str(item.get("type", "")).strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{2,40}", kind):
        raise ValueError(f"type {item.get('type')!r}: expected a short code such as provision, dilution, margin_cut")
    quote = str(item.get("quote", "")).strip()
    if len(quote) > MAX_QUOTE:
        raise ValueError(f"quote longer than {MAX_QUOTE} characters")
    return {"date": _date(str(item.get("date", "")), "date"), "evidence_type": ev, "impact": impact, "type": kind,
            "url": _url(str(item.get("url", ""))), "quote": quote}


def parse_insider(text: str) -> dict:
    """`YYYY-MM-DD..YYYY-MM-DD|shares|url` (active insider / related-party sell registration)."""
    parts = [p.strip() for p in text.split("|")]
    if len(parts) != 3 or ".." not in parts[0]:
        raise ValueError("insider sell needs 'start..end|shares|url'")
    start, _, end = parts[0].partition("..")
    try:
        shares = float(parts[1].replace(",", "").replace("_", ""))
    except ValueError:
        raise ValueError(f"insider shares {parts[1]!r}: not a number") from None
    return {"start": _date(start, "insider start"), "end": _date(end, "insider end"), "shares": shares,
            "url": _url(parts[2])}


def parse_margin(text: str) -> dict:
    """`hose_ineligible=yes|no|broker_cut=<date|none>|status=normal|warning|control|suspended|checked=<date>|<url>...`."""
    out = {"hose_ineligible": None, "broker_cut": None, "status": "normal", "checked": None, "urls": []}
    for part in (p.strip() for p in text.split("|") if p.strip()):
        key, sep, value = part.partition("=")
        if sep and not part.lower().startswith("http"):
            key, value = key.strip().lower(), value.strip()
            if key == "hose_ineligible":
                if value.lower() not in {"yes", "no"}:
                    raise ValueError("margin hose_ineligible must be yes or no")
                out["hose_ineligible"] = value.lower() == "yes"
            elif key == "broker_cut":
                out["broker_cut"] = None if value.lower() in {"", "none", "no"} else _date(value, "margin broker_cut")
            elif key == "status":
                if value.lower() not in {"normal", "warning", "control", "suspended"}:
                    raise ValueError("margin status must be normal, warning, control or suspended")
                out["status"] = value.lower()
            elif key == "checked":
                out["checked"] = _date(value, "margin checked")
            else:
                raise ValueError(f"unknown margin key {key!r}")
        else:
            out["urls"].append(_url(part, "margin url"))
    if out["hose_ineligible"] is None or out["checked"] is None:
        raise ValueError("margin needs hose_ineligible= and checked=")
    if not out["urls"]:
        raise ValueError("margin needs at least one source url")
    return out


IR_DOMAINS = "configs/swing_desk/company_ir_domains.json"


def load_ir_domains(paths: DeskPaths) -> dict:
    """{TICKER: [domain, ...]} from the repo's company IR map; empty when the file is absent."""
    try:
        data = json.loads((paths.repo / IR_DOMAINS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(t).upper(): [str(d).lower().strip(".") for d in hosts]
            for t, hosts in (data.get("domains") or {}).items()}


def _on_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def is_primary(url: str, ticker: str, ir_domains: dict | None = None) -> bool:
    """Exchange/regulator host, the ticker's mapped company domain, or (legacy) a host label
    starting with the ticker. Domains match exactly or as a parent of a subdomain, never as a
    substring, so look-alike hosts such as evilhsx.vn do not count."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if any(_on_domain(host, d) for d in PRIMARY_HOSTS):
        return True
    if any(_on_domain(host, d) for d in (ir_domains or {}).get(ticker.upper(), [])):
        return True
    return any(label.startswith(ticker.lower()) for label in host.split("."))


def load_bf_news(paths: DeskPaths) -> dict:
    if not paths.bf_news.is_file():
        return {}
    try:
        return json.loads(paths.bf_news.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def latest_news(store: dict, ticker: str) -> dict | None:
    sessions = (store or {}).get(ticker.upper()) or {}
    if not sessions:
        return None
    return sessions[max(sessions)]


def file_sha(path) -> str | None:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return None


def mark_bf_news(paths: DeskPaths, ticker: str, session: str, *, news_class: str, items=(), insider_sell=(),
                 margin: dict | None = None, searched=(), now: datetime) -> dict:
    ticker = ticker.upper()
    session = _date(session, "session")
    if news_class not in CLASSES:
        raise ValueError(f"class {news_class!r}: expected one of {sorted(CLASSES)}")
    items = [validate_item(i) for i in items]
    searched = [_url(u, "searched url") for u in searched]
    if news_class == "none_found":
        if len(searched) < 2:
            raise ValueError("class none_found needs >= 2 --searched URLs (sources actually opened)")
        domains = load_ir_domains(paths)
        if not any(is_primary(u, ticker, domains) for u in searched):
            raise ValueError("class none_found needs at least one primary source (HOSE/HNX/SSC disclosure page "
                             "or the company's own IR site) among --searched")
    if news_class == "fact_negative" and not items:
        raise ValueError("class fact_negative needs at least one --item")
    record = {"ticker": ticker, "checked_session": session, "class": news_class, "items": items,
              "insider_sell": list(insider_sell), "margin": margin, "searched": searched,
              "written_at": now.astimezone(timezone.utc).isoformat()}
    if margin is None:
        record.pop("margin")
    record["source_sha"] = sha256_bytes(canonical(record))
    store = load_bf_news(paths)
    store.setdefault(ticker, {})[session] = record
    paths.bf_news.parent.mkdir(parents=True, exist_ok=True)
    reclog._atomic_write(paths.bf_news, json.dumps(store, ensure_ascii=False, indent=2, sort_keys=True))
    return record


def mark_resolved(paths: DeskPaths, ticker: str, session: str, resolved: list[dict], source: str) -> dict:
    """Record that an event (EGM result, final issue terms) is resolved, with its source."""
    from .service import load_events
    if not (source or "").strip():
        raise ValueError("--resolved needs --source (the published result / terms)")
    source = _url(source, "source") if source.startswith("http") else source.strip()
    ticker = ticker.upper()
    session = _date(session, "session")
    data = load_events(paths) or {"checked": {}, "events": []}
    bucket = data.setdefault("resolved", {}).setdefault(ticker, [])
    for r in resolved:
        entry = {"date": _date(r["date"], "resolved date"), "type": r.get("type") or "event", "source": source,
                 "recorded_session": session}
        bucket[:] = [x for x in bucket if (x["date"], x["type"]) != (entry["date"], entry["type"])] + [entry]
    paths.events.parent.mkdir(parents=True, exist_ok=True)
    reclog._atomic_write(paths.events, json.dumps(data, ensure_ascii=False, indent=2))
    return data
