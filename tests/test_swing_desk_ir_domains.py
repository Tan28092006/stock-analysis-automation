"""Primary-source recognition for bf-news: exchange/regulator hosts plus a per-ticker company IR
domain map (configs/swing_desk/company_ir_domains.json), matched on the registrable domain or a
subdomain of it, never on a substring."""
from datetime import datetime
import json
from pathlib import Path

import pytest

from stock_agent.data.exchange_calendar import VN_TIMEZONE
from stock_agent.swing_desk import bf_facts
from stock_agent.swing_desk.config import DeskPaths

NOW = datetime(2026, 10, 8, 10, 0, tzinfo=VN_TIMEZONE)
REPO = Path(__file__).resolve().parents[1]


def desk(tmp_path, domains=None):
    if domains is not None:
        cfg = tmp_path / "configs" / "swing_desk"
        cfg.mkdir(parents=True)
        (cfg / "company_ir_domains.json").write_text(
            json.dumps({"version": 1, "domains": domains}), encoding="utf-8")
    return DeskPaths(repo=tmp_path, data_root=tmp_path, desk=tmp_path / "swing_desk")


def test_company_domain_without_ticker_in_host_counts_when_mapped(tmp_path):
    paths = desk(tmp_path, {"VRE": ["vincom.com.vn"], "SSB": ["seabank.com.vn"]})
    rec = bf_facts.mark_bf_news(paths, "VRE", "2026-10-08", news_class="none_found",
                                searched=["http://ir.vincom.com.vn/", "https://cafef.vn/a"], now=NOW)
    assert rec["class"] == "none_found"
    assert bf_facts.is_primary("https://cloud-cdn.seabank.com.vn/x.pdf", "SSB", bf_facts.load_ir_domains(paths))


def test_domain_belongs_only_to_its_ticker(tmp_path):
    paths = desk(tmp_path, {"VRE": ["vincom.com.vn"], "VHM": ["vinhomes.vn"]})
    with pytest.raises(ValueError, match="primary"):
        bf_facts.mark_bf_news(paths, "VHM", "2026-10-08", news_class="none_found",
                              searched=["http://ir.vincom.com.vn/", "https://cafef.vn/a"], now=NOW)


@pytest.mark.parametrize("url", [
    "https://evilhsx.vn/a",               # suffix without a dot boundary
    "https://hsx.vn.attacker.com/a",      # exchange name inside another domain
    "https://notvincom.com.vn/a",         # mapped domain as a substring
])
def test_lookalike_hosts_are_not_primary(tmp_path, url):
    domains = bf_facts.load_ir_domains(desk(tmp_path, {"VRE": ["vincom.com.vn"]}))
    assert not bf_facts.is_primary(url, "VRE", domains)


def test_exchange_hosts_and_ticker_fallback_still_work(tmp_path):
    domains = bf_facts.load_ir_domains(desk(tmp_path))          # no map file -> empty map
    assert domains == {}
    assert bf_facts.is_primary("https://www.hsx.vn/Modules/x", "HPG", domains)
    assert bf_facts.is_primary("https://vsd.vn/a", "HPG", domains)
    assert bf_facts.is_primary("https://hpg.com.vn/x", "HPG", domains)      # legacy: ticker in host label
    assert not bf_facts.is_primary("https://cafef.vn/hpg-news", "HPG", domains)


def test_repo_map_covers_current_vn30_and_held_tickers():
    data = json.loads((REPO / "configs/swing_desk/company_ir_domains.json").read_text(encoding="utf-8"))
    universe = json.loads((REPO / "configs/universe_vn30.json").read_text(encoding="utf-8"))
    symbols = universe if isinstance(universe, list) else universe.get("symbols") or universe.get("universe")
    missing = [s for s in symbols if not data["domains"].get(s)]
    assert not missing, missing
    for ticker, hosts in data["domains"].items():
        for host in hosts:
            assert "/" not in host and ":" not in host and host == host.lower(), (ticker, host)
