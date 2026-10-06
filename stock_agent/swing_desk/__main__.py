"""python -m stock_agent.swing_desk <command>  (advisory VN30 swing desk; never places orders)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from .config import DeskPaths


def _now(text: str | None) -> datetime:
    if not text:
        return datetime.now(timezone.utc)
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise SystemExit("--now must include a timezone, e.g. 2026-10-06T17:00:00+07:00")
    return value


def _paths(args) -> DeskPaths:
    repo = Path(args.repo).resolve()
    data_root = Path(args.data_root or os.environ.get("SWING_DESK_DATA_ROOT") or repo).resolve()
    desk = Path(args.desk).resolve() if args.desk else repo / "swing_desk"
    return DeskPaths(repo=repo, data_root=data_root, desk=desk)


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m stock_agent.swing_desk", description=__doc__)
    p.add_argument("--repo", default=".", help="repository root (configs/, swing_desk/)")
    p.add_argument("--data-root", help="root holding data/paper/snapshots and data/foreign_flows_v2 "
                                       "(default: --repo or $SWING_DESK_DATA_ROOT)")
    p.add_argument("--desk", help="desk folder (default <repo>/swing_desk)")
    p.add_argument("--now", help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status", help="read-only freshness report")
    s.add_argument("--no-flows", action="store_true", help="skip foreign-flow health (faster)")
    r = sub.add_parser("refresh", help="reuse/fetch verified snapshot, optional flows, build desk state")
    r.add_argument("--with-flows", action="store_true", help="collect foreign flows if stale (not 16:30-18:30 VN)")
    c = sub.add_parser("check", help="pre-trade check for one ticker")
    c.add_argument("ticker")
    c.add_argument("--price", type=float, help="live price you see now (VND)")
    c.add_argument("--amount", type=float, help="amount you intend to buy (VND)")
    c.add_argument("--reason", help="S1, S2, A-code, news, forum, tip, kol, group, other")
    c.add_argument("--emotion", help="calm, confident, fomo, fear, revenge, hope, hype, bored, other")
    c.add_argument("--ack", action="store_true", help="acknowledge a WARN after writing a THESIS line")
    c.add_argument("--nav", type=float, help="override NAV (VND) for sizing")
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-log", action="store_true", help="do not append to checks.jsonl")
    sub.add_parser("scan", help="gate table for all VN30 from current desk state")
    v = sub.add_parser("review", help="portfolio review of swing_desk/portfolio.txt")
    v.add_argument("--show-nav", action="store_true")
    v.add_argument("--json", action="store_true")
    f = sub.add_parser("freeze", help="append frozen recommendation rows (before 09:00 VN)")
    f.add_argument("--source", required=True, choices=["baseline", "desk", "single"],
                   help="baseline freezes baseline+control by code; desk/single need --file")
    f.add_argument("--file", type=Path, help="recommendations.json (array) for desk/single")
    f.add_argument("--model-id")
    f.add_argument("--n-rollouts", type=int, default=0)
    f.add_argument("--adjudication-base")
    f.add_argument("--open-forks", type=int)
    sub.add_parser("score", help="score matured rows against the newest verified snapshot")
    sub.add_parser("verify-log", help="verify recommendation + checks hash chains")
    e = sub.add_parser("events", help="record that the event calendar was checked for a ticker (G10)")
    e.add_argument("ticker")
    e.add_argument("--session", required=True, help="trade session the check covers (YYYY-MM-DD)")
    e.add_argument("--event", action="append", default=[], help="YYYY-MM-DD:type (repeatable), e.g. 2026-10-09:ex_dividend")
    e.add_argument("--source", required=True, help="URL or document the calendar came from")
    return p


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)
    paths = _paths(args)
    now = _now(args.now)
    from . import baseline, render, review, service
    try:
        if args.cmd == "status":
            _print(service.status(paths, now, with_flows=not args.no_flows))
        elif args.cmd == "refresh":
            from .refresh import refresh
            _print(refresh(paths, now=now, with_flows=args.with_flows))
        elif args.cmd == "check":
            result = service.check(paths, args.ticker, now=now, price=args.price, amount=args.amount,
                                   reason=args.reason, emotion=args.emotion, ack=args.ack, nav=args.nav,
                                   record=not args.no_log)
            if args.json:
                _print(result)
            else:
                print(render.render_check(result))
            return {"PASS": 0, "WARN": 1, "BLOCK": 2}[result["verdict"]]
        elif args.cmd == "scan":
            fresh = service.state_freshness(paths, now)
            if not fresh["ok"]:
                print(f"G0 BLOCK: {fresh['reason']}")
                return 2
            state = service.load_state(paths)
            print(service.gate_table(state))
        elif args.cmd == "review":
            rep = review.build_review(paths, now=now, show_nav=args.show_nav)
            folder, manifest = service.read_state(paths)
            if folder is not None and not rep.get("empty") and not args.show_nav:
                (folder / "portfolio_review.md").write_text(rep["markdown"], encoding="utf-8")
            if args.json:
                _print({k: v for k, v in rep.items() if k != "markdown"} if args.show_nav else
                       {k: rep.get(k) for k in ("as_of", "closed_summary", "by_gate", "by_emotion", "activity", "flags")})
            else:
                print(rep["markdown"])
        elif args.cmd == "freeze":
            if args.source == "baseline":
                out = baseline.freeze_code_arms(paths, now=now)
                _print({k: [r["ticker"] for r in v] for k, v in out.items()})
            else:
                if not args.file or not args.model_id:
                    raise ValueError("desk/single freeze needs --file and --model-id")
                rows = json.loads(args.file.read_text(encoding="utf-8"))
                adj = ({"base": args.adjudication_base, "open_forks": args.open_forks}
                       if args.adjudication_base or args.open_forks is not None else None)
                out = baseline.freeze_analyst(paths, rows, source=args.source, now=now, model_id=args.model_id,
                                              n_rollouts=args.n_rollouts, adjudication=adj)
                _print([r["id"] for r in out])
        elif args.cmd == "score":
            res = baseline.score_log(paths, now=now)
            _print({"as_of": res["as_of"], "chain": res["chain"], "summary": res["summary"],
                    "user_checks_by_verdict": res["user_checks_by_verdict"]})
        elif args.cmd == "verify-log":
            from . import reclog
            res = {"recommendations": reclog.verify_chain(paths.rec_log), "checks": reclog.verify_chain(paths.checks_log)}
            _print(res)
            return 0 if all(v["ok"] for v in res.values()) else 2
        elif args.cmd == "events":
            events = []
            for item in args.event:
                day, _, kind = item.partition(":")
                datetime.fromisoformat(day)
                events.append({"date": day, "type": kind or "event"})
            _print(service.mark_events(paths, args.ticker, args.session, events, args.source))
        return 0
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
