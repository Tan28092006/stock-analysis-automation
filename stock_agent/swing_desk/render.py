"""Vietnamese plain-text rendering for chat/terminal."""
from __future__ import annotations

RULE_LABEL = {
    "G0": "dữ liệu tươi", "G1": "thuộc VN30", "G2": "thị trường (VNINDEX vs EMA50)", "G3": "giá > SMA50",
    "G4": "không đuổi giá", "G5": "đuổi trong phiên", "G6": "stop so với nhiễu", "G7": "thanh khoản",
    "G8": "R:R", "G9": "ngân sách rủi ro", "G10": "sự kiện", "G11": "có setup",
    "B1": "FOMO/trần", "B2": "đuổi gap", "B3": "gỡ lỗ (revenge)", "B3P": "nghỉ sau 2 lệnh lỗ liên tiếp",
    "B4": "bình quân giá xuống", "B5": "tăng size", "B6": "hype", "B7": "giao dịch quá nhiều",
    "COOL": "cooling-off (đã BLOCK trong phiên)",
}


def vnd(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}".replace(",", ".")


def pct(value, digits=1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}%".replace(".", ",")


def mil(value) -> str:
    return "n/a" if value is None else f"{value / 1e6:,.1f} tr".replace(",", "X").replace(".", ",").replace("X", ".")


def render_check(r: dict) -> str:
    sha = (r.get("snapshot_manifest_sha256") or "")[:12]
    lines = [f"{r['ticker']} — {r['verdict']}   (dữ liệu EOD tới {r.get('data_as_of') or 'n/a'}, snapshot sha {sha or 'n/a'})",
             f"Phiên giao dịch: {r['trade_session']}. {r.get('eod_note', '')}"]
    bad = [x for x in r["rules"] if x["status"] in ("FAIL", "UNVERIFIED")]
    if bad:
        lines.append("Luật không đạt / chưa kiểm:")
        for x in bad:
            tag = "BLOCK" if x["status"] == "FAIL" and x["severity"] == "hard" else (
                "WARN" if x["status"] == "FAIL" else "UNVERIFIED")
            lines.append(f"  - {x['id']} {RULE_LABEL.get(x['id'], '')} [{tag}]: {x['reason']}")
    else:
        lines.append("Luật không đạt: không có")
    setups = r.get("setups") or []
    lines.append(f"Setup: {'+'.join(setups) if setups else 'không có (G11) — bạn đang mua không có tín hiệu'}")
    lines.append(f"Bán được sớm nhất: chiều {r['earliest_sell']} (T+2). Stop 3% KHÔNG thực thi được trước thời điểm này.")
    p = r.get("plan")
    if p:
        lines.append(f"Giá tham chiếu: {vnd(p['entry_ref'])} ({'giá bạn nhập' if r.get('price') else 'đóng cửa gần nhất'})")
        lines.append(f"Stop 3%: {vnd(p['stop_user'])} | stop đã kiểm định (RSI2, −8%): {vnd(p['stop_validated'])}")
        if r.get("target"):
            rr = r.get("reward_risk")
            lines.append(f"Target: {vnd(r['target'])} | R:R {('%.2f' % rr) if rr is not None else 'n/a'}")
        lines.append(f"Size gợi ý (rủi ro {pct(p['risk_budget_vnd'] / p['nav'] * 100)} NAV, lỗ hiệu dụng "
                     f"{pct(p['effective_loss_pct'])}): {vnd(p['qty'])} cp ≈ {mil(p['position_value'])}"
                     + (f" (giới hạn bởi {p['capped_by']})" if p.get("capped_by") else ""))
        lines.append(f"  VND rủi ro: ở stop 3% {mil(p['vnd_at_risk_nominal'])}; hiệu dụng (3% + gap + phí) "
                     f"{mil(p['vnd_at_risk_effective'])}. Nếu dùng stop −8%: {vnd(p['qty_if_validated_stop'])} cp")
        req = p.get("requested")
        if req:
            lines.append(f"Số tiền bạn nhập {mil(req['amount'])} → {vnd(req['qty'])} cp: rủi ro ở stop 3% "
                         f"{mil(req['vnd_at_risk_nominal'])} ({pct(req['risk_nominal_pct_nav'], 2)} NAV); hiệu dụng "
                         f"{mil(req['vnd_at_risk_effective'])} ({pct(req['risk_effective_pct_nav'], 2)} NAV)"
                         + (" > ngân sách" if req["over_budget"] else ""))
        lines.append(f"NAV dùng để tính: {r.get('nav_source')}")
    if r.get("exit_plan"):
        lines.append(f"Thoát: {r['exit_plan']}")
    fl = r.get("flows") or {}
    if fl.get("flow_status"):
        lines.append(f"Dòng tiền NN (chú thích A4, không đổi verdict): {fl.get('flow_status')}"
                     + (f", ròng 5 phiên {fl['f_net_5d_bn']:.1f} tỷ" if fl.get("f_net_5d_bn") is not None else ""))
    if r.get("horizon_warning"):
        lines.append(f"Cảnh báo lịch: {r['horizon_warning']}")
    if r.get("journal_errors"):
        lines.append(f"portfolio.txt có {len(r['journal_errors'])} dòng lỗi (dòng "
                     f"{', '.join(str(e['line']) for e in r['journal_errors'][:5])}); các dòng đó bị bỏ qua.")
    if r["verdict"] == "BLOCK":
        if r.get("behaviour_block"):
            lines.append(f"BLOCK do hành vi: chỉ kiểm lại từ {r['recheck_from']}.")
        else:
            lines.append("BLOCK: không mua theo luật desk (khuyến nghị dứt khoát; desk không chặn được lệnh ở CTCK).")
    elif r["verdict"] == "WARN":
        if r.get("acknowledged"):
            lines.append("Đã ghi nhận THESIS + xác nhận (--ack).")
        elif r.get("thesis_required"):
            lines.append("Cần: viết 1 dòng THESIS (setup, stop, target, điều kiện sai) vào portfolio.txt rồi chạy lại "
                         "với --ack; hoặc chờ phiên sau.")
        if r.get("ack_refused"):
            lines.append("--ack bị từ chối: chưa có dòng THESIS cho mã này từ phiên dữ liệu hiện tại.")
    lines.append("Tư vấn, không phải lệnh. PASS = không có cờ đỏ + có setup đã biết, không phải 'sẽ thắng'.")
    return "\n".join(lines)
