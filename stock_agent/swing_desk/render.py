"""Vietnamese plain-text rendering for chat/terminal."""
from __future__ import annotations

import re

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


def history_line(h: dict | None) -> list[str]:
    """Which series the long-window fields used; nothing with --history off (old output)."""
    if not h or h.get("history_mode") != "on":
        return []
    if h.get("history_source") == "history_store":
        src = (f"history_store {h.get('history_store_id')} (sha {(h.get('history_store_manifest_sha256') or '')[:12]}), "
               f"từ {h.get('history_first_date')}, {h.get('history_rows')} phiên")
    else:
        src = f"snapshot_650d ({h.get('history_rows')} phiên từ {h.get('history_first_date')})"
    lines = [f"Lịch sử dài (SMA200, đỉnh 250, MAE/MFE, đợt giảm): {src}"]
    lines += [f"  {n['level']}: {n['text']}" for n in h.get("history_notes") or [] if n.get("level") == "WARN"]
    if h.get("history_data_issues"):
        lines.append(f"  {len(h['history_data_issues'])} dòng store bị cách ly (không dùng): "
                     + ", ".join(str(i.get("date")) for i in h["history_data_issues"][:5]))
    return lines


def _label(x: dict) -> str:
    """Rule label; G1 names the active universe stated in its reason (the VN30 label is unchanged)."""
    if x["id"] == "G1":
        m = re.search(r"not in current (\S+) universe", x.get("reason") or "")
        if m:
            return f"thuộc {m.group(1)}"
    return RULE_LABEL.get(x["id"], "")


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
            lines.append(f"  - {x['id']} {_label(x)} [{tag}]: {x['reason']}")
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
    lines += history_line(r.get("history"))
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


BF_LABEL = {
    "BF-G0": "dữ liệu verified của mã", "BF-G1": "phạm vi lệnh BF mới", "BF-G2": "loại cú giảm vs hồ sơ",
    "BF-G3": "không mua ở sàn", "BF-G4": "ổn định sau sàn", "BF-G5": "sự kiện / pha loãng", "BF-G6": "tin FACT tiêu cực",
    "BF-G7": "thanh khoản", "BF-G8": "margin / cảnh báo", "BF-G9": "bối cảnh", "BF-G10": "ngân sách rủi ro BF",
    "BF-G11": "số lần thử", "BF-G12": "THESIS viết trước",
    "BF-B1": "bắt dao rơi ngày sàn", "BF-B2": "neo giá đỉnh cũ", "BF-B3": "'rẻ rồi'", "BF-B4": "bình quân giá xuống",
    "BF-B5": "gỡ lỗ cùng mã", "BF-B6": "lỗ giấy -> liều hơn", "BF-B7": "tăng size sau lệnh thắng", "BF-B8": "đám đông",
}
FALL_LABEL = {"a": "(a) hoảng loạn thị trường", "b": "(b) cú sốc cơ bản riêng", "c": "(c) xả kỹ thuật không tin",
              "UNCLASSIFIED": "UNCLASSIFIED (xử lý như b)", "NOT_A_BOTTOMFISH": "không phải cú giảm bắt đáy (dùng mode swing)"}


def _tag(x: dict) -> str:
    if x["status"] == "UNVERIFIED":
        return "UNVERIFIED"
    return "BLOCK" if x["severity"] == "hard" else "WARN"


def render_bf_check(r: dict) -> str:
    sha = (r.get("snapshot_manifest_sha256") or "")[:12]
    src = (r.get("snapshot") or {}).get("source", "vn30")
    plan = r.get("plan") or {}
    lines = [f"{r['ticker']} — {r['verdict']}  (mode bắt đáy, hồ sơ {r['profile']} {r.get('profile_label') or ''}; "
             f"dữ liệu EOD tới {r.get('data_as_of') or 'n/a'}; snapshot {src} sha {sha or 'n/a'})",
             f"Phiên giao dịch: {r['trade_session']}. {r.get('eod_note', '')}", r["default_note"]]
    if r.get("edge_note"):
        lines.append(f"Hồ sơ {r['profile']}: {r['edge_note']}.")
    fall = r.get("fall") or {}
    lines.append(f"Loại cú giảm: {FALL_LABEL.get(fall.get('fall_type'), fall.get('fall_type'))} — {fall.get('reason', '')}")
    for ev in fall.get("evidence", [])[:4]:
        lines.append(f"  · {ev}")
    gate = [x for x in r["rules"] if x["kind"] == "gate" and x["status"] in ("FAIL", "UNVERIFIED")]
    beh = [x for x in r["rules"] if x["kind"] == "behaviour" and x["status"] in ("FAIL", "UNVERIFIED")]
    lines.append("Luật không đạt:" if gate else "Luật không đạt: không có")
    for x in gate:
        lines.append(f"  - {x['id']} {BF_LABEL.get(x['id'], RULE_LABEL.get(x['id'], ''))} [{_tag(x)}]: {x['reason']}")
    lines.append("Cờ hành vi:" if beh else "Cờ hành vi: không có")
    for x in beh:
        lines.append(f"  - {x['id']} {BF_LABEL.get(x['id'], RULE_LABEL.get(x['id'], ''))} [{_tag(x)}]: {x['reason']}")
    lines.append(f"Nếu vẫn mua: bán được sớm nhất chiều {r['earliest_sell']} (T+2); không thoát được trước thời điểm này.")
    if plan:
        lines.append(f"Kế hoạch {plan['profile']}: giá tham chiếu {vnd(plan['entry_ref'])}; rủi ro {pct(plan['risk_pct_nav'])} NAV"
                     + (f" x {plan['risk_multiplier']:g}" if plan.get("multipliers") else "")
                     + f" = {vnd(plan['risk_vnd'])} VND ở lỗ hiệu dụng {pct(plan['effective_loss_pct'])} "
                       f"(stop + 1 phiên sàn + phí)")
        size = f"{vnd(plan['qty'])} cp" + (f" chia {plan['max_tranches']} tầng ≈ {vnd(plan['tranche_qty'])} cp/tầng"
                                            if plan.get("tranche_qty") else "")
        lines.append(f"  Size gợi ý: {size}; lệnh này {vnd(plan['order_qty'])} cp ≈ {vnd(plan['position_value'])} VND"
                     + (f" (giới hạn bởi {plan['capped_by']})" if plan.get("capped_by") else "")
                     + f"; VND rủi ro hiệu dụng {vnd(plan['vnd_at_risk_effective'])}")
        inv = f"đóng cửa < {vnd(plan['invalidation'])}" if plan.get("invalidation") else plan.get("invalidation_rule")
        cat = vnd(plan["catastrophe_stop"]) if plan.get("catastrophe_stop") else "không (size là stop)"
        lines.append(f"  Vô hiệu: {inv}; thảm họa: {cat}; time-stop: {plan['time_stop_session']} "
                     f"({plan['time_stop_sessions']} phiên); LO tối đa {vnd(plan['entry_limit'])}")
        sc = plan["scenarios"]
        lines.append(f"  Kịch bản thêm 3 phiên sàn khi đang khóa: -{pct(sc['three_more_floors_pct'])} vị thế "
                     f"≈ {vnd(sc['three_more_floors_vnd'])} VND (gồm phí); ở lỗ hiệu dụng: {vnd(sc['effective_loss_vnd'])} VND")
        req = plan.get("requested")
        if req:
            lines.append(f"  Số tiền bạn nhập {vnd(req['amount'])} VND → {vnd(req['qty'])} cp: rủi ro hiệu dụng "
                         f"{vnd(req['risk_effective_vnd'])} VND ({pct(req['risk_effective_pct_nav'], 2)} NAV)"
                         + (" > ngân sách" if req["over_budget"] else ""))
    else:
        lines.append("Không có kế hoạch size (hồ sơ P0 hoặc thiếu dữ liệu).")
    held = r.get("held")
    if held:
        lines.append(f"Vị thế đang giữ: {vnd(held['qty'])} cp, giá vốn {vnd(held['avg_cost'])}, bán được từ chiều "
                     f"{held['earliest_sell']}; THESIS invalid={held.get('invalid') or 'chưa ghi'}, "
                     f"time_stop={held.get('time_stop') or 'chưa ghi'}. {held['note']}")
    lines.append(f"NAV dùng để tính: {r.get('nav_source')}")
    lines += history_line(r.get("history"))
    if r.get("recheck_conditions"):
        lines.append("Kiểm lại sớm nhất khi: " + "; ".join(r["recheck_conditions"]) + ".")
    if r.get("ack_refused"):
        lines.append("--ack bị từ chối: chỉ WARN có THESIS hợp lệ mới được xác nhận; --ack không bỏ qua BF-G12.")
    if r.get("journal_errors"):
        lines.append(f"portfolio.txt có {len(r['journal_errors'])} dòng lỗi (dòng "
                     f"{', '.join(str(e['line']) for e in r['journal_errors'][:5])}); các dòng đó bị bỏ qua.")
    lines.append(f"Checklist ({r['checklist_path']}): 10 câu — trả lời bằng chữ trong dòng THESIS trước lệnh.")
    lines.append(r["caveat"])
    return "\n".join(lines)
