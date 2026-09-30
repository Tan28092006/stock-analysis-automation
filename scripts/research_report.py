"""Render immutable research results into a local, self-contained audit report."""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import pandas as pd


def table(rows):
    return pd.DataFrame(rows).to_html(index=False, border=0, escape=True, float_format=lambda n: f'{n:,.3f}')


def symbol_pnl(trial):
    rows = {}
    replay = trial['replay']
    for f in replay['fills']:
        item = rows.setdefault(f['symbol'], dict(symbol=f['symbol'], buys=0., sales=0., end_value=0.))
        item['buys' if f['side'] == 'BUY' else 'sales'] += f['qty'] * f['price'] + (f['fee'] if f['side'] == 'BUY' else -f['fee'])
    for symbol, position in trial['portfolio_history'][-1]['positions'].items():
        rows[symbol]['end_value'] = position['value']
    for item in rows.values():
        item['pnl_vnd'] = item['sales'] + item['end_value'] - item['buys']
    if abs(sum(r['pnl_vnd'] for r in rows.values()) - trial['summary']['pnl']) > 1e-5:
        raise ValueError('Symbol contribution does not reconcile')
    return sorted(rows.values(), key=lambda r: -r['pnl_vnd'])


def render(results, foreign, output):
    if output.exists():
        raise ValueError('Output must be new')
    output.mkdir(parents=True)
    data = json.loads(results.read_text(encoding='utf-8'))
    flows = json.loads(foreign.read_text(encoding='utf-8'))
    recent = data['blocks']['recent_6m']
    core = [('MR stop/target', recent['mr']['bracket']), ('MR fixed 15', recent['mr']['fixed15']), ('Momentum', recent['momentum']['baseline'])]
    sections = ['<h1>Kiểm định bắt đáy, momentum & khối ngoại</h1>',
        '<p>Snapshot EOD 29/09/2026 · 1 tỷ VND riêng cho mỗi chiến lược · VN30 lịch sử · không margin · không ML.</p>',
        '<p class="warning"><b>CHƯA ĐẠT điều kiện xác nhận lợi thế tiền thật.</b> Đây là dữ liệu phát triển đã xem, không phải holdout. Không biến thể nào vượt ngưỡng kiểm định gia đình đã đăng ký. Đừng chọn biến thể chỉ vì lãi cao nhất.</p>',
        '<h2>Sáu tháng: 01/04–29/09/2026</h2>', table([dict(strategy=name, **{k:t['summary'][k] for k in ('pnl','return_pct','max_drawdown_pct','average_exposure_pct','buys','sells','closed_lots')}) for name,t in core]),
        f'<p>VNINDEX giá, mua tại giá mở cửa phiên đầu: {recent["benchmark"]["return_pct"]:.3f}%. Chỉ số chưa trừ chi phí và không phải một sản phẩm đầu tư.</p>',
        '<p>Fixed 15: mua mở cửa phiên sau tín hiệu, bán đóng cửa tại entry + 15 phiên; vô hiệu stop/target sau mua nhưng vẫn dùng stop để định cỡ lệnh. Không cộng lợi nhuận % từng mã để thành lợi nhuận danh mục.</p>']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, trial in core:
        nav = pd.DataFrame(trial['replay']['nav'])
        ax.plot(pd.to_datetime(nav['date']), (nav['nav'] / 1e9 - 1) * 100, label=name)
    # Benchmark path is reconstructed only from the certified manifest, not fresh data.
    manifest_path = Path(data['manifest'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    idx = pd.read_csv(manifest_path.parent / manifest['files']['VNINDEX']['path'])
    idx = idx.loc[(idx['date'] >= recent['start']) & (idx['date'] <= recent['end'])]
    ax.plot(pd.to_datetime(idx['date']), (idx['close'] / idx.iloc[0]['open'] - 1) * 100, label='VNINDEX gross', linestyle='--')
    ax.axhline(0, color='#888', linewidth=.6); ax.set_ylabel('Portfolio return (%)')
    ax.legend(ncol=4); ax.grid(alpha=.15); fig.tight_layout()
    fig.savefig(output / 'six_month_nav.png', dpi=160)
    image = io.BytesIO(); fig.savefig(image, format='png', dpi=160); plt.close(fig)
    sections.append('<img alt="Six month NAV" src="data:image/png;base64,' + base64.b64encode(image.getvalue()).decode() + '">')
    for name, trial in core:
        key = {'MR stop/target':'mr_bracket', 'MR fixed 15':'mr_fixed15', 'Momentum':'momentum'}[name]
        sections += [f'<h3>{html.escape(name)} — lịch sử danh mục</h3>',
                     table([dict(month=m, return_pct=v) for m,v in trial['summary']['monthly_pct'].items()]),
                     '<details open><summary>Lãi/lỗ từng mã (VND, đã gồm chi phí khớp mô phỏng)</summary>', table(symbol_pnl(trial)), '</details>',
                     '<details><summary>Tất cả lệnh mua/bán</summary>', table(trial['replay']['fills']), '</details>',
                     '<details><summary>Các lot đã đóng</summary>', table(trial['replay']['closed_lots']), '</details>']
        history = [{**{k:v for k,v in r.items() if k != 'positions'},
                    'holdings': ', '.join(f'{s}: {p["qty"]:,}' for s,p in r['positions'].items()) or 'Tiền mặt'} for r in trial['portfolio_history']]
        sections += ['<details><summary>NAV, tiền mặt, khoản chờ về và số lượng từng mã mỗi phiên</summary>', table(history), '</details>']
        pd.DataFrame(history).to_csv(output / f'{key}_daily_portfolio.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(trial['replay']['fills']).to_csv(output / f'{key}_fills.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(trial['replay']['closed_lots']).to_csv(output / f'{key}_closed_lots.csv', index=False, encoding='utf-8-sig')
    sections.append('<h2>Các khung thị trường — lợi nhuận % sau chi phí mô phỏng</h2>')
    sections.append(table([dict(period=name, start=b['start'], end=b['end'], role=b['role'],
        MR=b['mr']['bracket']['summary']['return_pct'], MR15=b['mr']['fixed15']['summary']['return_pct'],
        momentum=b['momentum']['baseline']['summary']['return_pct'], VNINDEX=b['benchmark']['return_pct'],
        universe_equal=b['universe_equal']['summary']['return_pct']) for name,b in data['blocks'].items()]))
    sections.append('<p>Mỗi block khởi đầu tiền mặt. Continuous giữ liên tục qua các năm; không bằng cộng hoặc nối kết quả các block. Các cửa sổ sự kiện chồng lấn chỉ dùng chẩn đoán.</p>')
    for kind in ('momentum','mr'):
        continuous = data['blocks']['continuous'][kind]
        sections += [f'<h2>Ablation {kind}</h2>', table([dict(variant=v,
            recent_return=recent[kind][v]['summary']['return_pct'], continuous_return=t['summary']['return_pct'],
            max_drawdown=t['summary']['max_drawdown_pct'], closed_lots=t['summary']['closed_lots'],
            ci_family_20=str([round(x,3) for x in t['paired_tests'][0]['ci_family']]),
            ci_family_40=str([round(x,3) for x in t['paired_tests'][1]['ci_family']]),
            status=t['statistical_status']) for v,t in continuous.items()])]
    sections.append('<p>CI là chênh lệch trung bình lợi nhuận ngày quy năm (điểm %), không phải CI của CAGR. Baseline so VNINDEX; biến thể so baseline cùng chiến lược. Bootstrap cặp theo block 20/40 phiên, Bonferroni 22 so sánh chính, 4.000 mẫu. MR cần ≥30 lot đóng; lot và mã không được giả định độc lập. Mọi kết luận vẫn chỉ thuộc tập phát triển.</p>')
    sections += ['<h2>Stress chi phí và chậm một phiên</h2>', table([dict(period=name, scenario=scenario, strategy=s,
        return_pct=t['summary']['return_pct'], max_drawdown=t['summary']['max_drawdown_pct'])
        for name,b in data['blocks'].items() for scenario, trials in b['stresses'].items() for s,t in trials.items()])]
    sections += ['<h2>Khối ngoại: cách ly dữ liệu cũ</h2>',
        '<p>2.500 dòng chi tiết lệch một ngày theo timestamp gốc; loader cũ ghép triệu/tỷ VND gây lệch 1.000 lần. Không dòng nào có thời điểm thu thập để chứng minh đã sẵn có trước lệnh. Không sửa đè raw, không nạp tín hiệu vào production.</p>',
        table([dict(source=s, **{k:r.get(k) for k in ('rows','symbols','periods','first','last','duplicate_rows','off_calendar_stored_rows','corrected_epoch_rows','prospective_eligible_rows')}) for s,r in flows['files'].items()]),
        '<p>Thử nghiệm bên dưới chỉ là độ nhạy với giả định ngày biểu đồ +1 ngày lịch; không phải ngày đã được xác minh. Tín hiệu = (KL mua − KL bán)/(KL mua + KL bán), không phải đơn thuần số tỷ mua ròng; trung bình 1/5 phiên đủ dữ liệu, lệnh mở cửa sau 1/2 phiên, giữ 5/15/21 phiên. IC đã điều kiện hóa bỏ thành phần tương quan với biến động giá 5 phiên. Cần ít nhất 10 mã VN30/phiên. Dữ liệu chi tiết không đạt ngưỡng này.</p>',
        table([dict(source=source, test=k, dates=t['dates'], IC=t['mean_ic'], partial_IC=t['partial_price5_mean_ic'])
            for source,e in flows['experiments'].items() for k,t in e.get('tests',{}).items()]),
        '<h2>Giới hạn và các lỗi logic cần xử lý trước tiền thật</h2>',
        '<ul><li>Momentum dashboard cảnh báo thoát hằng ngày; replay chỉ tái cân bằng tháng. MR dashboard và replay khác khóa thanh toán. Chưa có parity end-to-end.</li><li>Volatility VNINDEX không bằng rủi ro danh mục; tương quan ngành/tập đoàn và đóng góp lợi nhuận tập trung phải theo dõi. Inverse-vol không tự động là risk parity.</li><li>Giá điều chỉnh hiện tại, cổ tức/corporate actions và giá tham chiếu chính thức chưa được đối soát theo lịch sử. Sàn/khối lượng dư bán có thể làm stop không khớp; OHLC không mô tả hàng chờ.</li><li>15 phiên không đảm bảo hồi phục: tin xấu cơ bản có thể là giảm giá lâu dài. Giữ mã thua vì muốn hòa vốn là disposition effect; chọn chart đẹp là hindsight/selection bias.</li><li>Khối ngoại có thể chạy theo giá, cơ cấu ETF, thỏa thuận hoặc hedging; mua ròng không đồng nghĩa thông tin tốt. Các nguồn khác định nghĩa không được ghép tùy tiện.</li><li>Chưa có news/earnings timestamps, sector PIT, breadth ngoài VN30, corporate actions certified, spread/depth. Chúng là giả thuyết kế tiếp cần dữ liệu trước khi ablation; không điền giá trị tương lai.</li><li>MR ít giao dịch: tỷ lệ thắng 75% từ 4 lệnh không chứng minh edge. Tiền mặt không hưởng lãi trong mô phỏng; benchmark là price index.</li></ul>',
        '<h2>Tái lập & trạng thái kỹ thuật</h2>',
        f'<p>Run: {html.escape(str(results.resolve()))}<br>Foreign: {html.escape(str(foreign.resolve()))}<br>Commit lúc bắt đầu: {html.escape(data["git_commit"])}</p>',
        '<p>Toàn bộ trial, cash/receivables, fills, open lots, source hashes và kiểm tra NAV độc lập được giữ trong JSON. Báo cáo không thay đổi scan, model hay ledger thật.</p>']
    document = '<!doctype html><html lang="vi"><meta charset="utf-8"><title>Research gate 2026-09-30</title><style>body{font:15px/1.55 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#172b3b;background:#f8fafc}h1,h2,h3{color:#123f60}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{padding:8px;border-bottom:1px solid #dce3eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e5edf5}details{overflow:auto;background:white;padding:10px;margin:10px 0}summary{cursor:pointer;font-weight:600}.warning{padding:16px;background:#fff1d6;border-left:5px solid #c67e14}img{max-width:100%}li{margin:8px 0}</style><body>' + '\n'.join(sections) + '</body></html>'
    (output / 'report.html').write_text(document, encoding='utf-8')
    print(output / 'report.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--foreign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    render(args.results, args.foreign, args.output)
