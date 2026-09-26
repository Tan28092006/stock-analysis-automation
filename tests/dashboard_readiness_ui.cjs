// Execute the real render functions, with an isolated DOM surface and no network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('web/index.html', 'utf8');
function source(name) {
  const start = html.indexOf(`    function ${name}(`);
  assert(start >= 0, `missing ${name}`);
  const remainder = html.slice(start + 5);
  const next = remainder.search(/^    (?:async )?function /m);
  assert(next >= 0);
  return html.slice(start, start + 5 + next);
}
const nodes = new Map();
const document = {getElementById(id) {
  if (!nodes.has(id)) nodes.set(id, {innerHTML: '', textContent: '', style: {}, setAttribute() {}});
  return nodes.get(id);
}};
const context = {document, mrFmt: String, dataAgeBadge: () => '', Intl};
vm.createContext(context);
for (const name of ['renderActionToday', 'renderMom', 'renderSwing']) {
  vm.runInContext(source(name), context);
}
const blocked = {status: 'blocked_data', active: false, market: {state: 'UNKNOWN'},
                 picks: [], buys: [], prob_buys: [], positions: [], sell_alerts: []};
context.renderActionToday(blocked);
assert.match(nodes.get('mrAction').innerHTML + nodes.get('mrAction').textContent, /DỮ LIỆU KHÔNG HỢP LỆ/);
assert.doesNotMatch(nodes.get('mrAction').innerHTML, /Chưa mã nào đạt/);
context.renderMom(blocked);
assert.match(nodes.get('momBody').innerHTML, /DỮ LIỆU KHÔNG HỢP LỆ/);
context.renderSwing(blocked);
assert.match(nodes.get('swingBody').innerHTML, /DỮ LIỆU KHÔNG HỢP LỆ/);
console.log('PASS: 3 real UI render functions distinguish blocked data from no market signal');
