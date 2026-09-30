// Manual read-only browser check. Point only to a local preview with POST/DELETE disabled.
// Usage: node tests/dashboard_guard_browser.cjs <preview-url> <existing-output-directory>
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const [url, output] = process.argv.slice(2);
assert(url && /^http:\/\/127\.0\.0\.1:\d+$/.test(url), 'local preview required');
assert(output && fs.statSync(output).isDirectory(), 'existing output directory required');
(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.BROWSER_QA_EXECUTABLE || undefined});
  const page = await browser.newPage();
  const errors = [];
  const failures = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('response', r => {if (r.status() >= 400 && /\/api\/(mr|momentum|swing)\/scan/.test(r.url())) failures.push(r.status());});
  // Independent defense: preview must never restore or mutate a user's positions.
  await page.route('**/*', route => ['GET', 'HEAD'].includes(route.request().method())
    ? route.continue() : route.abort());
  try {
    await page.goto(url, {waitUntil: 'networkidle'});
    for (const id of ['mrStatus', 'momStatus', 'swingStatus']) {
      await page.locator(`#${id}`).filter({hasText: 'CHẶN'}).waitFor({state: 'attached'});
    }
    for (const id of ['mrAction', 'momBody', 'swingBody']) {
      assert.match(await page.locator(`#${id}`).textContent(), /DỮ LIỆU KHÔNG HỢP LỆ/);
    }
    for (const id of ['mrBuyBody', 'mrProbBody', 'momBody', 'swingBody']) {
      assert.equal(await page.locator(`#${id} button`).count(), 0, `actionable button in ${id}`);
    }
    for (const width of [1440, 768, 375]) {
      await page.setViewportSize({width, height: 1000});
      await page.locator('#mrAction').scrollIntoViewIfNeeded();
      await page.screenshot({path: path.join(output, `blocked-${width}.png`)});
    }
    assert.deepEqual(failures, []);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({status: 'passed', viewports: [1440, 768, 375],
                               scanner_http_failures: failures, page_errors: errors,
                               actionable_buttons: 0, mode: 'isolated_read_only_preview'}));
  } finally {
    await browser.close();
  }
})().catch(e => {console.error(e); process.exitCode = 1;});
