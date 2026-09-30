# Pinned public-market research inputs

These archives contain only public VCI daily OHLCV responses, their canonical
CSV equivalents and an immutable provenance manifest. They contain no account,
position, order, cookie, token, model, application cache or personal ledger.
The exact inventory is tested. Keep the archived bytes unchanged; a new vintage
requires a new archive, digest, hypothesis and results, not an in-place refresh.

| Archive | Retrieval | Requested history | Purpose |
|---|---|---|---|
| research-snapshot-v1.zip | 2026-09-30 | 2021-08-24–2026-09-29 | Existing full VN30 historical-union regression |
| research-vn100-h1-v1.zip | 2026-09-30 | 2024-11-01–2026-06-30 | Paired VN30/VN100 H1 research with common warmup |

Source: `https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart`, public
ONE_DAY requests. The manifests retain request parameters, fetch times, units,
raw response SHA256 and canonical CSV SHA256. The absolute Windows price path
inside a manifest is informational; verification resolves only relative paths
inside the extracted directory. No network fetch runs in CI.

From the repository root, CI checks `sha256sum --check
tests/fixtures/research/SHA256SUMS`, extracts each archive, then verifies every
response/CSV hash and their exact parsed agreement. Membership comes from the
separate dated research timelines, never file presence. Current-adjusted prices
and manual membership reconstruction are not an archival corporate-action or
live-execution certification. The data is a research input, not investment advice.

This replaces the previously missing `RESEARCH_SNAPSHOT_RUN_ID` prerequisite.
There is no missing-data skip or synthetic-data fallback in the full replay job.
