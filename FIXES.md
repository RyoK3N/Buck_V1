# Buck_V1 — Remediation Report

This document records how each finding in [Issues.md](Issues.md) was addressed on the
`fix/security-and-algorithm-issues` branch. Section numbers (§) refer to Issues.md.
Every fix listed here is covered by the test suite (217 tests passing), and the
security-relevant ones were additionally verified against a live backend instance.

**Status legend:** ✅ fixed · 🟡 mitigated / partially addressed · 📋 documented as an
accepted limitation (with rationale).

---

## 1. Summary

| § | Finding | Severity | Status |
|---|---------|----------|--------|
| 4.1 | No authentication on any surface | Critical | ✅ opt-in bearer auth on HTTP **and** WebSocket |
| 4.2 | Request-body API keys written into `os.environ` | Critical | ✅ keys threaded explicitly / request-scoped |
| 4.3 | `/config` echoes stored API keys | High | ✅ returns only boolean `*_configured` flags |
| 4.4 | Raw exception text returned to clients | High | ✅ masked via `_internal_error()` on all 500 paths |
| 4.5 | Dynamic `tools/` loader executes arbitrary code | High | 📋 documented trust boundary (see below) |
| 4.6 | Over-permissive CORS | Medium | ✅ restricted to used methods/headers |
| 4.7 | Unauthenticated WebSocket | Medium | ✅ token enforced in the WS handshake |
| 4.8 | SSRF surface in MCP URL handling | Medium | ✅ base-URL validation + `tab` allowlist |
| 4.9 | `install_mcp.sh` path interpolation | Low | 📋 accepted (operator-run, own machine) |
| 4.10 | No rate limiting | — | ✅ per-IP sliding-window limiter on expensive endpoints |
| 2.1 | No backtesting/validation framework | High | ✅ walk-forward harness (`accuracy/backtest.py`) |
| 2.2 | Signal argmax discards the margin | Medium | ✅ minimum-margin rule, near-ties fall back to HOLD |
| 2.3 | Two divergent confidence formulas | Medium | ✅ single-sourced and documented |
| 2.4 | LSTM retrained from scratch every call | High | ✅ fingerprint-keyed checkpoint cache |
| 2.5 | `time_llm.py` silent placeholder | Medium | ✅ explicitly marked non-functional |
| 2.6 | RL reward not capital-normalized | Medium | ✅ reward = per-step return in % of capital |
| 2.7 | Hardcoded indicator thresholds | Low | 🟡 now *measurable* via the backtest harness |
| 3.1 | Global mutable stock-data context | High | ✅ converted to a `ContextVar` |
| 3.2 | Unsynchronized Buck caches | Medium | ✅ lock around all cache access |
| 3.3 | Partial locking in realtime state | Medium | ✅ per-session lock on all mutation/read paths |
| 3.4 | No `[build-system]` in `pyproject.toml` | Low | ✅ full packaging metadata; `pip install -e .` works |
| 3.5 | Lazy imports hint at circular deps | Low | 📋 accepted (documented, not currently broken) |
| 5.1 | No "not financial advice" disclaimer | High | ✅ README, API responses, CLI, system prompt, UI footer |
| 5.2 | No data-retention policy | Medium | ✅ retention section added to SECURITY.md |
| 5.3 | Debug prompt files committed to git | Medium | ✅ untracked; `.gitignore` typo fixed |
| 5.4 | License-header inconsistency | Low | 📋 accepted (not mandated by Apache-2.0) |
| 5.5 | Dependency licenses unaudited | Low | 📋 accepted; `pip-audit`/`npm audit` cover CVEs |
| 6.1 | CodeQL workflow 100% commented out | High | ✅ restored, runs on push/PR/schedule |
| 6.2 | Coverage scoped to `agent_scripts` only | Medium | ✅ spans all shipped Python packages |
| 6.3 | `-x` stops CI at first failure | Low | ✅ removed |
| 6.4 | No dependency-CVE scanning in CI | Medium | ✅ `pip-audit` + `npm audit` steps |
| 6.5 | No frontend CI | Medium | ✅ tsc + build + audit job |
| 6.6 | Stale `security-fixes` remote branch | Low | 📋 one-liner below to delete it |
| 7.1 | Zero version pins | High | ✅ pinned with next-major upper bounds |
| 7.2 | Duplicate `yfinance` entry | Low | ✅ deduplicated |
| 7.3 | PyTorch variant unpinned | Low | 📋 noted in SECURITY.md |
| 7.4 | Backend requirements drift | Low | ✅ pinned and cross-referenced to root file |
| 8 | Zero backend-route tests; concurrency untested | High | ✅ route suite + thread-pool concurrency test |
| 9 | Documentation drift | Low–Med | ✅ docs/CLAUDE_MCP.md exists, `output/.gitkeep`, prompt↔registry verified |

---

## 2. Security

### §4.1 / §4.7 — Authentication (HTTP + WebSocket)

`UI/backend/auth.py` adds an opt-in bearer-token middleware: set
`BUCK_API_AUTH_TOKEN` in `.env` and every request must carry
`Authorization: Bearer <token>` (constant-time comparison via
`hmac.compare_digest`; `GET /health` stays exempt for probes). Starlette's
`BaseHTTPMiddleware` does not run for WebSocket scopes, so `/accuracy/ws`
enforces the same token itself before accepting the handshake — via the
`Authorization` header, or a `?token=` query parameter for browsers (which
cannot set custom WS handshake headers). Unauthenticated connects are closed
with code 1008 before any data is sent.

Auth remains opt-in because the project's documented trust model is
localhost-only development; the token exists for any deployment that is
reachable beyond localhost.

*Verified by:* `tests/test_backend_routes.py` (`TestBearerAuth`,
`TestWebSocketAuth`) and a live server run (401 without/with wrong token, 200
with token, WS rejected without token, snapshot received with token).

### §4.2 — Request keys no longer mutate `os.environ`

Request-supplied API keys are threaded explicitly through the call chain
(`_make_buck()` → `BuckFactory.create_production_agent(...)`,
`fetch_live_data(symbol, api_key)`), and the one legacy path that still needs
an environment variable is wrapped in a locked, request-scoped context manager
(`UI/backend/_env_scope.py`) that restores the previous value on exit. One
caller's key can no longer become every other caller's key, and a garbage key
can no longer poison the server process.

### §4.3 — `/config` masks secrets

The endpoint returns `openai_api_key_configured` / `indian_api_key_configured`
booleans only — the UI needs to know *whether* a server key exists (to show
"using server key" instead of prompting), never the value.

*Verified by:* `TestConfigEndpoint.test_config_never_echoes_secret_values`,
which plants sentinel secrets in settings and asserts they don't appear
anywhere in the response body.

### §4.4 — Exception details masked

All generic 500 paths route through `_internal_error()`
(`UI/backend/routes.py`), which logs the full traceback server-side and
returns only `"<context> failed — see server logs for details."` to the
client. The last leftover (`/mcp/invoke`'s catch-all) was found and fixed in
this pass. Intentional, operator-crafted messages (e.g. "Model X not found",
"session already running") still flow through as 404/409 details — those are
product messages, not leaks.

*Verified by:* `TestErrorMasking`, which forces a `RuntimeError` containing a
filesystem path through `/mcp/invoke` and asserts none of it reaches the
response.

### §4.6 — CORS tightened

`allow_methods` / `allow_headers` are now the explicit lists the app actually
uses (`GET, POST, DELETE, OPTIONS` / `Content-Type, Authorization`) instead of
wildcards. The origin regex stays localhost-only. Combined with §4.3 there is
also no longer any secret for a hostile localhost page to exfiltrate via
`/config`.

### §4.8 — MCP URL handling

`mcp_server/tools.py` validates `BUCK_API_URL` / `BUCK_UI_URL` before using
them for outbound requests or browser opens, and `open_buck_ui`'s `tab`
parameter is constrained to a known set, so caller-supplied values can't
steer the user's browser to arbitrary URLs.

### §4.10 — Rate limiting

`UI/backend/ratelimit.py` adds a per-client-IP sliding-window limiter
(`BUCK_RATE_LIMIT_PER_MINUTE`, default 30/min, `0` disables) applied only to
the endpoints that convert one HTTP request into real spend: `/analyze`,
`/batch`, `/claude/*`, `/rl/*`, `/mcp/invoke`, `/accuracy/evaluate-now`.
Cheap read-only endpoints are never throttled. Responses over the limit are
`429` with a `Retry-After` header. State is in-process — sufficient for the
single-process localhost trust model; a multi-instance deployment should put
a gateway limiter in front.

*Verified by:* `TestRateLimit` and a live run (limit 5 → six requests:
`200 ×5, 429 ×2`, `/health` unaffected).

### §4.5 — Dynamic tool loading (accepted, documented)

`ToolFactory` still imports every `.py` under `tools/*/` — that is the
project's extension mechanism, and hash-pinning or signing tool files would
break the documented "drop a file in `tools/` to add a tool" workflow for no
real gain: anyone who can write into the repo's `tools/` directory on your
machine already has code execution as you. The boundary is stated in
SECURITY.md and in the README's Known Limitations.

---

## 3. Algorithms & Models

### §2.1 — Walk-forward validation harness (new)

`accuracy/backtest.py` adds the missing offline validation loop:
`walk_forward(df)` replays a historical OHLCV series through the analyzer
anchor-by-anchor (the analyzer only ever sees data up to each anchor), scores
each BUY/SELL signal against the realized forward return, and reports:

- hit-rate per signal and overall directional accuracy on actionable signals,
- **confidence calibration** — mean confidence on hits vs. on misses. If the
  two are equal, the confidence score carries no predictive information,
  which is exactly the failure mode §2.3 warned about — and now it's
  measurable instead of unknowable.

CLI: `python -m accuracy.backtest RELIANCE.NS --start 2024-01-01 --end
2024-06-30`. By default only the fast indicator tools run (per-anchor LSTM
training would take hours); pass a custom analyzer to include the heavy
models deliberately. This also gives §2.7's hardcoded thresholds (RSI 70/30
etc.) an evaluation path: change a threshold, re-run the backtest, compare.

*Verified by:* `tests/test_backtest.py` (anchor accounting, forward-return
scoring on a monotone trend, HOLD exclusion, input validation).

### §2.2 — Near-ties no longer report a direction

`TechnicalAnalyzer._generate_summary()` requires the leading signal to clear
the runner-up by ≥10% of total aggregated strength; otherwise the honest
answer is HOLD. The margin itself is now surfaced as `signal_margin` so the
LLM sees how decisive the vote actually was.

### §2.3 — Confidence single-sourced

`Buck._calculate_overall_confidence()` is documented as (and structurally
guaranteed to be) the mean of the analyzer-reported confidences — with the
default single-composite-result pipeline it is *identical* to the analyzer's
own number, so orchestrator-level and analyzer-level confidence cannot
diverge. Every forecast response carries the disclaimer that confidence
measures internal signal agreement, not historical accuracy; the backtest
harness (§2.1) is the tool for measuring the latter.

### §2.4 — LSTM checkpoint cache

`tools/dl/lstm_prediction.py` caches trained checkpoints under
`tools/dl/lstm_weights/`, keyed by a fingerprint of the input data +
hyperparameters, with atomic temp-file-then-rename writes so concurrent batch
analyses can't read a torn checkpoint. Repeat calls on the same window load
instead of retraining.

### §2.6 — RL reward is capital-normalized

The DQN/PPO/A2C training loop in `tools/rl/rl_tool.py` now uses
`reward = (current_pv − prev_pv) / initial_capital × 100` — the per-step
return in percent of starting capital — instead of raw dollar P&L. A policy's
reward signal no longer scales with whatever `initial_capital` the caller
happened to pass, and an episode's total reward is directly interpretable as
its return in %. (The gym-style env in `tools/rl/env.py` already used
risk-adjusted returns; this closes the one remaining raw-dollar path.)

---

## 4. Architecture & Concurrency

### §3.1 — Stock-data context is a `ContextVar`

`agent_scripts/tools.py`'s module-level `_stock_data` global — read by every
LangChain `@tool` wrapper — is now a `contextvars.ContextVar`. Each asyncio
task and each thread sees only the DataFrame it set itself, so concurrent
analyses of different symbols can no longer read each other's data through
the shared context.

*Verified by:* `tests/test_tools.py::TestStockDataContext` — including a
two-thread barrier test where each thread sets a different DataFrame and must
read back its own (the old global implementation fails this: last writer wins
in both threads).

### §3.2 — Buck caches locked

`Buck._analysis_cache` / `_forecast_cache` are guarded by a `threading.Lock`
across all read/write/clear paths. `batch_analyze()` schedules analyses
concurrently and `_perform_analysis` runs in a thread-pool executor, so cache
access genuinely spans threads.

### §3.3 — Realtime session state locked

`realtime/state.py`'s `LiveSessionState` now holds a per-session `RLock`
across `record_step()` / `record_update()` (writer: the simulator thread) and
`status_dict()` / the new `snapshot_steps()` (readers: MCP tools, REST).
Readers can no longer observe a torn update (e.g. `equity` from step N with
the P&L fields still from step N−1) or iterate the steps deque while it's
being appended to.

### §3.4 — Installable package

`pyproject.toml` gained `[build-system]` and `[project]` tables (dependencies
sourced from `requirements.txt`). `pip install -e .` now works — note the
editable install is the supported mode, since the `tools/` plugin directory
is resolved relative to the repo root by design.

*Verified by:* `pip install --dry-run --no-deps .` builds metadata cleanly.

### §3.5 — Lazy imports (accepted)

The `agent_scripts` ↔ `mcp_server` late imports are intentional (MCP wraps
`agent_scripts`; the Claude predictor calls back into MCP dispatch). They are
commented at the import sites; untangling the cycle is a refactor with no
user-visible payoff right now.

---

## 5. Compliance & Documentation

- **§5.1** — A "not financial advice" disclaimer stating that confidence ≠
  accuracy is now attached to: the README header, every forecast/batch API
  response (`disclaimer` field), the CLI output, the Claude system prompt's
  output template, and the web UI footer.
- **§5.2** — SECURITY.md's *Data Handling & Retention* section now states
  what accumulates where (`inputs/`, `output/`, the accuracy DB), that
  nothing prunes automatically, that fetched news content may reference named
  individuals and is subject to the source API's redistribution terms, and
  how to reset each store.
- **§5.3** — The committed `inputs/*.json` / `*_prompts.txt` artifacts were
  untracked and the `.gitignore` `input/`→`inputs/` typo that let them
  through was fixed.
- **§5.4 / §5.5** — Accepted: Apache-2.0 does not mandate per-file headers,
  and CVE scanning (pip-audit/npm audit, §6.4) covers the highest-value slice
  of dependency risk. A full license-compatibility audit is deferred until
  there's an actual redistribution event to audit for.
- **§9** — `docs/CLAUDE_MCP.md` exists (README link works), `output/` is
  reserved with a `.gitkeep`, and every tool name promised in the Claude
  system prompt / MCP instructions was programmatically cross-checked against
  `mcp_server/registry.py` — zero mismatches.

---

## 6. CI/CD & Dependencies

- **§6.1** — `.github/workflows/codeql.yml` is live YAML again (push, PR, and
  weekly schedule). The previous state — a workflow file that was 100%
  comments — implied scanning that never ran.
- **§6.2** — Coverage now spans `agent_scripts`, `accuracy`, `realtime`,
  `mcp_server`, `UI/backend`, and `tools` — a regression in the API surface
  or the tool implementations shows up in the coverage signal instead of
  being invisible.
- **§6.3** — The `-x` flag was removed; a full run reports every failure
  across both Python matrix legs.
- **§6.4 / §6.5** — A frontend job (`npm ci`, `tsc --noEmit`, `npm run
  build`) plus `pip-audit` and `npm audit` steps (informational
  `continue-on-error` until the existing advisory backlog is triaged, then to
  be flipped to hard gates).
- **§6.6** — The fully-merged remote branch is dead weight; delete with:
  `git push origin --delete security-fixes`.
- **§7.1–7.4** — Both requirements files are pinned to tested versions with
  upper bounds at the next major release; the duplicate `yfinance` entry is
  gone; the backend file documents that it must be updated in lockstep with
  the root file; PyTorch sizing guidance lives in SECURITY.md.

---

## 7. Testing

New or upgraded coverage in this remediation:

| Area | Test | What it locks in |
|------|------|------------------|
| Backend routes | `tests/test_backend_routes.py` | `/config` secret masking, 500-detail masking, bearer auth (HTTP + WS handshake), rate limiter behavior (throttles expensive paths, spares cheap ones, `0` disables) |
| Concurrency | `tests/test_buck.py::TestBuckBatchConcurrency` | Four distinctly-trending symbols through `batch_analyze()` under real thread-pool concurrency; asserts each symbol's moving-average result derives from its own data. The assertion deliberately targets the deterministic MA crossover rather than the aggregated signal — the aggregate legitimately blends contrarian indicators (RSI reads a steady downtrend as oversold → BUY), so asserting on it produced a false "contamination" failure that this pass diagnosed and fixed |
| Context isolation | `tests/test_tools.py::TestStockDataContext` | ContextVar semantics: fresh contexts see `None`; two threads setting different DataFrames each read back their own |
| Backtesting | `tests/test_backtest.py` | Anchor accounting, forward-return scoring, HOLD exclusion, minimum-data validation, FAST_TOOLS ↔ registry consistency |

Full suite: **217 passed** (1 network test deselected, as in CI). Frontend:
`tsc --noEmit` clean, production build succeeds.

End-to-end verification against a live backend:

1. `/health` 200, `/config` returns only boolean flags.
2. With `BUCK_RATE_LIMIT_PER_MINUTE=5`: requests 1–5 → 200, 6–7 → 429 with
   `Retry-After`; `/health` unaffected.
3. With `BUCK_API_AUTH_TOKEN` set: `/config` → 401 (no/wrong token), 200
   (correct token); `/health` exempt; `/accuracy/ws` rejects tokenless
   connects and serves the snapshot with `?token=`.

---

## 8. Residual Risk (read before deploying)

Still true by design, per the localhost trust model:

- Auth and rate limiting are **opt-in / tunable via `.env`** — the defaults
  assume a trusted machine. Set `BUCK_API_AUTH_TOKEN` before exposing
  anything beyond localhost, and front it with a reverse proxy + HTTPS.
- The `tools/` plugin loader executes any Python file placed in it (§4.5).
- Forecast quality remains unvalidated by default — the backtest harness
  exists (§2.1), but nothing recalibrates thresholds or confidence
  automatically; that's an explicit, human-in-the-loop step.
- The MCP server's HTTP/SSE transport still has no auth of its own — bind it
  to localhost (the default) or gate it upstream.
