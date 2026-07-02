# Buck_V1 — Issues, Limitations & Risk Assessment

This document is a critical analysis of the current state of the repository: algorithmic
soundness, architecture, security, compliance, CI/CD, testing, and documentation. It is
written for internal engineering use — not a marketing document. Severity is rated
**Critical / High / Medium / Low**. File:line references point at the current `main`
branch (commit `26e4ace`) unless noted.

> **Verification note:** every finding below was confirmed by direct code inspection.
> Several plausible-sounding claims generated during analysis (a SQL-injection theory in
> `accuracy/repository.py`, a "vanilla DQN" theory in `dqn_agent.py`, a fee
> double-counting theory in `wallet.py`, and a Sharpe-ratio formula theory) were checked
> against the actual code and turned out to be **incorrect** — those mechanisms are
> implemented correctly. They are intentionally omitted so this document only contains
> real issues.

---

## 1. Executive Summary

Buck is a single-process, localhost-oriented research prototype: a FastAPI backend, a
React UI, an MCP server for Claude Desktop, and an agent pipeline that blends technical
indicators, an LSTM, an RL stack, and an LLM into a next-day OHLC forecast. It is **not**
production-hardened:

- No authentication anywhere (REST, WebSocket, or MCP HTTP/SSE transports).
- API keys are accepted from HTTP request bodies, written into the live process's
  `os.environ`, and one endpoint (`/config`) echoes stored keys back to any caller.
- The prediction pipeline has no backtesting, walk-forward validation, or held-out
  accuracy calibration anywhere in the codebase — confidence scores are structural
  (tool-agreement ratios), not validated against outcomes.
- The only CI security scanner in the repo (CodeQL) is entirely commented out — it does
  not run, despite the workflow file existing and looking active in a file listing.
- There is no disclaimer anywhere in the product (README, UI, CLI output, or LLM system
  prompt) stating that forecasts are not financial advice — a real compliance exposure
  for a tool that emits price targets with a "confidence score."

---

## 2. Algorithms, Models & Forecasting Methodology

### 2.1 No backtesting / validation framework (High)
There is no walk-forward backtest, no held-out test harness, and no calibration step
anywhere in `agent_scripts/`, `tools/`, or `accuracy/`. `accuracy/` only records
*live* predictions vs. later-observed actuals going forward (`accuracy/repository.py`,
`accuracy/evaluator.py`) — it cannot validate the system before shipping a change, and
nothing in the repo consumes its output to recalibrate signal thresholds, tool weights,
or confidence formulas. Every magic number below (RSI 70/30, MACD crossover, LSTM
0.6/0.4 thresholds, confidence blend weights) is asserted, not fitted or tested.

### 2.2 Signal aggregation conflates confidence and direction (Medium)
`agent_scripts/analyzers.py:120` picks the overall signal via
`max(signals, key=signals.get)` over aggregated BUY/SELL/HOLD strength sums. A BUY
"winning" by a hair (e.g. 0.11 vs. HOLD's 0.10) is reported identically to a BUY that
wins by a landslide — the margin is discarded before the LLM ever sees it, and there is
no minimum-margin threshold to fall back to HOLD on a near-tie.

### 2.3 Confidence score is structural, not predictive (Medium)
`agent_scripts/analyzers.py:132-163` computes
`confidence = base_confidence*0.6 + consistency*0.4`, where `base_confidence` is just
"fraction of tools that ran without error" and `consistency` is agreement among tools.
This is a measure of *internal agreement*, not of *historical accuracy* — a forecast can
report 90% confidence purely because all six indicator tools executed successfully and
happened to agree, independent of whether that combination has ever been right. Buck's
own `_calculate_overall_confidence()` in `agent_scripts/buck.py` recomputes confidence a
second, different way (plain average), so the analyzer-level and orchestrator-level
confidence numbers can diverge for the same run with no reconciliation.

### 2.4 LSTM is retrained from scratch on every single call (High)
`tools/dl/lstm_prediction.py` defines `_LSTMNet` and trains it inline inside
`execute()` for every analysis request — there is no `torch.save`/`torch.load` anywhere
in the file, confirmed by direct search. Consequences:
- Every prediction is computationally wasteful (full training run per HTTP request).
- Predictions are non-deterministic and not reproducible run-to-run (fresh random init,
  small dataset, no fixed seed verified).
- The 80/20 chronological split is recomputed and retrained per call instead of being
  validated once and reused — there's no persisted, versioned model to audit or roll
  back.

### 2.5 `tools/dl/time_llm.py` is a non-functional placeholder (Medium)
The file contains only comments describing a planned architecture ("Data Engineering
Layer", "Compression Layer", "A2A policy layer", etc.) with zero executable code, no
`TOOL_CLASS`/`TOOL_FUNC` exports. It is not wired into `ToolFactory` and cannot be
invoked, but its presence in `tools/dl/` alongside a real, working `lstm_prediction.py`
is misleading to anyone scanning the directory expecting parity with the README's
"tools/dl/" description.

### 2.6 RL stack: reward shaping is not capital-normalized (Medium)
`tools/rl/rl_tool.py` trains on raw dollar P&L rather than returns normalized by
account size, so the learned policy's reward signal scales with the size of
`initial_capital` passed in by the caller — a model trained against one capital size
isn't guaranteed to behave sensibly at another, and nothing in the codebase tests for
this scale-invariance failure mode.

### 2.7 Hardcoded, unvalidated thresholds throughout the indicator suite (Low)
RSI overbought/oversold (70/30), MACD crossover, OBV fixed-strength-0.7 regardless of
trend magnitude, LSTM 0.6/0.4 direction thresholds, and sentiment quality thresholds
(`news_count >= 15`, `relevance >= 0.3` in `agent_scripts/analyzers.py:503-573`) are all
textbook defaults pulled in without any evidence they're appropriate for the specific
symbols/intervals Buck targets (Indian equities, intraday data). This is a design
limitation rather than a bug, but it compounds with §2.1 — there's no mechanism to ever
discover these defaults are wrong.

---

## 3. Architecture & Concurrency

### 3.1 Global mutable stock-data context is not safe under concurrent batch use (High)
`agent_scripts/tools.py` exposes a **module-level** `_stock_data` singleton via
`set_stock_data()`/`get_stock_data()`, used by every `@tool`-decorated LangChain function
(the pattern every file in `tools/maths/`, `tools/dl/` etc. follows). `Buck.batch_analyze()`
(`agent_scripts/buck.py`) runs multiple symbols concurrently behind an `asyncio.Semaphore`.
`TechnicalAnalyzer` itself calls `tool.execute(df)` directly (bypassing the global), but
**any path that goes through the LangChain `@tool` wrapper** (the MCP server, and any
agent loop that lets an LLM call tools directly) reads `get_stock_data()` — a single
shared global. Concurrent requests for different symbols can read each other's data,
silently corrupting which symbol a tool's analysis applies to.

### 3.2 In-memory caches in `Buck` are unsynchronized under concurrency (Medium)
`agent_scripts/buck.py` `_analysis_cache` / `_forecast_cache` are plain dicts written
from concurrently-scheduled `asyncio` tasks in `batch_analyze()` with no lock. Plain
dict writes are not guaranteed atomic across `await` points in all code paths that touch
them, and there's no test exercising this.

### 3.3 Realtime session state has partial locking (Medium)
`realtime/state.py` only takes its `Lock` inside `register_session()` and `_resolve()`.
`record_step()`/`record_update()` on `LiveSessionState` mutate session history without
holding the lock, while the simulator thread (`realtime/sim.py`) and MCP read tools can
touch the same session concurrently.

### 3.4 No `[build-system]` in `pyproject.toml` (Low)
`pyproject.toml` contains only `[tool.pytest.ini_options]` — no `[build-system]`,
no `[project]` table, no version/metadata. The repo is structured like an installable
package (`agent_scripts`, `mcp_server`, `tools` all have `__init__.py`) but
`pip install -e .` / `pip install .` will not work; it can only be used via
`requirements.txt` + running scripts from the repo root. This silently breaks anyone
trying to depend on Buck as a library or package it (e.g. for the `claude_extension`'s
"bring your own venv" instructions to be more turnkey).

### 3.5 Lazy/late imports hint at circular dependencies (Low)
`agent_scripts/claude_predictor.py` imports `mcp_server.tools.dispatch_async` inside a
method body rather than at module scope — a common workaround for `agent_scripts` ↔
`mcp_server` forming a dependency cycle (MCP wraps `agent_scripts`, but the Claude
predictor inside `agent_scripts` calls back into MCP tools). Not currently broken, but
fragile to refactor.

---

## 4. Security

### 4.1 No authentication on any surface (Critical)
The FastAPI backend (`UI/backend/`), its WebSocket endpoint, and the MCP server's
HTTP/SSE transport (`mcp_server/runner.py`) have no auth layer. `SECURITY.md` itself
acknowledges this ("There is no built-in authentication on the API endpoints") but the
mitigation — "don't expose it" — is not enforced anywhere in code; nothing stops
`--host 0.0.0.0` or a reverse-proxy misconfiguration from putting an unauthenticated
trading-forecast API and an unauthenticated MCP server on the open network.

### 4.2 API keys accepted via request body and written into live process environment (Critical)
`UI/backend/routes.py` — at minimum lines 143-145, 164-166, 413-414, 512-513, 588-589,
711-714, 923, 1023 — every one of these does `os.environ["OPENAI_API_KEY"] = req.openai_api_key`
(or `INDIAN_API_KEY` / `ANTHROPIC_API_KEY`) directly from an inbound HTTP request. This:
- Mutates global interpreter state from untrusted network input — one caller's submitted
  key becomes the key used by *every other concurrent request* until overwritten again.
- Means any client that can reach the backend can overwrite the server's live credentials
  (a trivial DoS: submit a garbage key, every subsequent LLM call fails) or potentially
  redirect API usage/billing to a key of their choosing.
- Persists secrets in process memory/environment beyond the lifetime of the single
  request that supplied them.

### 4.3 `GET /config` echoes stored API keys back to any caller (High)
`UI/backend/routes.py:92-104` — the `/config` endpoint returns `SETTINGS.openai_api_key`
and `SETTINGS.indian_api_key` verbatim (only filtering a literal `"__placeholder__"`
sentinel). Any unauthenticated client that can reach the backend can retrieve the
server's configured secrets simply by calling this endpoint.

### 4.4 Raw exception text returned to clients (High)
Multiple handlers (e.g. `UI/backend/routes.py:159, 179, 212, 338, 407, 576, 603, 750, 890`)
return `str(exc)` in the HTTP response body. This leaks stack traces, internal file
paths, and library internals to any caller — useful reconnaissance for an attacker and a
direct violation of "don't leak implementation details in error responses."

### 4.5 Dynamic tool loading executes arbitrary Python with no integrity check (High)
`agent_scripts/tools.py:95-99` uses `importlib.util.spec_from_file_location` +
`spec.loader.exec_module(mod)` to import every `.py` file under `tools/*/` at startup.
There is no signature/hash verification and no allowlist — `SECURITY.md` correctly notes
"Only place trusted code in this directory," but nothing in code enforces that boundary.
Anyone who can write a file into `tools/` (a compromised dependency, a malicious PR, a
writable deployment volume) gets arbitrary code execution in the backend process on next
restart.

### 4.6 CORS allows all methods/headers for credentialed cross-origin requests (Medium)
`UI/backend/main.py:76-82` — `CORSMiddleware(allow_origin_regex=r"^http://localhost:\d+$",
allow_credentials=True, allow_methods=["*"], allow_headers=["*"])`. The origin regex
limits this to localhost, but it matches **any** local port, and combined with
`allow_credentials=True` + wildcard methods/headers, any process running locally
(e.g. malicious local software, a compromised browser extension, another locally-served
web page) can make full credentialed requests against the API, including the
secret-echoing `/config` endpoint in §4.3.

### 4.7 Unauthenticated WebSocket broadcasts live accuracy/session data (Medium)
`UI/backend/routes.py` — the `/accuracy/ws` (and realtime session) WebSocket endpoints
accept any connecting client with no auth check, broadcasting live prediction/accuracy
events to anyone who connects.

### 4.8 SSRF surface in MCP "open/drive the web app" tools (Medium)
`mcp_server/tools.py` reads `BUCK_API_URL` / `BUCK_UI_URL` from environment/settings and
uses them to make outbound requests and construct a browser-opened URL
(`open_buck_ui`, ~line 646) via `urlencode` on a `tab`/`symbol` the *caller* (an LLM
tool-call, ultimately steerable by whatever the MCP client is told) supplies. There's no
allowlist restricting these base URLs to localhost, so a tampered `.env` (or an MCP
client merging in a malicious server config) can redirect Buck's "drive the web app"
behavior, and unsanitized `symbol`/`tab` values flow into a URL the user's browser will
open.

### 4.9 `install_mcp.sh` passes unvalidated paths into an embedded Python subprocess (Low)
`install_mcp.sh:87-124` interpolates `sys.argv[1..3]` (config path, interpreter path,
repo dir) into an embedded Python script without validation. Low risk in the normal
"you run this against your own machine" flow, but it's a script that edits Claude
Desktop's config file with attacker-influenceable values if any of the inputs are ever
templated from an untrusted source.

### 4.10 No rate limiting anywhere
Every LLM-calling endpoint (`/analyze`, `/batch`, `/claude/predict`, RL training/predict
endpoints) is unthrottled. A single client can trigger unbounded LLM spend (your
OpenAI/Anthropic bill) or unbounded LSTM/RL training load (CPU/GPU exhaustion) with no
backpressure.

---

## 5. Compliance & Legal

### 5.1 No "not financial advice" disclaimer anywhere (High)
Searched README, SECURITY.md, CONTRIBUTING.md, the LLM system prompt
(`agent_scripts/claude_predictor.py`), and the UI components — there is **no disclaimer**
stating that forecasts are not investment/financial advice, no risk warning, and no
statement of accuracy limitations shown to an end user. A tool that emits a "next-day
OHLC forecast" with a numeric "confidence score" and ships a public web UI is squarely
in territory regulators (SEC/FINRA in the US, SEBI in India — relevant given the
Indian-equities focus: `MARKET_EXCHANGE=NSE`, `INDIAN_API_KEY`, `.NS`-suffixed symbols
throughout) scrutinize for "investment advice" framing. This is a real exposure if this
tool is ever shown to, or used to make decisions by, someone other than its developer.

### 5.2 No data-handling/privacy policy despite collecting third-party data (Medium)
News content is fetched from third-party APIs (`agent_scripts/data_providers.py`) and
persisted to local files (`inputs/`); `SECURITY.md` mentions reviewing `inputs/` before
sharing the project folder, but there's no retention policy, no PII handling statement
(news content could reference named individuals), and no statement on redistribution
rights for the fetched news content being stored to disk indefinitely.

### 5.3 Saved LLM prompts are tracked in git (Medium)
`inputs/BHEL.NS_*.json` / `*_prompts.txt` and `inputs/ITC.NS_*` are **committed to the
repository** (confirmed via `git ls-files inputs/`), despite `SECURITY.md` describing
`inputs/` as a local debugging directory to review "before sharing your project folder."
These files don't appear to contain API keys, but committing real run artifacts
(potentially containing licensed news-API content, or symbol/strategy information the
maintainer didn't intend to publish) into a public Apache-2.0-licensed repo is the kind
of thing the security docs explicitly warn against doing, yet it has already happened.

### 5.4 Apache-2.0 NOTICE / header inconsistency (Low)
The repo ships an Apache 2.0 `LICENSE`, but only `agent_scripts/__init__.py` carries any
copyright/license header among the Python sources spot-checked — most source files have
none. Not a blocking issue under Apache-2.0 (headers aren't mandated by the license
itself), but inconsistent with the LICENSE file's own boilerplate-header guidance and
worth normalizing if this is ever distributed as a dependency.

### 5.5 Third-party dependency licenses unaudited
`requirements.txt` and `UI/frontend/package.json` pull in dozens of transitive
dependencies (PyTorch, FastAPI, Plotly, React ecosystem, `headroom-ai[all]`, etc.) with
no license-compatibility audit against Apache-2.0 redistribution, and no `pip-licenses`
/ `license-checker` step in CI.

---

## 6. CI/CD — GitHub Actions Errors & Gaps

### 6.1 CodeQL workflow is entirely commented out — it never runs (High)
`.github/workflows/codeql.yml` is **100% comments** (every line is prefixed `#`,
confirmed by direct read — there is no active YAML in the file at all). This means:
- There is **no CodeQL job in this repo's Actions tab at all** — GitHub will not even
  register it as a workflow, since a file with no live YAML produces nothing to run.
- The repo's recent commit history (`5ec8d7d Add CodeQL analysis workflow configuration`,
  `26e4ace Refactor comments in codeql.yml`) gives the appearance that CodeQL scanning
  was added and tuned, when in fact zero scanning has ever executed from this file.
  Anyone glancing at `.github/workflows/` and seeing `codeql.yml` present will reasonably
  but incorrectly assume static security analysis is running on every push/PR.
- **This is the most actionable, lowest-effort fix in this entire document**: uncomment
  the file (it's a clean, valid GitHub-generated default-setup workflow) or delete it to
  stop implying coverage that doesn't exist.

### 6.2 `python-package.yml` only tests `agent_scripts/` (Medium)
`.github/workflows/python-package.yml` runs
`pytest tests/ -v --tb=short -x -m "not network" --cov=agent_scripts --cov-report=term-missing`.
Coverage is scoped to `agent_scripts` only — `UI/backend/` (the FastAPI app, routes,
auth-relevant code from §4), `mcp_server/`, `accuracy/`, `realtime/`, and `tools/rl/`
are **not measured for coverage**, and the workflow never installs/builds/lints the
frontend (`UI/frontend/`) or runs `npm test`/`npm run build` — a broken frontend build
would not be caught by CI at all.

### 6.3 `-x` flag means CI stops at the first failure (Low)
`pytest ... -x` aborts the whole matrix job on the first failing test, so a single
unrelated flaky/networked test can mask every other failure in the same run, costing a
full debug cycle to discover the next failure (especially painful across a 2-version
Python matrix where you'd want to see both legs' results).

### 6.4 No dependency-vulnerability scanning in CI (Medium)
Neither `pip-audit`/`safety` (Python) nor `npm audit` (frontend) run in any workflow.
`SECURITY.md` recommends running both manually but nothing automates or enforces it —
combined with §7.1 (unpinned dependencies), there is no signal when a new CVE lands in a
transitive dependency.

### 6.5 No frontend CI at all
There is no workflow that installs `UI/frontend` dependencies, type-checks
(`tsconfig.json` exists, implying intent to type-check), lints, or builds the React app.
A TypeScript compile error or a broken `npm run build` would only be discovered by a
human running `main.py` locally.

### 6.6 Branch hygiene: stale `security-fixes` branch (Low)
`origin/security-fixes` is fully merged into `main` (no unique commits ahead) but still
exists as a remote branch and diffs against current `main` with large deletions for
files `main` has since re-added/expanded (e.g. `codeql.yml`, `mcp_server/tools.py`). It
is dead weight that risks confusing future contributors into thinking it represents
in-flight security work.

---

## 7. Dependency Management

### 7.1 No version pins anywhere in `requirements.txt` (High)
`requirements.txt` has **zero** `==` pins (confirmed: 0 matches for `==`). Core,
high-blast-radius dependencies — `pydantic`, `pydantic-settings`, `fastapi`, `torch`,
`anthropic` (`>=0.40.0`), `mcp` (`>=1.0.0`) — float to whatever is latest at install
time. A breaking upstream release (e.g. a Pydantic v2→v3 jump, an MCP protocol change,
an Anthropic SDK breaking change) can silently break the app for every fresh install or
CI run with no warning, and two installs done a month apart can resolve to different
dependency trees, undermining reproducibility.

### 7.2 Duplicate `yfinance` entry (Low)
`requirements.txt` lists `yfinance` on both line 5 and line 19 — harmless to pip, but
signals the file isn't being curated carefully.

### 7.3 PyTorch with no CPU/GPU variant pinned (Low)
`torch` is listed bare; `SECURITY.md` itself notes it's a large dependency you may want
to strip out. Without pinning a CPU wheel explicitly, `pip install` can pull a large
CUDA-enabled build on machines without a GPU, ballooning install size/time for the one
LSTM tool that needs it.

### 7.4 `UI/backend/requirements.txt` partially duplicates root `requirements.txt`
A second, separate `fastapi`/`uvicorn`/`python-dotenv`/`plotly` requirements file exists
under `UI/backend/`, with no pin alignment guarantee against the root file — two
independent unpinned dependency lists for what is one application increases the chance
the backend and the rest of the app drift to incompatible library versions when
installed separately.

---

## 8. Testing Gaps

- **No tests for `UI/backend/routes.py` at all** — the entire FastAPI surface (every
  endpoint discussed in §4) has zero test coverage; the security issues in §4 would not
  be caught by any existing or CI-run test.
- **`batch_analyze()`'s concurrency is untested** — given the shared-global-state risk in
  §3.1, this is the one place a regression test would have real value, and it's missing.
- **Tools are tested only via direct `.execute()` calls**, never through the LangChain
  `@tool` wrapper / `get_stock_data()` context path that MCP and any LLM-driven tool call
  actually uses (`tests/test_tools.py`) — the code path most exposed to the global-state
  bug in §3.1 is the one path the test suite never exercises.
- **Only one network test exists** (`tests/test_data_provider.py`, marked
  `@pytest.mark.network`) and CI explicitly excludes it (`-m "not network"`) — meaning
  the only test that hits a real data provider never runs in CI at all.
- **No CLI tests** — `agent_scripts/cli.py`'s `analyze`/`batch`/`demo`/`simulate-rt`
  commands have no test coverage.
- **Accuracy telemetry swallows all errors silently** (`agent_scripts/buck.py`,
  `_record_prediction_telemetry()`) — a corrupted or failed write to the accuracy DB is
  logged and discarded, so a regression here would never surface as a test failure or
  even a visible runtime error.
- **RL and realtime modules (`tools/rl/`, `realtime/`) have thin or no test coverage**
  relative to their complexity (custom Gym-style env, two agent architectures, an
  ensemble dispatcher, a live simulator) — exactly the modules with the most intricate,
  hard-to-eyeball-correct logic in the repo.

---

## 9. Documentation Drift

- **README links to `docs/CLAUDE_MCP.md`** (`README.md:236`) for "the full tool surface
  and HTTP/SSE transport options" — **the `docs/` directory does not exist in the repo**.
  Any user following this link gets a 404 on GitHub.
- **README's Project Structure diagram lists an `output/` directory** as part of the
  shipped layout — it does not exist until the app creates it on first run; there's no
  `.gitkeep` reserving it (unlike `tools/rl/sessions/.gitkeep`, which does exist),
  so a fresh clone doesn't match the documented tree.
- **`tools/ml/`, `tools/utility/`, `tools/web/` are documented as having tools** ("6
  planned", "6 planned", "7 planned" in README) but contain only `readme.md` spec files
  and no implementation — accurately labeled "planned" in the README itself, but
  `CONTRIBUTING.md` directs new contributors at these directories as the "biggest area
  open for contribution" without flagging that the analyzer/predictor pipeline currently
  has no integration point exercised by tests for a third-party-contributed tool beyond
  the existing `tools/maths` pattern.
- **Claude system prompt references tools whose registration isn't verified against
  `mcp_server/registry.py` in this audit** (`agent_scripts/claude_predictor.py:84-90`
  mentions `rl_ensemble_predict`, `rl_simulate`, `list_rl_models`,
  `get_prediction_accuracy`, `compare_predictions_vs_actual`) — worth a direct
  cross-check that every tool name the system prompt promises Claude actually exists in
  the live registry, since a mismatch fails silently at tool-call time, not at startup.

---

## 10. Summary Table

| # | Area | Finding | Severity |
|---|------|---------|----------|
| 4.1 | Security | No authentication on any HTTP/WS/MCP surface | Critical |
| 4.2 | Security | Request-body API keys mutate live `os.environ`, cross-request | Critical |
| 6.1 | CI/CD | CodeQL workflow fully commented out — never runs | High |
| 4.3 | Security | `/config` echoes stored API keys to any caller | High |
| 4.4 | Security | Raw exception strings returned to clients | High |
| 4.5 | Security | Dynamic `tools/` import = unsandboxed code execution | High |
| 2.1 | Algorithms | No backtesting / validation / calibration framework anywhere | High |
| 2.4 | Algorithms | LSTM retrained from scratch every call, never persisted | High |
| 3.1 | Architecture | Global stock-data context unsafe under concurrent batch use | High |
| 5.1 | Compliance | No "not financial advice" disclaimer anywhere | High |
| 7.1 | Dependencies | Zero version pins in `requirements.txt` | High |
| 4.6 | Security | CORS: credentialed + wildcard methods/headers for any localhost port | Medium |
| 4.7 | Security | Unauthenticated WebSocket broadcasts live data | Medium |
| 4.8 | Security | SSRF-prone URL construction in MCP "open UI" tool | Medium |
| 5.2/5.3 | Compliance | No data-retention policy; debug prompt files committed to git | Medium |
| 6.2/6.4/6.5 | CI/CD | No frontend CI, no dep-vuln scanning, coverage scoped to one package | Medium |
| 2.2/2.3 | Algorithms | Signal aggregation & confidence score are structural, not validated | Medium |
| 3.2/3.3 | Architecture | Unsynchronized in-memory caches / partial locking in realtime state | Medium |
| 8 | Testing | Zero backend-route tests; concurrency path untested | High |
| 6.3/6.6/4.9/5.4/5.5/7.2-7.4/2.5-2.7/3.4-3.5/9 | Misc | See respective sections | Low–Medium |

---

## 11. Suggested First Fixes (lowest effort → highest leverage)

1. Uncomment or delete `.github/workflows/codeql.yml` — one-line-effort, closes a real
   "false sense of security" gap (§6.1).
2. Stop writing request-supplied API keys into `os.environ`; thread the key through the
   call explicitly instead of mutating process-global state (§4.2).
3. Remove API key values from the `/config` response, or gate it behind auth (§4.3).
4. Add a visible "not financial advice" disclaimer to the README, UI footer, and the
   LLM system prompt's output template (§5.1) — trivial to add, meaningfully reduces
   exposure.
5. Pin `requirements.txt` (`pip freeze` against a known-good environment) and add
   `pip-audit`/`npm audit` as a CI step (§7.1, §6.4).
6. Add an integration test that runs `Buck.batch_analyze()` against two symbols
   concurrently and asserts each result's data matches its requested symbol — this is
   the cheapest way to get a regression guard on §3.1 before attempting to fix it.
