# Contributing to Buck

Thanks for your interest in contributing. This guide covers how to set up a development environment, the code conventions we follow, and how to submit changes.

---

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/Buck_V1.git
cd Buck_V1
```

### 2. Set up the Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Set up the frontend

```bash
cd UI/frontend
npm install
cd ../..
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your API key
```

### 5. Verify everything works

```bash
# Run tests
pytest tests/ -v

# Start the app
python main.py

# Check tool discovery
PYTHONPATH=. python -c "from agent_scripts.tools import ToolFactory; print(ToolFactory.get_available_tools())"
```

---

## What to Work On

### Planned tools (highest impact)

The `tools/` directory has three categories with detailed specs waiting to be implemented:

- **`tools/ml/`** — 6 scikit-learn tools (Random Forest, Gradient Boosting, Isolation Forest, SVM, Logistic Regression, KNN). See `tools/ml/readme.md`.
- **`tools/utility/`** — 6 risk/portfolio tools (risk metrics, volatility analysis, correlation, position sizing, feature engineering, drawdown analysis). See `tools/utility/readme.md`.
- **`tools/web/`** — 7 external data tools (news sentiment, economic calendar, earnings, SEC filings, social sentiment, insider transactions, options flow). See `tools/web/readme.md`.

Each readme has the tool name, parameters, output fields, signal convention, and dependencies. Pick one and implement it.

### Other areas

- Tests — we need more coverage, especially for the tools and analyzers
- Documentation — improving inline code comments, adding docstrings
- Bug fixes — check the GitHub issues
- Frontend improvements — UI/UX polish, new chart types
- Performance — profiling and optimizing the analysis pipeline

---

## Code Conventions

### Python

- Python 3.10+ (we use `X | Y` union syntax and `list[T]` generics)
- Type hints on all function signatures
- Imports: `from __future__ import annotations` at the top of every file
- Formatting: standard PEP 8, 4-space indentation
- Logging: use `LOGGER` from `agent_scripts.config`, not `print()`

### Tool files

Every tool file in `tools/<category>/` must follow this pattern:

```python
"""Docstring explaining what the tool does."""

from __future__ import annotations
import json
from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool

from agent_scripts.tools import BaseTool, get_stock_data


class MyTool(BaseTool):
    def __init__(self):
        super().__init__("my_tool", "One-line description")

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        # Analysis logic
        return {"signal": "BUY", "strength": 0.7}


@tool
def my_tool() -> str:
    """Docstring the LLM sees."""
    data = get_stock_data()
    if data is None:
        return json.dumps({"error": "No stock data available"})
    return json.dumps(MyTool().execute(data), default=str)


TOOL_CLASS = MyTool
TOOL_FUNC = my_tool
```

Rules:
- Every tool must return at minimum `signal` (BUY/SELL/HOLD) and `strength` (0.0 to 1.0)
- The `@tool` function must return a JSON string (the LLM reads it)
- No look-ahead bias in ML/DL tools: labels at bar `i` use only data from bars `0..i`
- Handle edge cases (insufficient data, NaN values) gracefully — return HOLD with strength 0.0
- Export `TOOL_CLASS` and `TOOL_FUNC` at module level

### Frontend (TypeScript/React)

- TypeScript strict mode
- Functional components with hooks
- Tailwind CSS for styling (no separate CSS files)
- Types defined in `UI/frontend/src/types/index.ts`

### Commits

- Write clear commit messages: what changed and why
- Keep commits focused — one logical change per commit
- Reference issue numbers when applicable (`Fixes #12`)

---

## Submitting Changes

### Pull request process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes

3. Run the tests:
   ```bash
   pytest tests/ -v
   ```

4. For frontend changes, verify TypeScript compiles:
   ```bash
   cd UI/frontend && npx tsc --noEmit
   ```

5. Push and open a PR against `main`:
   ```bash
   git push origin feature/my-feature
   ```

6. Fill in the PR description:
   - What does this change do?
   - How did you test it?
   - Any breaking changes?

### PR checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] New or changed code includes matching tests (see [testing requirements](#contributor-testing-requirements))
- [ ] No new linting errors
- [ ] TypeScript compiles if frontend was changed (`npx tsc --noEmit`)
- [ ] New tools follow the `TOOL_CLASS` + `TOOL_FUNC` pattern
- [ ] New dependencies added to `requirements.txt`
- [ ] Readme updated if applicable (tool directory readme or top-level README)

### Review

- PRs need at least one review before merge
- CI must pass (Python 3.10 and 3.11 test matrix)
- We'll provide feedback within a few days

---

## Running Tests Locally

### Test categories

The test suite is split into two categories:

| Category | Marker | What it tests | Network? |
|----------|--------|---------------|----------|
| **Unit tests** | *(default)* | Tools, analyzers, predictors, Buck orchestrator, config | No — all external calls are mocked |
| **Integration tests** | `@pytest.mark.network` | Live data from Yahoo Finance / Indian API | Yes — requires internet |

### Commands

```bash
# Full test suite (unit + integration)
pytest tests/ -v

# Unit tests only (offline, no API keys needed)
pytest tests/ -v -m "not network"

# Specific file
pytest tests/test_tools.py -v

# With coverage report
pytest tests/ --cov=agent_scripts --cov-report=term-missing

# HTML coverage report
pytest tests/ --cov=agent_scripts --cov-report=html
open htmlcov/index.html
```

### Shared fixtures (tests/conftest.py)

The `conftest.py` provides reusable fixtures for all test files:

| Fixture | Type | Description |
|---------|------|-------------|
| `sample_ohlcv_df` | `pd.DataFrame` | 200-row OHLCV DataFrame with DatetimeIndex |
| `sample_stock_data` | `StockData` | TypedDict wrapping the DataFrame |
| `sample_news_data` | `NewsData` | TypedDict with mixed-sentiment headlines |
| `sample_analysis_result` | `AnalysisResult` | Pre-built analysis result for predictor tests |
| `_patch_settings` | *(auto-use)* | Sets `OPENAI_API_KEY=test-key` and clears the LRU cache |

### Contributor testing requirements

When you submit a PR, your tests must match the area you changed:

| If you changed… | Add / update tests in… |
|-----------------|----------------------|
| A tool in `tools/` | `tests/test_tools.py` |
| `agent_scripts/analyzers.py` | `tests/test_analyzers.py` |
| `agent_scripts/predictors.py` | `tests/test_predictors.py` |
| `agent_scripts/buck.py` | `tests/test_buck.py` |
| `agent_scripts/config.py` | `tests/test_config.py` |
| `agent_scripts/data_providers.py` | `tests/test_data_provider.py` |

**New tools** must include at least one test in `test_tools.py` that:
1. Calls `execute()` with `sample_ohlcv_df`
2. Asserts the result contains `signal` (BUY/SELL/HOLD) and `strength` (0.0–1.0)

---

## Project Layout for Contributors

If you're new to the codebase, here's the reading order:

1. `agent_scripts/interfaces.py` — the protocols everything implements
2. `agent_scripts/tools.py` — BaseTool, ToolFactory, data context
3. `tools/maths/rsi.py` — a simple tool to understand the pattern
4. `agent_scripts/analyzers.py` — how tools are coordinated
5. `agent_scripts/predictors.py` — how analysis results become LLM prompts
6. `agent_scripts/buck.py` — the orchestrator that ties it all together

---

## Code of Conduct

We follow the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful, be constructive, and focus on the work.

---

## Questions?

Open an issue on GitHub. We're happy to help you get started.
