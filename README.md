# Buck 

An AI-powered stock analysis agent that combines technical indicators, deep learning models, sentiment analysis, and LLM reasoning to generate next-day price forecasts.

Buck fetches OHLCV data, runs it through a configurable set of analysis tools, feeds the structured results to an LLM (OpenAI or any compatible API), and returns a forecast with reasoning. The whole thing ships with a React UI, a FastAPI backend, and a CLI.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)
- [The Tools System](#the-tools-system)
- [The UI](#the-ui)
- [Python API](#python-api)
- [CLI](#cli)
- [Testing](#testing)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

---

## How It Works

```
User Input (symbol, dates, interval)
        |
        v
  Data Collection ──────── Yahoo Finance (OHLCV)
        |                  Indian API / RSS (News)
        v
  Tool Execution ──────── tools/maths/   (RSI, MACD, MA, OBV, candlestick, S/R)
        |                  tools/dl/      (LSTM price direction)
        |                  tools/ml/      (planned: RF, SVM, anomaly detection)
        |                  tools/web/     (planned: earnings, insider trades)
        |                  tools/utility/ (planned: risk metrics, vol analysis)
        v
  Analysis Layer ──────── TechnicalAnalyzer  (aggregates tool signals)
        |                  SentimentAnalyzer  (scores news headlines)
        v
  LLM Prediction ──────── OpenAI / OpenRouter / any compatible API
        |                  System prompt + structured analysis context
        v
  Forecast Output ──────── {date, open, high, low, close, confidence, reasoning}
```

Each tool returns a `signal` (BUY / SELL / HOLD) and a `strength` (0.0 - 1.0). The TechnicalAnalyzer aggregates these signals and passes the full analysis context to the LLM, which synthesizes everything into a next-day OHLC forecast with confidence score and detailed reasoning.

Tools use LangChain's `@tool` decorator so the agent can selectively invoke them. Users pick which tools to run from the UI.

---

## Project Structure

```
Buck_V1/
├── main.py                     # Entry point — starts backend + frontend
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
│
├── agent_scripts/              # Core agent logic
│   ├── buck.py                 # Buck orchestrator + BuckFactory
│   ├── interfaces.py           # Protocols: ITool, IAnalyzer, IPredictor, etc.
│   ├── tools.py                # BaseTool, ToolFactory (dynamic loader), data context
│   ├── analyzers.py            # TechnicalAnalyzer, SentimentAnalyzer, CompositeAnalyzer
│   ├── predictors.py           # OpenAIPredictor, EnsemblePredictor
│   ├── data_providers.py       # YahooFinanceProvider, IndianNewsProvider
│   ├── config.py               # Pydantic settings, logger
│   └── cli.py                  # Command-line interface
│
├── tools/                      # Modular analysis tools (auto-discovered)
│   ├── maths/                  # Technical indicators (6 tools)
│   ├── dl/                     # Deep learning models (1 tool, 4 planned)
│   ├── ml/                     # Classical ML models (6 planned)
│   ├── utility/                # Risk & portfolio utilities (6 planned)
│   ├── web/                    # External data fetchers (7 planned)
│   └── buck_visualizer/        # Plotly chart scripts (not agent tools)
│
├── UI/
│   ├── backend/                # FastAPI server
│   │   ├── main.py             # App factory, CORS, lifespan
│   │   ├── routes.py           # API endpoints
│   │   ├── models.py           # Pydantic request/response schemas
│   │   └── visualizer.py       # Chart generation logic
│   └── frontend/               # React + TypeScript + Vite
│       ├── src/
│       │   ├── App.tsx         # Root component (tabs: Single / Batch / Visualizer)
│       │   ├── api/client.ts   # Axios API client
│       │   ├── components/     # UI components
│       │   └── types/          # TypeScript interfaces
│       └── package.json
│
├── tests/                      # pytest test suite
├── inputs/                     # Saved LLM prompts (for debugging)
├── output/                     # Analysis result JSON files
│
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
└── LICENSE                     # Apache 2.0
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm (for the frontend)
- An LLM API key (OpenAI, OpenRouter, or any OpenAI-compatible endpoint)

### Installation

```bash
git clone https://github.com/RyoK3N/Buck_V1.git
cd Buck_V1

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies (handled automatically by main.py, or manually)
cd UI/frontend && npm install && cd ../..
```

### Environment

Copy the example config and fill in your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Optional — use OpenRouter for free models
OPENAI_BASE_URL=https://openrouter.ai/api/v1
CHAT_MODEL=nvidia/nemotron-3-super-120b-a12b:free

# Optional — Indian stock news API (https://stock.indianapi.in/)
INDIAN_API_KEY=

# Tuning
TEMPERATURE=0.0
MAX_COMPLETION_TOKENS=1500
NEWS_ITEMS=8
LOG_LEVEL=INFO
OUTPUT_DIR=output
```

Or set keys programmatically (creates/updates `.env` for you):

```python
from agent_scripts import set_api_keys
set_api_keys("your-openai-key", "your-indian-key")
```

---

## Running the Application

### Full stack (recommended)

```bash
python main.py
```

This starts both servers:
- Backend API at `http://localhost:8000`
- Frontend dev server at `http://localhost:5173`

Open `http://localhost:5173` in your browser.

### Backend only

```bash
python main.py --backend-only
# or directly:
uvicorn UI.backend.main:app --reload --port 8000
```

### Options

```
python main.py --port 9000        # custom backend port
python main.py --no-reload        # disable auto-reload
python main.py --backend-only     # skip frontend
```

Press `Ctrl+C` to stop all servers.

### Claude Desktop (MCP)

Buck ships an MCP server that exposes its operations as tools for Claude Desktop.
The easiest way to wire it up is the installer, which installs dependencies,
bootstraps `.env`, verifies the server, and registers the `buck` connector in
Claude Desktop's config (merging into any existing servers):

```bash
# Run from the repo root. Activate your virtualenv first if you use one —
# the installer registers whichever interpreter is active.
bash install_mcp.sh
```

Then restart Claude Desktop. To register it manually instead, add this to
`~/Library/Application Support/Claude/claude_desktop_config.json` (use an
**absolute** path to your Python and to the repo):

```json
{
  "mcpServers": {
    "buck": {
      "command": "/absolute/path/to/python3",
      "args": ["-m", "mcp_server.runner", "--transport", "stdio"],
      "cwd": "/absolute/path/to/Buck_V1",
      "env": {}
    }
  }
}
```

API keys are read from the repo's `.env` (loaded automatically by the runner),
so you don't need to put secrets in the Claude Desktop config. See
[docs/CLAUDE_MCP.md](docs/CLAUDE_MCP.md) for the full tool surface and HTTP/SSE
transport options.

---

## Configuration

All configuration is managed through environment variables (`.env` file) and loaded via Pydantic settings in `agent_scripts/config.py`.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | Your LLM API key |
| `OPENAI_BASE_URL` | No | (OpenAI default) | Custom endpoint (OpenRouter, local LLM, etc.) |
| `CHAT_MODEL` | No | `gpt-4o` | Model name |
| `TEMPERATURE` | No | `0.0` | LLM temperature |
| `MAX_COMPLETION_TOKENS` | No | `1500` | Max response tokens |
| `NEWS_ITEMS` | No | `8` | Number of news articles to fetch |
| `INDIAN_API_KEY` | No | — | Indian stock news API key |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `OUTPUT_DIR` | No | `output` | Directory for saved results |

---

## The Tools System

Tools live in `tools/<category>/` and are auto-discovered by `ToolFactory` at startup. Each tool file exports:

- `TOOL_CLASS` — a `BaseTool` subclass with the analysis logic
- `TOOL_FUNC` — a LangChain `@tool`-decorated function the agent can invoke

The `ToolFactory` scans `tools/*/`, imports every `.py` file that has a `TOOL_CLASS`, and registers it. No manual registration needed — drop a file in the right directory and it's available.

### Implemented tools

**tools/maths/** — Technical Analysis (6 tools)

| Tool | Signal Logic |
|------|-------------|
| `moving_average` | SMA/EMA crossover: short > long = BUY |
| `rsi` | RSI > 70 = SELL (overbought), < 30 = BUY (oversold) |
| `macd` | MACD > signal line = BUY, < signal = SELL |
| `obv` | OBV above its MA = BUY (buying pressure) |
| `candlestick_patterns` | Detects Doji, Hammer, Engulfing, Shooting Star, etc. |
| `support_resistance` | Price near support = BUY, near resistance = SELL |

**tools/dl/** — Deep Learning (1 tool)

| Tool | Description |
|------|-------------|
| `lstm_prediction` | 2-layer PyTorch LSTM, predicts UP/DOWN direction with probability |

### Planned tools

See the `readme.md` in each directory for detailed specs:

- **tools/ml/** — Random Forest, Gradient Boosting regime detection, Isolation Forest anomaly detection, SVM, Logistic Regression, KNN pattern matching
- **tools/utility/** — Risk metrics (Sharpe, VaR), volatility analysis, correlation/Hurst exponent, position sizing, drawdown analysis
- **tools/web/** — News sentiment, economic calendar, earnings data, SEC filings, Reddit sentiment, insider transactions, options flow

### Writing a new tool

```python
# tools/maths/my_indicator.py
import json
from typing import Any, Dict
import pandas as pd
from langchain_core.tools import tool
from agent_scripts.tools import BaseTool, get_stock_data

class MyIndicatorTool(BaseTool):
    def __init__(self):
        super().__init__("my_indicator", "Does something useful")

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        # Your analysis logic here
        return {"signal": "BUY", "strength": 0.7, "details": "..."}

@tool
def my_indicator() -> str:
    """Description the LLM sees when deciding whether to use this tool."""
    data = get_stock_data()
    if data is None:
        return json.dumps({"error": "No stock data available"})
    return json.dumps(MyIndicatorTool().execute(data), default=str)

TOOL_CLASS = MyIndicatorTool
TOOL_FUNC = my_indicator
```

That's it. Restart the server and the tool appears in the UI.

---

## The UI

The frontend is a React + TypeScript app built with Vite and styled with Tailwind CSS.

### Tabs

- **Single Analysis** — Analyze one stock. Pick a symbol, date range, interval, select which tools to run, and get a forecast with full indicator breakdown.
- **Batch Analysis** — Same thing but for multiple symbols at once, run concurrently.
- **Visualizer** — Interactive Plotly charts (candlestick, Bollinger Bands, MACD, RSI, volatility, returns histogram, news overlay).

### Components

| Component | What it does |
|-----------|-------------|
| `ConfigPanel` | API key and model configuration |
| `ToolsConfigPanel` | Checkbox UI for selecting which tools to run |
| `AnalysisForm` | Single-stock analysis form |
| `BatchForm` | Multi-stock batch form |
| `ResultsPanel` | Displays forecast, indicators, and sentiment |
| `VisualizerPanel` | Chart type selector and interactive chart display |
| `ForecastCard` | Next-day OHLC forecast with confidence |
| `SignalBadge` | Color-coded BUY / SELL / HOLD badge |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/config` | Server configuration (pre-fills UI forms) |
| GET | `/intervals` | Available data intervals |
| GET | `/tools` | List of available tool names |
| GET | `/tools-registry` | Tool categories with metadata (for the UI) |
| POST | `/analyze` | Run analysis on a single stock |
| POST | `/batch` | Run batch analysis on multiple stocks |
| GET | `/chart-types` | Available visualization types |
| POST | `/visualize` | Generate an interactive chart |

---

## Python API

```python
import asyncio
from agent_scripts import create_agent

async def main():
    agent = create_agent(
        openai_api_key="your-key",
        model="gpt-4o",
    )

    result = await agent.analyze_and_predict(
        symbol="AAPL",
        start_date="2024-06-01",
        end_date="2024-06-15",
        interval="1h",
    )

    forecast = result["forecast"]
    print(f"Close: ${forecast['close']:.2f}")
    print(f"Confidence: {forecast['confidence']:.0%}")
    print(f"Reasoning: {forecast['reasoning']}")

asyncio.run(main())
```

### Batch analysis

```python
result = await agent.batch_analyze(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date="2024-06-01",
    end_date="2024-06-15",
    interval="1h",
    max_concurrent=3,
)

for symbol, analysis in result["results"].items():
    print(f"{symbol}: {analysis['forecast']['close']:.2f}")
```

### With specific tools

```python
from agent_scripts import BuckFactory

agent = BuckFactory.create_production_agent(
    openai_api_key="your-key",
    model="gpt-4o",
    selected_tools=["rsi", "macd", "lstm_prediction"],
)
```

---

## CLI

```bash
# Single stock analysis
python -m agent_scripts.cli analyze AAPL --start 2024-06-01 --end 2024-06-15

# Batch analysis
python -m agent_scripts.cli batch AAPL GOOGL MSFT --start 2024-06-01 --end 2024-06-15

# Demo mode
python -m agent_scripts.cli demo --symbol BHEL.NS

# Custom parameters
python -m agent_scripts.cli analyze BHEL.NS \
    --start 2024-01-01 \
    --end 2024-01-10 \
    --interval 30m \
    --model gpt-4o \
    --api-key your-key
```

---

## Testing

The test suite covers tools, analyzers, predictors, the Buck orchestrator, and configuration. Tests are split into offline unit tests and network-dependent integration tests.

```bash
# Full test suite
pytest tests/ -v

# Unit tests only (no network, no API keys needed)
pytest tests/ -v -m "not network"

# With coverage
pytest tests/ --cov=agent_scripts --cov-report=term-missing
```

### Test files

| File | What it tests |
|------|---------------|
| `test_config.py` | Settings, API key persistence, LRU cache |
| `test_tools.py` | ToolFactory discovery, BaseTool, RSI, MACD, MA, OBV, candlestick, S/R |
| `test_analyzers.py` | TechnicalAnalyzer, SentimentAnalyzer, CompositeAnalyzer, AnalyzerFactory |
| `test_predictors.py` | OpenAIPredictor (mocked), EnsemblePredictor, PredictorFactory |
| `test_buck.py` | Buck orchestrator, cache, confidence, context manager, BuckFactory |
| `test_version.py` | Version constant and CLI output |
| `test_data_provider.py` | Yahoo Finance live fetch (`@pytest.mark.network`) |

CI runs on every push to `main` and on pull requests, testing against Python 3.10 and 3.11 with coverage reporting. See `.github/workflows/python-package.yml`.

For contributor testing requirements, see [CONTRIBUTING.md](CONTRIBUTING.md#contributor-testing-requirements).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. The short version:

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run `pytest` and make sure it passes
5. Open a pull request against `main`

The biggest area open for contribution is implementing the planned tools in `tools/ml/`, `tools/utility/`, and `tools/web/`. Each directory has a `readme.md` with detailed specs for every proposed tool.

---

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

**Key points:**
- Never commit API keys or `.env` files (`.gitignore` excludes them)
- The application stores API keys in environment variables, not in code
- LLM prompts and analysis inputs are saved to `inputs/` locally for debugging — review this directory before sharing

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---


