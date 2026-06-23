"""Smoke-test every MCP tool via dispatch_async. Skips long/destructive ones."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

from mcp_server.runner import _init_accuracy_db  # noqa: E402
from mcp_server.tools import _IMPLS, dispatch_async  # noqa: E402

_init_accuracy_db()


end = date.today() - timedelta(days=1)
start = end - timedelta(days=30)
SYM = "AAPL"
RL_SYM = "BHEL.NS"  # existing dqn_model_best was trained on NSE data
RL_MODEL = "dqn_model_best"
PPO_MODEL = "smoke_ppo_continuous"  # trained inline by the test

CASES = [
    ("list_available_intervals", {}),
    ("list_chart_types", {}),
    ("list_tools_registry", {}),
    ("list_rl_models", {}),
    ("get_prediction_accuracy", {"symbol": SYM, "window_days": 30}),
    ("list_recent_predictions", {"symbol": SYM, "limit": 5}),
    ("compare_predictions_vs_actual", {"symbol": SYM, "lookback_days": 30}),
    ("visualize", {"symbol": SYM, "start_date": str(start), "end_date": str(end),
                   "chart_type": "price_ma", "interval": "1d"}),
    ("single_analyze", {"symbol": SYM, "start_date": str(start), "end_date": str(end),
                        "interval": "1d", "selected_tools": ["rsi", "macd"]}),
    ("batch_analyze", {"symbols": [SYM, "MSFT"], "start_date": str(start),
                       "end_date": str(end), "interval": "1d",
                       "selected_tools": ["rsi"], "max_concurrent": 2}),
    ("rl_predict", {"symbol": RL_SYM, "start_date": str(start), "end_date": str(end),
                    "model_id": RL_MODEL, "interval": "1d"}),
    ("rl_simulate", {"symbol": RL_SYM, "model_id": RL_MODEL, "interval": "1d"}),
    # New stack — tiny PPO-continuous train + predict + ensemble
    ("rl_train", {"symbol": RL_SYM, "start_date": str(start - timedelta(days=180)),
                  "end_date": str(end), "model_id": PPO_MODEL,
                  "algorithm": "ppo_continuous", "episodes": 2, "interval": "1d"}),
    ("rl_predict", {"symbol": RL_SYM, "start_date": str(start), "end_date": str(end),
                    "model_id": PPO_MODEL, "interval": "1d"}),
    ("rl_ensemble_predict", {"symbol": RL_SYM, "start_date": str(start), "end_date": str(end),
                              "models": [
                                  {"model_id": "dqn_model_best", "interval": "1d", "weight": 1.0},
                                  {"model_id": PPO_MODEL, "interval": "1d", "weight": 0.5},
                              ]}),
]
SKIP: set[str] = set()  # nothing skipped — we exercise rl_train via PPO-continuous


async def main() -> int:
    print(f"Registered impls: {len(_IMPLS)}  |  cases: {len(CASES)}  |  skipped: {sorted(SKIP)}")
    print(f"OPENAI_API_KEY set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    print(f"ANTHROPIC_API_KEY set: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
    print("=" * 80)

    results = {}
    for name, args in CASES:
        print(f"\n→ {name}({json.dumps(args)[:120]})")
        try:
            res = await asyncio.wait_for(dispatch_async(name, args), timeout=120)
            keys = list(res.keys()) if isinstance(res, dict) else type(res).__name__
            print(f"  OK  keys={keys}")
            results[name] = ("OK", str(keys)[:200])
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc().splitlines()
            print(f"  FAIL  {type(exc).__name__}: {exc}")
            for line in tb[-8:]:
                print(f"    {line}")
            results[name] = ("FAIL", f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, (status, info) in results.items():
        print(f"  {status:4}  {name:<32}  {info}")

    fail = [n for n, (s, _) in results.items() if s == "FAIL"]
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
