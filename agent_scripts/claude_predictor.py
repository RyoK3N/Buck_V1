"""
agent_scripts.claude_predictor
──────────────────────────────
Claude-as-research-analyst predictor.

Unlike `OpenAIPredictor` (which formats Buck's analysis into a single prompt
and asks for a forecast), `ClaudePredictor` exposes Buck's user-facing
operations (single_analyze, batch_analyze, rl_predict, visualize, accuracy
introspection, …) as native Anthropic tools. Claude iterates: pick tools,
read results, reason, call more tools, then emit a JSON forecast.

This gives Claude the same capabilities a human Buck user has, and lets us
attribute correct/incorrect predictions to the specific tools Claude used
via the request_metadata trace stored in the accuracy DB.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

from .config import LOGGER, SETTINGS
from .interfaces import AnalysisResult, Forecast, IPredictor


_FORECAST_SCHEMA = (
    "Return ONE final JSON object with EXACTLY these keys:\n"
    '  "date":       next trading day, YYYY-MM-DD\n'
    '  "open":       float, 2 decimals\n'
    '  "high":       float, 2 decimals\n'
    '  "low":        float, 2 decimals\n'
    '  "close":      float, 2 decimals\n'
    '  "confidence": 0.0-1.0, 2 decimals\n'
    '  "reasoning":  string explaining the call (300-800 words)\n'
    "Wrap the JSON object in ```json ... ``` so it can be reliably extracted.\n"
    "Constraints: high >= max(open, close); low <= min(open, close)."
)


_SYSTEM_PROMPT = """You are Buck's research-analyst forecasting brain.

Your goal is to produce a next-trading-day OHLC forecast for the given
symbol. You DO NOT have direct market data — you must obtain it by calling
Buck's tools:

  single_analyze       — full pipeline (indicators + sentiment + a forecast
                         from the OpenAI predictor) for one symbol; this is
                         your primary evidence call
  batch_analyze        — concurrent analysis across peers / a watchlist for
                         cross-sectional context
  list_tools_registry  — discover what indicators are available so you can
                         pass a meaningful `selected_tools` list
  visualize            — request a Plotly chart you can reason over textually
  rl_predict           — directional signal from a trained RL model. For
                         PPO-continuous models this returns a target position
                         fraction in [0, 1]; for discrete DQN/A2C it returns
                         HOLD / BUY / SELL.
  rl_ensemble_predict  — multi-timeframe ensemble: stack a daily model with
                         an hourly/15-minute model for a more robust signal.
                         Weight each by its out-of-sample Sharpe (use
                         get_prediction_accuracy to fetch the numbers).
  rl_simulate          — latest live RL action snapshot
  list_rl_models       — see what models exist before calling rl_predict
  get_prediction_accuracy / list_recent_predictions /
  compare_predictions_vs_actual — ALWAYS consult these before committing.
                         They tell you how well models have been doing and
                         let you calibrate confidence honestly.

Workflow (think step-by-step, but only emit tool calls and the final JSON):

  1. Call get_prediction_accuracy for the symbol and recent window to know
     how trustworthy past forecasts have been.

  2. Call single_analyze on the symbol over the requested window. Note the
     individual signals (RSI / MACD / OBV / S/R / candlestick / LSTM) and
     whether they agree.

  3. Get a corroborating RL signal:
       - If a symbol-specific model exists, call rl_predict with it.
       - Otherwise call rl_ensemble_predict stacking the generic
         dqn_model_best (a mature, 1.17M-step generalist) with whatever
         else is available. Treat dqn_model_best as a directional filter —
         when its signal disagrees with single_analyze's forecast, DOWN-
         WEIGHT confidence rather than overriding.

  4. Apply the CONFLUENCE rule: a high-conviction call requires the RL
     signal AND at least 3 of the technical signals from single_analyze
     to agree on direction. If you don't get that confluence, your
     confidence MUST be ≤ 0.55, and you should explain in the reasoning
     why the signals diverge.

  5. SCALE confidence to evidence quality:
       confluence (RL + ≥3 techs aligned) + recent accuracy good   → 0.70-0.85
       confluence but recent accuracy mediocre                     → 0.55-0.70
       partial confluence (RL + 2 techs, or RL alone)              → 0.45-0.55
       conflicting signals or poor recent accuracy                 → 0.30-0.45
       no clear signal                                             → 0.20-0.35
     Confidence below 0.55 means "I am not betting this; skip the trade."

  6. Emit ONE final forecast as JSON. Be conservative — typical daily moves
     are 0.5-3%. The reasoning MUST cite (a) which tools you called, (b)
     the RL signal (continuous fraction or discrete action) and ensemble
     breakdown, (c) which technical signals agreed, (d) the recent accuracy
     window you consulted, and (e) the confidence band you landed in and why.

""" + _FORECAST_SCHEMA


_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


class ClaudePredictor(IPredictor):
    """Claude-as-research-analyst predictor using Anthropic native tool use."""

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        max_iterations: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        session_token_budget: Optional[int] = None,
    ):
        if not api_key:
            raise ValueError("ClaudePredictor requires an Anthropic API key")
        self.client = Anthropic(api_key=api_key)
        self.model = model or SETTINGS.claude_model
        self.max_iterations = max_iterations or SETTINGS.claude_max_iterations
        self.thinking_budget = thinking_budget or SETTINGS.claude_thinking_budget
        self.session_token_budget = session_token_budget or SETTINGS.claude_session_token_budget
        # Populated after each predict() call so callers (buck.py telemetry,
        # the chat endpoint) can persist what Claude actually did.
        self.last_trace: List[Dict[str, Any]] = []
        self.last_tokens_used: int = 0

    # ── IPredictor ───────────────────────────────────────────────────────────

    async def predict(
        self,
        analysis_results: List[AnalysisResult],
        **kwargs: Any,
    ) -> Forecast:
        symbol = analysis_results[0]["symbol"] if analysis_results else kwargs.get("symbol", "UNKNOWN")
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date") or datetime.now().strftime("%Y-%m-%d")
        interval = kwargs.get("interval", "1h")
        selected_tools = kwargs.get("selected_tools")

        user_prompt = self._build_user_prompt(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            selected_tools=selected_tools,
            seed_analysis=analysis_results,
        )

        try:
            text, trace = await self._run_tool_loop(symbol=symbol, user_prompt=user_prompt)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("ClaudePredictor failed for %s: %s", symbol, exc)
            self.last_trace = []
            return self._failure_forecast(str(exc))

        self.last_trace = trace
        return self._extract_forecast(text)

    # ── Tool loop ────────────────────────────────────────────────────────────

    async def _run_tool_loop(self, symbol: str, user_prompt: str) -> tuple[str, List[Dict[str, Any]]]:
        from mcp_server.registry import BUCK_TOOLS
        from mcp_server.tools import dispatch_async

        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        trace: List[Dict[str, Any]] = []
        tokens_used = 0

        for iteration in range(self.max_iterations):
            if tokens_used > self.session_token_budget:
                raise RuntimeError(
                    f"Claude session token budget ({self.session_token_budget}) exceeded"
                )

            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                tools=BUCK_TOOLS,
                messages=messages,
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens_used += int(getattr(usage, "input_tokens", 0) or 0)
                tokens_used += int(getattr(usage, "output_tokens", 0) or 0)

            # Append the assistant turn so the conversation stays coherent.
            messages.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})

            if response.stop_reason != "tool_use":
                final_text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
                return final_text, trace

            tool_results: List[Dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                tool_name = block.name
                args = block.input or {}
                LOGGER.info("[claude:%s] iter=%d → tool_use %s", symbol, iteration, tool_name)
                try:
                    result = await dispatch_async(tool_name, args)
                    ok = True
                    error = None
                except Exception as exc:  # noqa: BLE001
                    result = {"error": str(exc)}
                    ok = False
                    error = str(exc)
                trace.append({"iter": iteration, "name": tool_name, "args": args, "ok": ok, "error": error})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(_truncate_for_model(result), default=str),
                    "is_error": not ok,
                })
            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"max_iterations ({self.max_iterations}) exhausted")

    # ── Prompt + parsing helpers ─────────────────────────────────────────────

    def _build_user_prompt(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: str,
        interval: str,
        selected_tools: Optional[List[str]],
        seed_analysis: List[AnalysisResult],
    ) -> str:
        seed = ""
        if seed_analysis:
            seed = (
                "\n\nFor context, here is a Buck analysis snapshot already computed for this symbol "
                "(you may verify or supplement it with further tool calls):\n"
                f"{json.dumps(_truncate_for_model(seed_analysis, limit=4000), default=str)[:6000]}"
            )
        sel = f" Use selected_tools={selected_tools!r} when calling single_analyze." if selected_tools else ""
        return (
            f"Forecast the next trading day's OHLC for **{symbol}**.\n"
            f"Data window for tool calls: start_date={start_date or '(last ~60 days)'}, "
            f"end_date={end_date}, interval={interval}.{sel}"
            f"{seed}"
        )

    def _extract_forecast(self, text: str) -> Forecast:
        match = _JSON_BLOCK.search(text or "")
        raw = match.group(1) if match else (text or "")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Last-resort regex fallback.
            data = _regex_extract_forecast(raw)

        return _normalize_forecast(data)

    def _failure_forecast(self, reason: str) -> Forecast:
        return Forecast(
            date=datetime.now().strftime("%Y-%m-%d"),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            confidence=0.0,
            reasoning=f"Claude prediction failed: {reason}"[:800],
        )


# ── Module-level helpers ─────────────────────────────────────────────────────

def _truncate_for_model(value: Any, limit: int = 8000) -> Any:
    """Cheap recursive truncation so tool results don't blow up the context window."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…[truncated]"
    if isinstance(value, list):
        out = []
        used = 0
        for item in value:
            if used > limit:
                out.append("…[truncated list]")
                break
            t = _truncate_for_model(item, limit=max(200, limit // 4))
            out.append(t)
            used += len(str(t))
        return out
    if isinstance(value, dict):
        return {k: _truncate_for_model(v, limit=max(200, limit // 2)) for k, v in value.items()}
    return value


def _regex_extract_forecast(text: str) -> Dict[str, Any]:
    fields = {}
    for key in ("date",):
        m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
        if m:
            fields[key] = m.group(1)
    for key in ("open", "high", "low", "close", "confidence"):
        m = re.search(rf'"{key}"\s*:\s*([\-0-9.]+)', text)
        if m:
            fields[key] = float(m.group(1))
    m = re.search(r'"reasoning"\s*:\s*"(.+?)"\s*[},]', text, re.DOTALL)
    if m:
        fields["reasoning"] = m.group(1)
    return fields


def _normalize_forecast(data: Dict[str, Any]) -> Forecast:
    required = {"date", "open", "high", "low", "close", "confidence"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Claude forecast missing required fields: {sorted(missing)}")
    open_ = float(data["open"])
    high = float(data["high"])
    low = float(data["low"])
    close = float(data["close"])
    if high < max(open_, close):
        high = max(open_, close)
    if low > min(open_, close):
        low = min(open_, close)
    return Forecast(
        date=str(data["date"]),
        open=round(open_, 2),
        high=round(high, 2),
        low=round(low, 2),
        close=round(close, 2),
        confidence=round(max(0.0, min(1.0, float(data["confidence"]))), 2),
        reasoning=str(data.get("reasoning", ""))[:4000],
    )
