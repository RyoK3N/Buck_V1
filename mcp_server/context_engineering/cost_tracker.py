"""
mcp_server.context_engineering.cost_tracker
─────────────────────────────────────────────
The process-wide cost / token tracker (Observer sink).

This is *the variable that tracks cost and tokens* referenced by the task:
a single `USAGE` singleton every wrapped MCP tool reports into. It accumulates
raw-vs-compressed token counts and a USD cost estimate, plus a per-tool
breakdown, and exposes a JSON-safe `snapshot()` consumed by the `headroom_stats`
MCP tool and the `/mcp/status` REST endpoint.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def estimate_tokens(text: str) -> int:
    """Best-effort token count.

    Prefers headroom's own counter if available, then `tiktoken`, and finally a
    ~4-chars-per-token heuristic so the tracker works with zero extra deps.
    """
    if not text:
        return 0
    # 1) headroom may ship a counter
    try:  # pragma: no cover - optional dep
        from headroom import count_tokens as _hr_count  # type: ignore

        return int(_hr_count(text))
    except Exception:
        pass
    # 2) tiktoken if present
    try:  # pragma: no cover - optional dep
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        pass
    # 3) heuristic
    return max(1, len(text) // 4)


class UsageTracker:
    """Thread-safe accumulator of token + cost usage across MCP tool calls."""

    def __init__(self, price_per_mtok: Optional[float] = None) -> None:
        self._lock = threading.Lock()
        self._price_per_mtok = price_per_mtok if price_per_mtok is not None else self._default_price()
        self.reset()

    @staticmethod
    def _default_price() -> float:
        try:
            from agent_scripts.config import SETTINGS

            return float(getattr(SETTINGS, "headroom_price_per_mtok", 15.0))
        except Exception:
            return 15.0

    # ── mutation ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.compressed_calls = 0
            self.cache_hits = 0
            self.tokens_raw = 0
            self.tokens_compressed = 0
            self.per_tool: Dict[str, Dict[str, Any]] = {}
            self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record(
        self,
        tool: str,
        tokens_raw: int,
        tokens_compressed: int,
        *,
        compressed: bool = True,
        cache_hit: bool = False,
    ) -> Dict[str, Any]:
        """Record one tool call. Returns the per-call delta (for logging)."""
        tokens_raw = max(0, int(tokens_raw))
        tokens_compressed = max(0, int(tokens_compressed))
        saved = max(0, tokens_raw - tokens_compressed)
        with self._lock:
            self.calls += 1
            if compressed:
                self.compressed_calls += 1
            if cache_hit:
                self.cache_hits += 1
            self.tokens_raw += tokens_raw
            self.tokens_compressed += tokens_compressed

            slot = self.per_tool.setdefault(
                tool,
                {"calls": 0, "tokens_raw": 0, "tokens_compressed": 0, "tokens_saved": 0, "cache_hits": 0},
            )
            slot["calls"] += 1
            slot["tokens_raw"] += tokens_raw
            slot["tokens_compressed"] += tokens_compressed
            slot["tokens_saved"] += saved
            if cache_hit:
                slot["cache_hits"] += 1

        return {
            "tool": tool,
            "tokens_raw": tokens_raw,
            "tokens_compressed": tokens_compressed,
            "tokens_saved": saved,
            "cost_saved_usd": round(saved / 1_000_000 * self._price_per_mtok, 6),
        }

    # ── read ──────────────────────────────────────────────────────────────────
    def _cost(self, tokens: int) -> float:
        return round(tokens / 1_000_000 * self._price_per_mtok, 6)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            tokens_saved = max(0, self.tokens_raw - self.tokens_compressed)
            reduction_pct = (
                round(100.0 * tokens_saved / self.tokens_raw, 2) if self.tokens_raw else 0.0
            )
            per_tool = {
                name: {
                    **vals,
                    "reduction_pct": (
                        round(100.0 * vals["tokens_saved"] / vals["tokens_raw"], 2)
                        if vals["tokens_raw"]
                        else 0.0
                    ),
                    "cost_saved_usd": self._cost(vals["tokens_saved"]),
                }
                for name, vals in self.per_tool.items()
            }
            return {
                "started_at": self.started_at,
                "price_per_mtok_usd": self._price_per_mtok,
                "calls": self.calls,
                "compressed_calls": self.compressed_calls,
                "cache_hits": self.cache_hits,
                "tokens_raw": self.tokens_raw,
                "tokens_compressed": self.tokens_compressed,
                "tokens_saved": tokens_saved,
                "reduction_pct": reduction_pct,
                "est_cost_raw_usd": self._cost(self.tokens_raw),
                "est_cost_compressed_usd": self._cost(self.tokens_compressed),
                "est_cost_saved_usd": self._cost(tokens_saved),
                "per_tool": per_tool,
            }


# The single process-wide tracker every wrapped tool reports into.
USAGE = UsageTracker()
