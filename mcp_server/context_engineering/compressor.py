"""
mcp_server.context_engineering.compressor
────────────────────────────────────────────
Thin, defensive bridge to `headroom` (https://github.com/headroomlabs-ai/headroom).

We compress a tool's JSON result by handing it to `headroom.compress(...)` as a
**tool message** — that's the shape headroom's ContentRouter sends through its
SmartCrusher / CodeCompressor (plain user/assistant prose is deliberately
protected). The returned `CompressResult` carries the compressed messages plus
exact `tokens_before` / `tokens_after`, which we use for accounting.

Everything is guarded by a circuit breaker and a broad try/except: if `headroom`
is missing, the breaker is open, or anything raises, we return the original
payload uncompressed — correctness always beats savings.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

from .cost_tracker import estimate_tokens
from .patterns import CircuitBreaker, CircuitOpenError

_BREAKER = CircuitBreaker(fail_max=3, reset_timeout=60.0, name="headroom")

# Cached probe state. _OK is None until probed.
_OK: Optional[bool] = None


def _have_headroom() -> bool:
    global _OK
    if _OK is None:
        try:
            import headroom  # noqa: F401

            _OK = hasattr(headroom, "compress")
        except Exception:
            _OK = False
    return bool(_OK)


def headroom_available() -> bool:
    """True if a usable headroom compression entry point is importable."""
    return _have_headroom()


def _extract_text(messages: Any) -> Optional[str]:
    """Pull the compressed content string out of headroom's returned messages."""
    if not messages:
        return None
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else last
    if isinstance(content, str):
        return content
    # content may be a list of blocks: [{"type": "...", "content"/"text": "..."}]
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                parts.append(str(blk.get("content") or blk.get("text") or ""))
        return "".join(parts) if parts else None
    return None


def _default_model() -> Optional[str]:
    try:
        from agent_scripts.config import SETTINGS

        return getattr(SETTINGS, "claude_model", None)
    except Exception:
        return None


def _enabled() -> bool:
    try:
        from agent_scripts.config import SETTINGS

        return bool(getattr(SETTINGS, "headroom_enabled", True))
    except Exception:
        return True


def _run_headroom(text: str, model: Optional[str]) -> Tuple[str, int, int]:
    """Call headroom.compress on a tool message. Returns (text, before, after)."""
    import headroom

    result = headroom.compress(
        [{"role": "tool", "content": text}],
        model=model or _default_model() or "claude-opus-4-5",
    )
    compressed = _extract_text(getattr(result, "messages", None))
    if compressed is None:
        raise ValueError("headroom returned no compressed content")
    before = int(getattr(result, "tokens_before", estimate_tokens(text)))
    after = int(getattr(result, "tokens_after", estimate_tokens(compressed)))
    return compressed, before, after


def compress_payload(obj: Any, model: Optional[str] = None) -> Tuple[Any, dict]:
    """Compress a JSON-serialisable payload via headroom.

    Returns ``(payload, stats)`` where ``stats`` always carries
    ``tokens_raw`` / ``tokens_compressed`` / ``compressed`` (bool). On any
    failure the original ``obj`` is returned unchanged with ``compressed=False``.
    """
    try:
        # Default separators (with spaces) on purpose: headroom's ContentRouter
        # classifies minified JSON as `noop` and skips it, whereas normal spaced
        # JSON is sent through SmartCrusher. We measure savings against this same
        # baseline so the numbers are honest (not inflated by pretty-printing).
        raw_text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    except Exception:
        raw_text = str(obj)
    fallback_raw = estimate_tokens(raw_text)

    def _passthrough(reason: str) -> Tuple[Any, dict]:
        return obj, {"tokens_raw": fallback_raw, "tokens_compressed": fallback_raw, "compressed": False, "reason": reason}

    if not _enabled():
        return _passthrough("disabled")
    if not _have_headroom():
        return _passthrough("headroom_unavailable")

    try:
        compressed_text, before, after = _BREAKER.call(_run_headroom, raw_text, model)
    except CircuitOpenError:
        return _passthrough("breaker_open")
    except Exception as exc:  # noqa: BLE001
        return _passthrough(f"error:{type(exc).__name__}")

    # If compression didn't help, keep the original (cleaner for Claude).
    if after >= before:
        return obj, {"tokens_raw": before, "tokens_compressed": before, "compressed": False, "reason": "no_gain"}

    return compressed_text, {
        "tokens_raw": before,
        "tokens_compressed": after,
        "compressed": True,
        "reason": "ok",
    }
