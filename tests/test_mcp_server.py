"""
Tests for the MCP server: registry/impl coverage and the network-free dispatch
paths that the rest of the system depends on.
"""

from __future__ import annotations

import asyncio

import pytest

from accuracy import db as accuracy_db
from accuracy import repository
import inspect

from mcp_server.registry import BUCK_TOOLS, BUCK_TOOLS_BY_NAME
from mcp_server.tools import LAST_CALL, _IMPLS, dispatch_async, get_wrapped, list_tool_names


@pytest.fixture
def fresh_db(tmp_path):
    accuracy_db.init_db(tmp_path / "accuracy.db")
    yield
    accuracy_db._DB_PATH = None


def test_registry_and_impls_match():
    reg_names = {t["name"] for t in BUCK_TOOLS}
    impl_names = set(_IMPLS.keys())
    assert reg_names == impl_names, (
        f"registry/impl mismatch: only-in-registry={reg_names - impl_names}, "
        f"only-in-impl={impl_names - reg_names}"
    )


def test_every_tool_has_input_schema_and_description():
    for tool in BUCK_TOOLS:
        assert tool.get("name"), tool
        assert tool.get("description"), tool
        schema = tool.get("input_schema")
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_list_tool_names_returns_all():
    names = list_tool_names()
    assert set(names) == set(BUCK_TOOLS_BY_NAME)


def test_wrapped_tools_preserve_impl_signature():
    """Regression: the context-engineering wrapper must expose each impl's real
    signature (not a single **kwargs param), otherwise FastMCP generates a schema
    with only an opaque `kwargs` field and MCP clients can't call the tools.
    """
    for name, impl in _IMPLS.items():
        wrapped = get_wrapped(name)
        wrapped_params = inspect.signature(wrapped).parameters
        impl_params = inspect.signature(impl).parameters
        assert set(wrapped_params) == set(impl_params), (
            f"{name}: wrapped signature {set(wrapped_params)} != impl {set(impl_params)}"
        )
        assert "kwargs" not in wrapped_params or "kwargs" in impl_params, (
            f"{name}: wrapper leaked a bare **kwargs param into the schema"
        )


def test_single_analyze_schema_exposes_real_params():
    """The single_analyze impl's required args must surface as signature params."""
    params = inspect.signature(get_wrapped("single_analyze")).parameters
    for required in ("symbol", "start_date", "end_date"):
        assert required in params
        assert params[required].default is inspect.Parameter.empty


def test_dispatch_unknown_tool_raises():
    with pytest.raises(KeyError):
        asyncio.run(dispatch_async("does_not_exist", {}))


def _payload(envelope):
    """dispatch_async returns a headroom envelope {_headroom, data}; when the
    compressor is a passthrough (headroom not installed) `data` is the raw dict."""
    assert "_headroom" in envelope and "data" in envelope
    return envelope["data"]


def test_dispatch_list_intervals_no_network():
    result = _payload(asyncio.run(dispatch_async("list_available_intervals", {})))
    assert "intervals" in result
    assert "1d" in result["intervals"]


def test_dispatch_list_recent_predictions_reads_db(fresh_db):
    repository.record_prediction(
        symbol="DSP.NS", model="claude",
        forecast={"date": "2026-06-22", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "confidence": 0.4},
    )
    result = _payload(asyncio.run(dispatch_async("list_recent_predictions", {"limit": 5})))
    assert "predictions" in result
    assert any(r["symbol"] == "DSP.NS" for r in result["predictions"])


def test_dispatch_visualize_session_no_session():
    """visualize_session must work network-free: no live session → empty spec."""
    result = _payload(asyncio.run(dispatch_async("visualize_session", {"chart": "equity_curve"})))
    assert result["active"] is False
    assert result["spec"]["chart"] == "equity_curve"
    assert result["spec"]["meta"].get("empty") is True


def test_dispatch_visualize_accuracy_reads_db(fresh_db):
    result = _payload(asyncio.run(dispatch_async("visualize_accuracy", {"window_days": 30})))
    assert "spec" in result
    assert result["spec"]["spec_version"] == "d3-buck/1"


def test_dispatch_visualize_predictions_reads_db(fresh_db):
    result = _payload(asyncio.run(dispatch_async("visualize_predictions", {"symbol": "DSP.NS"})))
    assert result["symbol"] == "DSP.NS"
    assert result["spec"]["spec_version"] == "d3-buck/1"


def test_last_call_telemetry_records_after_dispatch():
    LAST_CALL.clear()
    asyncio.run(dispatch_async("list_available_intervals", {}))
    assert "list_available_intervals" in LAST_CALL
    entry = LAST_CALL["list_available_intervals"]
    assert entry["ok"] is True
    assert entry["latency_ms"] >= 0
