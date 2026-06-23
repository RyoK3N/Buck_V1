"""
Tests for the MCP server: registry/impl coverage and the network-free dispatch
paths that the rest of the system depends on.
"""

from __future__ import annotations

import asyncio

import pytest

from accuracy import db as accuracy_db
from accuracy import repository
from mcp_server.registry import BUCK_TOOLS, BUCK_TOOLS_BY_NAME
from mcp_server.tools import LAST_CALL, _IMPLS, dispatch_async, list_tool_names


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


def test_dispatch_unknown_tool_raises():
    with pytest.raises(KeyError):
        asyncio.run(dispatch_async("does_not_exist", {}))


def test_dispatch_list_intervals_no_network():
    result = asyncio.run(dispatch_async("list_available_intervals", {}))
    assert "intervals" in result
    assert "1d" in result["intervals"]


def test_dispatch_list_recent_predictions_reads_db(fresh_db):
    repository.record_prediction(
        symbol="DSP.NS", model="claude",
        forecast={"date": "2026-06-22", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "confidence": 0.4},
    )
    result = asyncio.run(dispatch_async("list_recent_predictions", {"limit": 5}))
    assert "predictions" in result
    assert any(r["symbol"] == "DSP.NS" for r in result["predictions"])


def test_last_call_telemetry_records_after_dispatch():
    LAST_CALL.clear()
    asyncio.run(dispatch_async("list_available_intervals", {}))
    assert "list_available_intervals" in LAST_CALL
    entry = LAST_CALL["list_available_intervals"]
    assert entry["ok"] is True
    assert entry["latency_ms"] >= 0
