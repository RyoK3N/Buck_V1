"""
Tests for the realtime SessionManager run-control surface (network-free).
"""

from __future__ import annotations

import pytest

from realtime.runner import SessionManager
from realtime.state import get_status


def test_status_for_unknown_symbol_is_inactive():
    mgr = SessionManager()
    st = mgr.status("NOPE.NS")
    assert st["active"] is False
    assert st["running"] is False


def test_replay_without_dates_raises():
    mgr = SessionManager()
    with pytest.raises(ValueError):
        mgr.start(symbol="X.NS", model_id="m", replay=True)


def test_stop_unknown_symbol_raises():
    mgr = SessionManager()
    with pytest.raises(KeyError):
        mgr.stop("GHOST.NS")


def test_list_sessions_empty_by_default():
    mgr = SessionManager()
    assert mgr.list_sessions() == []


def test_global_state_get_status_no_session():
    st = get_status("DEFINITELY_NOT_RUNNING.NS")
    assert st["active"] is False
