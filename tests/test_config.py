import os
from agent_scripts.config import set_api_keys, get_settings, ensure_api_keys


def test_set_api_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)

    # Clear the LRU cache before testing so stale values don't leak
    get_settings.cache_clear()

    set_api_keys("test-openai", "test-indian", env_file=str(env_file))
    settings = get_settings()
    assert settings.openai_api_key == "test-openai"
    assert settings.indian_api_key == "test-indian"

    # Cleanup: clear again so subsequent tests start fresh
    get_settings.cache_clear()


def test_ensure_api_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # create temporary settings
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert ensure_api_keys()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert ensure_api_keys() is False
