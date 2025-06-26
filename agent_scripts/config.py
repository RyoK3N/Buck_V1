"""
Buck_V1.config
────────────────────────
Global configuration and logging utilities.
"""

from __future__ import annotations
import logging, sys
from functools import lru_cache

import os
from dotenv import load_dotenv, set_key
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings

load_dotenv()  # quietly loads .env if present


class Settings(BaseSettings):
    # --- OpenAI -------------------------------------------------------------
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    chat_model: str = "gpt-4.1"                 # e.g. "gpt-4o", "o4-mini"
    embed_model: str = "text-embedding-ada-002"

    # --- Indian API for News -----------------------------------------------
    indian_api_key: str = Field("", env="INDIAN_API_KEY")

    # --- Generation controls -----------------------------------------------
    temperature: float = 0.0                   # deterministic JSON
    max_completion_tokens: int = 1500          # Increased from 1000 to 1500 for complete detailed reasoning

    # --- Domain tweaks ------------------------------------------------------
    news_items: int = 8

    # --- Output -------------------------------------------------------------
    output_dir: str = Field("output", env="OUTPUT_DIR")

    # --- Logging ------------------------------------------------------------
    log_level: str = "INFO"

    model_config = dict(env_file=".env", case_sensitive=False)

    # Helper: which token param?
    @property
    def use_completion_tokens(self) -> bool:
        return self.chat_model.lower().startswith("o4") or "gpt-4o" in self.chat_model.lower()


# --- singletons --------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        print(
            "[CONFIG] Missing configuration.\n"
            "Please set OPENAI_API_KEY and other variables via environment or .env file.\n",
            exc,
            file=sys.stderr,
        )
        print(
            "Hint: you can call agent_scripts.config.set_api_keys('<openai>', '<indian>')"
            " to create the .env file automatically.",
            file=sys.stderr,
        )
        sys.exit(1)


@lru_cache(maxsize=1)
def get_logger() -> logging.Logger:
    s = get_settings()
    logger = logging.getLogger("market_forecaster")
    if not logger.handlers:
        logger.setLevel(getattr(logging, s.log_level.upper(), logging.INFO))
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(h)
    return logger


SETTINGS = get_settings()
LOGGER   = get_logger()

def set_api_keys(openai_api_key: str, indian_api_key: str = "", env_file: str = ".env") -> None:
    """Persist API keys to a .env file and refresh settings."""
    set_key(env_file, "OPENAI_API_KEY", openai_api_key)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    if indian_api_key:
        set_key(env_file, "INDIAN_API_KEY", indian_api_key)
        os.environ["INDIAN_API_KEY"] = indian_api_key

    get_settings.cache_clear()
    global SETTINGS
    SETTINGS = get_settings()
    LOGGER.info("API keys updated and saved to %s", env_file)


def ensure_api_keys() -> bool:
    """Validate that the OpenAI API key is available."""
    if os.getenv("OPENAI_API_KEY"):
        return True

    LOGGER.error(
        "OPENAI_API_KEY is missing. Set it via environment, a .env file, or call set_api_keys()."
    )
    return False

__all__ = [
    "SETTINGS",
    "LOGGER",
    "get_settings",
    "get_logger",
    "set_api_keys",
    "ensure_api_keys",
]
