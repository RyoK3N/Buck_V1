"""
Buck_V1.config
────────────────────────
Global configuration and logging utilities.
"""

from __future__ import annotations
import logging, sys
from functools import lru_cache

from dotenv import load_dotenv
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
        print("[CONFIG] Validation error:\n", exc, file=sys.stderr)
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

__all__ = ["SETTINGS", "LOGGER", "get_settings", "get_logger"]
