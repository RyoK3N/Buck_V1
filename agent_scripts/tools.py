"""
Buck_V1.tools
──────────────────────────
BaseTool base class, shared data context, and dynamic ToolFactory that
auto-discovers LangChain @tool functions from the ``tools/`` directory.
"""

from __future__ import annotations
import importlib.util
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .interfaces import ITool
from .config import LOGGER


# ── Shared stock-data context ────────────────────────────────────────────────
# Tools loaded from tools/ call get_stock_data() to access the DataFrame
# that was set by the analyzer before tool invocation.

_stock_data: Optional[pd.DataFrame] = None


def set_stock_data(data: pd.DataFrame) -> None:
    """Store the current stock DataFrame so @tool functions can access it."""
    global _stock_data
    _stock_data = data


def get_stock_data() -> Optional[pd.DataFrame]:
    """Retrieve the current stock DataFrame (set by the analyzer)."""
    return _stock_data


# ── BaseTool ─────────────────────────────────────────────────────────────────

class BaseTool(ABC):
    """Base class for all analysis tools."""

    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)


# ── ToolFactory ──────────────────────────────────────────────────────────────

class ToolFactory:
    """Dynamic loader that discovers tool classes and LangChain @tool
    functions from ``tools/*/``."""

    # tool_name -> (tool_class, category, langchain_tool_func_or_None)
    _registry: Optional[Dict[str, Tuple[type, str, Any]]] = None

    # ── scanning ─────────────────────────────────────────────────────────

    @classmethod
    def _scan(cls) -> Dict[str, Tuple[type, str, Any]]:
        """Walk ``tools/*/`` and import modules that export ``TOOL_CLASS``
        and (optionally) a ``TOOL_FUNC`` LangChain tool."""
        registry: Dict[str, Tuple[type, str, Any]] = {}
        tools_root = Path(__file__).resolve().parents[1] / "tools"

        if not tools_root.is_dir():
            LOGGER.warning("tools/ directory not found at %s", tools_root)
            return registry

        for subdir in sorted(tools_root.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith((".", "__")):
                continue

            category = subdir.name

            for py_file in sorted(subdir.glob("*.py")):
                if py_file.name.startswith("__"):
                    continue

                module_name = f"tools.{category}.{py_file.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec is None or spec.loader is None:
                        continue
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[union-attr]
                except Exception as exc:
                    LOGGER.debug("Skipping %s: %s", py_file, exc)
                    continue

                tool_cls = getattr(mod, "TOOL_CLASS", None)
                if tool_cls is None:
                    continue

                tool_func = getattr(mod, "TOOL_FUNC", None)

                try:
                    instance = tool_cls()
                    registry[instance.name] = (tool_cls, category, tool_func)
                except Exception as exc:
                    LOGGER.warning("Failed to instantiate %s from %s: %s",
                                   tool_cls, py_file, exc)

        LOGGER.info("ToolFactory: discovered %d tools from %s", len(registry), tools_root)
        return registry

    @classmethod
    def _ensure_loaded(cls) -> Dict[str, Tuple[type, str, Any]]:
        if cls._registry is None:
            cls._registry = cls._scan()
        return cls._registry

    # ── public API ───────────────────────────────────────────────────────

    @classmethod
    def create_tool(cls, tool_name: str) -> ITool:
        """Create a tool instance by name."""
        registry = cls._ensure_loaded()
        if tool_name not in registry:
            raise ValueError(f"Unknown tool: {tool_name}")
        tool_cls, _, _ = registry[tool_name]
        return tool_cls()

    @classmethod
    def get_available_tools(cls) -> List[str]:
        """Get list of available tool names."""
        return list(cls._ensure_loaded().keys())

    @classmethod
    def create_all_tools(cls) -> Dict[str, ITool]:
        """Create instances of all available tools."""
        return {name: cls.create_tool(name) for name in cls._ensure_loaded()}

    # ── LangChain integration ────────────────────────────────────────────

    @classmethod
    def get_langchain_tools(
        cls,
        selected: Optional[List[str]] = None,
    ) -> List[Any]:
        """Return LangChain ``@tool`` functions for the given tool names.

        Parameters
        ----------
        selected:
            Tool names to include.  ``None`` means all discovered tools
            that export a ``TOOL_FUNC``.

        Returns
        -------
        list
            LangChain tool objects ready to pass to an agent or
            ``bind_tools()``.
        """
        registry = cls._ensure_loaded()
        tools: List[Any] = []

        names = selected if selected is not None else list(registry.keys())
        for name in names:
            entry = registry.get(name)
            if entry is None:
                LOGGER.warning("Tool %r not found in registry, skipping", name)
                continue
            _, _, tool_func = entry
            if tool_func is None:
                LOGGER.debug("Tool %r has no TOOL_FUNC, skipping", name)
                continue
            tools.append(tool_func)

        return tools

    # ── registry metadata (for /tools-registry endpoint) ─────────────────

    @classmethod
    def get_registry(cls) -> Dict[str, Any]:
        """Return category-grouped metadata for the ``/tools-registry``
        endpoint."""
        registry = cls._ensure_loaded()

        FRIENDLY: Dict[str, str] = {
            "buck_visualizer": "Buck Visualizer",
            "dl": "Deep Learning",
            "maths": "Technical Analysis",
            "ml": "Machine Learning",
            "utility": "Utility",
            "web": "Web",
        }

        grouped: Dict[str, list] = {}
        for tool_name, (tool_cls, category, _) in registry.items():
            grouped.setdefault(category, [])
            inst = tool_cls()
            grouped[category].append({
                "id": tool_name,
                "name": tool_name.replace("_", " ").title(),
                "description": inst.description,
            })

        categories = []
        for cat_id in sorted(grouped):
            cat_name = FRIENDLY.get(cat_id, cat_id.replace("_", " ").title())
            categories.append({
                "id": cat_id,
                "name": cat_name,
                "description": f"{cat_name} tools",
                "tools": grouped[cat_id],
            })

        return {"categories": categories}
