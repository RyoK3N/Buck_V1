#!/bin/bash
# Buck MCP Installer
# Run from inside Buck_V1:  bash install_mcp.sh
#
# Installs dependencies, bootstraps .env, verifies the MCP server, and registers
# the "buck" connector in Claude Desktop's config (merging into any existing one).

set -euo pipefail

BUCK_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$HOME/Library/Application Support/Claude"
CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"

echo "=== Buck MCP Installer ==="
echo "Buck dir: $BUCK_DIR"
echo ""

# 0. Pick a single Python interpreter and use it for everything (install,
#    verification, and the command written into Claude Desktop's config).
#
#    Priority:
#      1. An already-active virtualenv  (VIRTUAL_ENV)
#      2. An already-active *named* conda env (not base)
#      3. A project-local virtualenv at ./.venv (created if missing)
#
#    We avoid installing into a Homebrew / system / conda-base interpreter,
#    which is "externally managed" (PEP 668) and refuses `pip install`.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
  echo "→ Using active virtualenv: $VIRTUAL_ENV"
elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "base" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PYTHON="$CONDA_PREFIX/bin/python"
  echo "→ Using active conda env: ${CONDA_DEFAULT_ENV:-?} ($CONDA_PREFIX)"
else
  VENV_DIR="$BUCK_DIR/.venv"
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    BASE_PY="$(command -v python3 || true)"
    if [ -z "$BASE_PY" ]; then
      echo "✗ Could not find python3 to create a virtualenv. Install Python 3 and re-run." >&2
      exit 1
    fi
    echo "→ No dedicated environment active — creating project virtualenv at .venv"
    "$BASE_PY" -m venv "$VENV_DIR"
  else
    echo "→ Reusing existing project virtualenv: $VENV_DIR"
  fi
  PYTHON="$VENV_DIR/bin/python"
fi
# Resolve to an absolute path so Claude Desktop can launch it regardless of PATH.
PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"
echo "  Python: $PYTHON"
echo ""

# 1. Install Python dependencies (use the same interpreter, not a stray pip).
echo "→ Installing Python dependencies..."
"$PYTHON" -m pip install --upgrade pip --quiet
"$PYTHON" -m pip install -r "$BUCK_DIR/requirements.txt" --quiet
echo "  ✓ Dependencies installed"
echo ""

# 2. Bootstrap .env from the template if it does not exist yet.
if [ ! -f "$BUCK_DIR/.env" ]; then
  echo "→ No .env found — creating one from .env.example"
  cp "$BUCK_DIR/.env.example" "$BUCK_DIR/.env"
  echo "  ✓ Created $BUCK_DIR/.env"
fi
# Warn (but don't fail) if the required key is still blank.
if ! grep -Eq '^OPENAI_API_KEY=.+' "$BUCK_DIR/.env"; then
  echo "  ⚠ OPENAI_API_KEY is empty in .env — set it before using Buck."
fi
echo ""

# 3. Verify the MCP server imports with the SAME interpreter we'll register.
echo "→ Verifying MCP server..."
( cd "$BUCK_DIR" && "$PYTHON" -c "from mcp_server.server import mcp; print('  ✓ MCP server imports OK')" )
echo ""

# 4. Create/update Claude Desktop config (merge, preserving other servers).
echo "→ Writing Claude Desktop config..."
mkdir -p "$CONFIG_DIR"

if [ -f "$CONFIG_FILE" ]; then
  echo "  Existing config found. Backing up to claude_desktop_config.json.bak"
  cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
fi

"$PYTHON" - "$CONFIG_FILE" "$PYTHON" "$BUCK_DIR" << 'PYEOF'
import json, os, sys

config_path, python_path, buck_dir = sys.argv[1], sys.argv[2], sys.argv[3]

config = {}
if os.path.exists(config_path):
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, ValueError):
        print("  ⚠ Existing config was not valid JSON — starting fresh "
              "(a .bak backup was kept).")
        config = {}

if not isinstance(config, dict):
    config = {}

# Invoke the runner by its ABSOLUTE FILE PATH rather than `-m mcp_server.runner`.
# `-m` only works if the launcher's cwd is the repo root, and Claude Desktop does
# not reliably honour the `cwd` field — that caused "No module named 'mcp_server'".
# runner.py injects the repo root into sys.path itself, so a direct path works
# from any working directory.
runner = os.path.join(buck_dir, "mcp_server", "runner.py")

# The server reads API keys from the repo's .env (runner.py loads it via an
# absolute path), so we keep env empty.
config.setdefault("mcpServers", {})["buck"] = {
    "command": python_path,
    "args": [runner, "--transport", "stdio"],
    "cwd": buck_dir,
    "env": {},
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print("  ✓ Registered 'buck' MCP server in Claude Desktop config")
PYEOF

echo ""
echo "=== Done ==="
echo "Restart Claude Desktop to load the Buck MCP connector."
echo ""
echo "Config written to: $CONFIG_FILE"
cat "$CONFIG_FILE"
