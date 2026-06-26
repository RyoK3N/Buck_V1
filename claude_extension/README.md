# Buck — Claude Desktop Extension

Install Buck's MCP server into Claude Desktop as a one-click extension instead of
hand-editing `claude_desktop_config.json`.

> Buck has heavy dependencies (PyTorch, etc.), so the bundle does **not** ship a
> Python environment. You point it at the `.venv` the installer creates. Run
> `bash install_mcp.sh` from the repo root first to create `.venv` and install
> dependencies.

## Build the bundle

```bash
bash claude_extension/build.sh
```

This produces `claude_extension/buck.mcpb`. (A `.mcpb` — formerly `.dxt` — is just
a ZIP with `manifest.json` at its root.)

## Install in Claude Desktop

1. Open **Claude Desktop → Settings → Extensions**.
2. Choose **Install from file** (or drag `buck.mcpb` onto the window).
3. Fill in the two settings when prompted:
   - **Python interpreter** → `<Buck_V1>/.venv/bin/python`
   - **Buck repo directory** → `<Buck_V1>` (the folder with `mcp_server/` and `.env`)
4. Restart Claude Desktop. Buck's tools (`single_analyze`, `batch_analyze`,
   `rl_*`, `visualize`, accuracy tools) appear under the `buck` connector.

API keys are read from the repo's `.env`, so you don't enter secrets here.

## Troubleshooting

- **`No module named 'mcp_server'`** — the interpreter or repo path is wrong. The
  extension launches `<python> <buck_dir>/mcp_server/runner.py` by absolute path,
  so both settings must point at real locations.
- **Server disconnects immediately** — make sure dependencies are installed in the
  selected interpreter (`<Buck_V1>/.venv/bin/python -m pip install -r requirements.txt`).
