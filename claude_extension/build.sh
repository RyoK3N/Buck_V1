#!/bin/bash
# Build the Buck Claude Desktop extension bundle (buck.mcpb).
#
# A .mcpb (formerly .dxt) is just a ZIP with manifest.json at its root. Claude
# Desktop installs it via Settings → Extensions → "Install from file" (or by
# dragging the file onto that window).
#
# Usage:  bash claude_extension/build.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/buck.mcpb"

# Prefer the official packer if available (validates the manifest); else zip.
if command -v npx >/dev/null 2>&1 && npx --yes @anthropic-ai/mcpb --help >/dev/null 2>&1; then
  echo "→ Packing with @anthropic-ai/mcpb ..."
  ( cd "$HERE" && npx --yes @anthropic-ai/mcpb pack . "$OUT" )
else
  echo "→ mcpb CLI not available — building the bundle with zip"
  rm -f "$OUT"
  ( cd "$HERE" && zip -q -r "$OUT" manifest.json )
fi

echo "  ✓ Built: $OUT"
echo ""
echo "Install it in Claude Desktop:"
echo "  Settings → Extensions → Install from file → choose buck.mcpb"
echo "Then set:"
echo "  • Python interpreter → <Buck_V1>/.venv/bin/python"
echo "  • Buck repo directory → <Buck_V1>"
