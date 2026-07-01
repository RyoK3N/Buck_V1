"""TimeLLM tool — NOT IMPLEMENTED.

This file is a design placeholder, not a working tool. It exports no
`TOOL_CLASS`/`TOOL_FUNC`, so `ToolFactory` does not register it and it is
not invokable from the UI, CLI, or MCP server — unlike `lstm_prediction.py`
in this same directory, which is a real, working tool.

Planned design notes (kept for whoever picks this up):
  - Data engineering/preprocessing layer: cleaning, normalization, feature
    extraction for both historical and realtime-streaming input.
  - Feature building layer: technical-indicator features + temporal context
    for the TimeLLM backbone.
  - Compression layer: map input to a lower-dimensional representation to
    reduce backbone compute, with a corresponding decompression/mapping-back
    layer for interpretability.
  - Warmup/initialization, training/fine-tuning, and validation/testing
    layers for the backbone itself.
  - RL integration layer: expose predictions/updates to tools/rl's agents.
  - A2A policy layer: let Claude drive the model interactively via MCP.
  - Simulation/replay layer: backtest predictions over historical scenarios.

See CONTRIBUTING.md for the tool contract (`TOOL_CLASS` + `TOOL_FUNC`) a
working implementation needs to follow.
"""
