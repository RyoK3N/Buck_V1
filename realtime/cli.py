"""
realtime.cli
────────────
CLI entry point for the real-time intraday simulation.

Examples
--------
Live (only runs during market hours for the exchange):
    python -m realtime.cli --symbol BHEL.NS --model-id my_ppo --interval 1m

Replay (works any time — streams recent historical bars through the same loop):
    python -m realtime.cli --symbol BHEL.NS --model-id my_ppo --replay \\
        --replay-start 2024-01-02 --replay-end 2024-01-05 --interval 1d
"""

from __future__ import annotations

import argparse
import os

from .sim import IntradaySimulator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m realtime.cli",
        description="Real-time intraday RL simulation with online learning (Buck).",
    )
    p.add_argument("--symbol", required=True, help="Ticker, e.g. BHEL.NS")
    p.add_argument("--model-id", required=True, help="A trained ppo_continuous model id")
    p.add_argument("--interval", default="1m", help="Bar interval (default: 1m)")
    p.add_argument("--exchange", default=None, help="Exchange for market-hours gate (default: env MARKET_EXCHANGE/NSE)")
    p.add_argument("--poll-seconds", type=float, default=None, help="Seconds between live polls (default: env RT_POLL_SECONDS)")
    p.add_argument("--capital", type=float, default=100_000.0, help="Starting capital")
    p.add_argument("--online-update-every", type=int, default=None,
                   help="Run an online PPO update every N bars (default: env RT_ONLINE_UPDATE_EVERY)")
    p.add_argument("--max-steps", type=int, default=2000, help="Stop after this many observed bars")
    p.add_argument("--indian-api-key", default=None, help="Live-data API key (or env INDIAN_API_KEY)")
    # replay
    p.add_argument("--replay", action="store_true", help="Replay historical bars instead of polling live")
    p.add_argument("--replay-start", default=None, help="Replay start date YYYY-MM-DD")
    p.add_argument("--replay-end", default=None, help="Replay end date YYYY-MM-DD")
    return p


def run_from_args(args) -> dict:
    try:
        from agent_scripts.config import SETTINGS

        exchange = args.exchange or getattr(SETTINGS, "market_exchange", "NSE")
        poll_seconds = args.poll_seconds if args.poll_seconds is not None else getattr(SETTINGS, "rt_poll_seconds", 30.0)
        online_every = args.online_update_every if args.online_update_every is not None else getattr(SETTINGS, "rt_online_update_every", 4)
    except Exception:
        exchange = args.exchange or "NSE"
        poll_seconds = args.poll_seconds if args.poll_seconds is not None else 30.0
        online_every = args.online_update_every if args.online_update_every is not None else 4

    if args.replay and (not args.replay_start or not args.replay_end):
        raise SystemExit("--replay requires --replay-start and --replay-end")

    api_key = args.indian_api_key or os.environ.get("INDIAN_API_KEY", "")

    sim = IntradaySimulator(
        symbol=args.symbol,
        model_id=args.model_id,
        interval=args.interval,
        exchange=exchange,
        poll_seconds=poll_seconds,
        capital=args.capital,
        replay=args.replay,
        replay_start=args.replay_start,
        replay_end=args.replay_end,
        online_update_every=online_every,
        api_key=api_key,
        max_steps=args.max_steps,
    )
    return sim.run()


def main() -> None:
    args = build_parser().parse_args()
    result = run_from_args(args)
    print("\n── session summary ──")
    for k, v in result.items():
        if k == "last_update_stats":
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
