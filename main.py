"""
Buck_V1 — main entry point
──────────────────────────
Starts the FastAPI backend and (optionally) the React frontend dev server.

Usage:
    python main.py                  # backend + frontend
    python main.py --backend-only   # backend only
    python main.py --port 9000      # custom backend port (default: 8000)
    python main.py --no-reload      # disable uvicorn auto-reload
"""

from __future__ import annotations
import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = REPO_ROOT / "UI" / "frontend"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Buck application")
    p.add_argument("--backend-only", action="store_true", help="Skip the frontend dev server")
    p.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    p.add_argument("--no-reload", action="store_true", help="Disable uvicorn --reload")
    return p.parse_args()


def start_frontend() -> subprocess.Popen | None:
    """Launch `npm run dev` in UI/frontend/. Returns None if npm is unavailable."""
    npm = "npm.cmd" if sys.platform == "win32" else "npm"

    # Install deps if node_modules is missing
    if not (FRONTEND_DIR / "node_modules").exists():
        print("[buck] node_modules not found — running npm install …")
        ret = subprocess.run([npm, "install"], cwd=FRONTEND_DIR)
        if ret.returncode != 0:
            print("[buck] npm install failed — skipping frontend", file=sys.stderr)
            return None

    print(f"[buck] Starting frontend dev server → http://localhost:5173")
    return subprocess.Popen(
        [npm, "run", "dev"],
        cwd=FRONTEND_DIR,
    )


def start_backend(port: int, reload: bool) -> subprocess.Popen:
    """Launch uvicorn serving UI.backend.main:app."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "UI.backend.main:app",
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")

    print(f"[buck] Starting backend API server → http://localhost:{port}")
    return subprocess.Popen(cmd, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()

    processes: list[subprocess.Popen] = []

    frontend_proc = None
    if not args.backend_only:
        frontend_proc = start_frontend()
        if frontend_proc:
            processes.append(frontend_proc)

    backend_proc = start_backend(port=args.port, reload=not args.no_reload)
    processes.append(backend_proc)

    print("[buck] Press Ctrl+C to stop all servers.\n")

    def shutdown(sig, frame):  # noqa: ANN001
        print("\n[buck] Shutting down …")
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait — exit if the backend dies unexpectedly
    try:
        backend_proc.wait()
    finally:
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
