"""
run_web.py — Launch the MonoArm web dashboard
==============================================

Starts the FastAPI application that serves the browser dashboard, processes
live camera frames, and streams the resulting joint angles to Unity over UDP.

Usage
-----
    python scripts/run_web.py
    python scripts/run_web.py --host 0.0.0.0 --port 8000
    python scripts/run_web.py --udp-port 9000 --filter kalman
    python scripts/run_web.py --no-udp          # dashboard only, no Unity stream

Then open http://127.0.0.1:8000 in a browser and allow camera access.

The page captures the webcam, sends frames to this server, and displays the
estimated joint angles, the derivation behind them, live filter comparisons
and the exact UDP packets being delivered to the Unity avatar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package root (python/) importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="MonoArm web dashboard")
    p.add_argument("--host", default="127.0.0.1",
                   help="Interface to bind (use 0.0.0.0 to allow other devices)")
    p.add_argument("--port", type=int, default=8000, help="HTTP port")
    p.add_argument("--udp-host", default=Config.UDP_IP,
                   help="Destination host for the Unity angle stream")
    p.add_argument("--udp-port", type=int, default=Config.UDP_PORT,
                   help="Destination UDP port for the Unity angle stream")
    p.add_argument("--no-udp", action="store_true",
                   help="Do not transmit to Unity (dashboard only)")
    p.add_argument("--filter", default=Config.DEFAULT_FILTER_TYPE,
                   choices=["kalman", "ma", "sg"], help="Initial temporal filter")
    p.add_argument("--hz", type=float, default=Config.STREAM_HZ,
                   help="UDP transmission rate in Hz")
    p.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = p.parse_args()

    import uvicorn

    from webapp import server as server_module

    pipe = server_module.state.pipeline
    pipe.set_filter(args.filter)
    pipe.set_udp(enabled=not args.no_udp, host=args.udp_host,
                 port=args.udp_port, hz=args.hz)

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print("=" * 70)
    print("  MonoArm — monocular arm joint angle estimation")
    print("=" * 70)
    print(f"  Dashboard      : {url}")
    print(f"  Unity stream   : "
          f"{'disabled' if args.no_udp else f'UDP {args.udp_host}:{args.udp_port} @ {args.hz:g} Hz'}")
    print(f"  Initial filter : {args.filter}")
    print("  Open the dashboard and allow camera access to begin.")
    print("=" * 70)

    uvicorn.run(
        "webapp.server:app" if args.reload else server_module.app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
