"""Control the live-submission bot: arm / disarm / status.

  python tools/control.py status      # is it armed? scheduler loaded? recent sweeps?
  python tools/control.py arm         # allow autonomous submission (kill switch off)
  python tools/control.py disarm      # INSTANT stop — daemon keeps running but submits nothing

Disarming is the safe kill switch: the scheduler keeps sweeping (and logging what
it WOULD do) but sends nothing. To stop the scheduler entirely, unload the
LaunchAgent (see deploy/com.jumpcworldcup.livesubmit.plist).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTO_CFG = ROOT / "config" / "auto_trade.json"
LOG = ROOT / "data" / "live_submit.log"
LABEL = "com.jumpcworldcup.livesubmit"


def set_armed(on: bool) -> None:
    cfg = json.loads(AUTO_CFG.read_text())
    cfg["armed"] = on
    AUTO_CFG.write_text(json.dumps(cfg, indent=1))
    print(f"armed = {on}" + ("  (bot will submit on the next sweep)" if on
                             else "  (KILL SWITCH ON — no submissions)"))


def status() -> None:
    cfg = json.loads(AUTO_CFG.read_text())
    print(f"armed: {cfg.get('armed')}")
    try:
        out = subprocess.run(["launchctl", "list", LABEL], capture_output=True,
                             text=True, timeout=5)
        print("scheduler loaded:", "yes" if out.returncode == 0 else "no")
    except Exception:                            # noqa: BLE001
        print("scheduler loaded: unknown")
    if LOG.exists():
        lines = [l for l in LOG.read_text().splitlines() if "=== sweep" in l
                 or "===" in l and ("submit" in l or "FAILED" in l)]
        print("recent sweeps:")
        for l in lines[-4:]:
            print("  " + l)
    else:
        print("no sweeps logged yet")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("status", "arm", "disarm"):
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "arm":
        set_armed(True)
    elif cmd == "disarm":
        set_armed(False)
    else:
        status()


if __name__ == "__main__":
    main()
