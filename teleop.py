#!/usr/bin/env python3
"""
Phase 0 — drive the car from the keyboard.

Proves the serial link, the protocol, the dead-time and the
watchdog all work before any camera or model is involved.

    python3 teleop.py
    python3 teleop.py --port /dev/ttyUSB0

Keep car.py in the same directory.

A terminal cannot see key releases, so commands latch: press a key
and the car keeps doing that until you press another one or space.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

from car import Car, BridgeError

# key -> (label, (left, right))
KEYS = {
    "w": ("forward",    (1, 1)),
    "s": ("backward",   (-1, -1)),
    "a": ("spin left",  (-1, 1)),
    "d": ("spin right", (1, -1)),
    "q": ("arc left",   (0, 1)),
    "e": ("arc right",  (1, 0)),
    " ": ("stop",       (0, 0)),
}

HELP = """
  w  forward        s  backward
  a  spin left      d  spin right
  q  arc left       e  arc right

  space  stop       x  quit
  i      show firmware config

Commands latch until you press another key.
"""


def read_key(timeout: float = 0.1) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="serial device")
    args = ap.parse_args()

    try:
        car = Car(port=args.port)
    except BridgeError as e:
        print(f"could not open the bridge: {e}", file=sys.stderr)
        return 1

    print(f"connected on {car.port}")
    print(car.info())
    print(HELP)

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    label = "stop"

    try:
        tty.setcbreak(fd)
        while True:
            key = read_key()
            if key is None:
                continue

            key = key.lower()
            if key == "x":
                break
            if key == "i":
                print("\r" + car.info())
                continue
            if key not in KEYS:
                continue

            label, (left, right) = KEYS[key]
            try:
                car.drive(left, right)
            except BridgeError as e:
                print(f"\rerror: {e}")
                continue

            # \r and padding keep the status on one line in cbreak mode
            print(f"\r{label:<12} left={left:+d} right={right:+d}   ", end="")
            sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        car.close()
        print("\nstopped, port closed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
