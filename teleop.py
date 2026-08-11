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
import os
import select
import sys
import termios
import tty

from car import Car, BridgeError, boot_warning

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


def read_keys(timeout: float = 0.1) -> str:
    """Every key pressed since the last call, in order.

    Reads the file descriptor rather than sys.stdin, because the two
    do not agree with select(). sys.stdin.read(1) pulls the whole
    pending chunk out of the OS and hands back one character, leaving
    the rest in a buffer select() cannot see — so the next select()
    reports nothing waiting and those keys sit unnoticed until
    another one arrives.

    It bites hardest exactly when it matters: if the bridge is slow,
    every command blocks for the read timeout, keystrokes pile up
    behind it, and the one that goes missing is the stop you just
    pressed twice.
    """
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return ""
    return os.read(sys.stdin.fileno(), 64).decode(errors="replace")


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

    warning = boot_warning(car.boot_reason)
    if warning:
        print(f"\n  !! {warning}\n")

    print(HELP)

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    quitting = False

    try:
        tty.setcbreak(fd)
        while not quitting:
            for key in read_keys().lower():
                if key == "x":
                    quitting = True
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
        resets = car.resets
        car.close()
        print("\nstopped, port closed")
        if resets:
            print(f"\n!! the bridge restarted {resets} time(s) during that session.")
            print("   Each restart is a window where the relays answered to")
            print("   nothing. Check the boot reason above and the 5V supply.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
