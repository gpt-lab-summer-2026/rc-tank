#!/usr/bin/env python3
"""
Find out why the camera mast servo is buzzing.

    python3 servocheck.py
    python3 servocheck.py --sweep     walk the whole range, find the quiet band

Nothing drives. The tank does not move.

-------------------------------------------------------------
WHY A SCRIPT AND NOT AN ANSWER

A servo has no feedback wire. Nothing in this project can tell a
servo happily holding a position from one stalled against a stop
tearing its own gears — both look like "commanded 125, said ok".
The only instrument available is your ears, so this drives the
servo into the few states that tell the causes apart and asks.

WHAT THE ANSWERS MEAN

  Buzzes even when RELEASED
      It is not holding anything, so nothing mechanical is being
      fought. The signal line is picking up noise, or the pin was
      left floating when the pulse train stopped. The firmware
      parks GPIO 13 low on release — if you have not reflashed
      since that change, this is expected and a flash fixes it.

  Quiet released, buzzes at EVERY held angle
      The servo cannot hold anything steady. Supply or signal:
      a sagging rail, a thin shared ground, or a signal wire run
      alongside the relay harness. Not a calibration problem —
      no angle will be quiet.

  Quiet released and in the middle, buzzes at BOTH ends
      Both endpoints are outside what the linkage can actually
      reach, so the servo is jamming at each one and hunting.
      This is the common one, and it is a calibration fix, not a
      hardware fault: --sweep finds the real travel.

  Quiet everywhere except one end
      That one constant is wrong. Same fix, one number.
"""

from __future__ import annotations

import argparse
import sys
import time

from car import BridgeError, Car


def ask(question: str) -> bool:
    try:
        return input(f"    {question} [y/N] ").strip().lower().startswith("y")
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt


def hold(car: Car, angle: int, seconds: float, what: str) -> bool:
    """Put the servo somewhere, leave it there, and ask."""
    print(f"\n  {what}")
    car.mast(angle)
    time.sleep(seconds)
    return ask("buzzing, humming or twitching?")


def sweep(car: Car, step: int, seconds: float) -> None:
    """Walk the range so the quiet band can be heard directly.

    Released between steps rather than driven from one angle to the
    next: a servo crossing its own stop on the way past would buzz
    at an angle that is otherwise fine, which is exactly the reading
    this is trying to get right.
    """
    print("\n  Walking 0 to 180. Note where it goes quiet and where it")
    print("  strains — the quiet band is the travel your linkage has.\n")
    quiet = []
    try:
        for angle in range(0, 181, step):
            car.mast(angle)
            time.sleep(seconds)
            ok = ask(f"{angle:>3} deg — QUIET?")
            if ok:
                quiet.append(angle)
    finally:
        car.mast(-1)

    print()
    if not quiet:
        print("  Nothing was quiet anywhere. That is not calibration —")
        print("  see the RELEASED and EVERY ANGLE cases in the docstring.")
        return

    lo, hi = min(quiet), max(quiet)
    print(f"  Quiet from {lo} to {hi} degrees.")
    print(f"  Anything outside that is the servo fighting a mechanical stop.\n")
    # Back off the edges: the outermost quiet step is the last one
    # tested, not necessarily the last one that is safe.
    pad = step
    print(f"  Suggested, backed off by one step so it rests rather than presses:")
    print(f"      MAST_DOWN = {min(lo + pad, hi)}")
    print(f"      MAST_UP   = {max(hi - pad, lo)}")
    print(f"\n  Both live in car.py. Neither needs a reflash.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="ESP32 serial device")
    ap.add_argument("--sweep", action="store_true",
                    help="walk the whole range to find the usable travel")
    ap.add_argument("--step", type=int, default=15, help="sweep step, degrees")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds to listen at each position")
    args = ap.parse_args()

    try:
        car = Car(port=args.port)
    except BridgeError as e:
        print(f"could not open the bridge: {e}", file=sys.stderr)
        return 1

    print(f"connected on {car.port}")
    print(f"  MAST_DOWN={Car.MAST_DOWN}  MAST_UP={Car.MAST_UP}")

    try:
        if args.sweep:
            sweep(car, max(5, args.step), args.settle)
            return 0

        print("\n  Four tests. Listen to the servo, not the terminal.")

        # Released first. Everything after this is compared against it,
        # because a servo that buzzes while holding nothing rules out
        # every mechanical explanation at once.
        print("\n  Releasing — no pulses at all.")
        car.mast(-1)
        time.sleep(args.settle)
        released = ask("buzzing, humming or twitching?")

        mid = hold(car, 90, args.settle, "Holding 90, mid travel.")
        low = hold(car, Car.MAST_DOWN, args.settle,
                   f"Holding {Car.MAST_DOWN}, the lowered position.")
        high = hold(car, Car.MAST_UP, args.settle,
                    f"Holding {Car.MAST_UP}, the raised position.")

        car.mast(-1)

        print("\n" + "-" * 58)
        if released:
            print("  Buzzes while RELEASED, holding nothing.")
            print("  Not mechanical and not calibration. The signal line is")
            print("  floating or picking up noise. Reflash first — the")
            print("  firmware parks GPIO 13 low on release, and without that")
            print("  the pin is left wherever LEDC dropped it. If it still")
            print("  buzzes after a flash, the wire is the problem: keep it")
            print("  away from the relay harness and check the ground return.")
        elif mid and low and high:
            print("  Quiet released, buzzes at EVERY held angle.")
            print("  The servo cannot hold anything steady, so no angle will")
            print("  fix it. Supply or ground: measure 5V at the servo while")
            print("  it holds, not at the bank. A thin shared ground counts.")
        elif low and high and not mid:
            print("  Quiet in the middle, buzzes at BOTH ends.")
            print("  Both endpoints are outside the travel the linkage has,")
            print("  so it jams at each. Calibration, not a fault.")
            print("  Run:  python3 servocheck.py --sweep")
        elif low or high:
            which = "MAST_DOWN" if low else "MAST_UP"
            print(f"  Quiet except at one end. {which} is past the stop.")
            print("  Run:  python3 servocheck.py --sweep")
        else:
            print("  Quiet everywhere it was asked to hold.")
            print("  Whatever you heard was not a held position — try again")
            print("  while the tank drives, since relay switching on the")
            print("  same rail is then the remaining suspect.")
        print("-" * 58)
        return 0

    except KeyboardInterrupt:
        print("\n  aborted")
        return 1
    finally:
        try:
            car.mast(-1)          # never leave it driving
        except Exception:
            pass
        car.close()


if __name__ == "__main__":
    sys.exit(main())
