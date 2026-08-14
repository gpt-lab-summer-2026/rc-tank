#!/usr/bin/env python3
"""
Phase 0 — drive the car from the keyboard.

Proves the serial link, the protocol, the dead-time and the
watchdog all work before any camera or model is involved.

    python3 teleop.py
    python3 teleop.py --port /dev/ttyUSB0
    python3 teleop.py --selftest        click each relay (motors off!)

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
import time
import tty

from car import Car, BridgeError, boot_warning

# key -> (label, (left, right)).  Latching states.
KEYS = {
    "w": ("forward",   (1, 1)),
    "s": ("backward",  (-1, -1)),
    "q": ("arc left",  (0, 1)),
    "e": ("arc right", (1, 0)),
    " ": ("stop",      (0, 0)),
}

# key -> side.  Timed nudges, not states — see Car.soft_arc.
ARCS = {"a": "left", "d": "right"}

# Camera mast. t and g are the two presets; the nudges exist so the
# raised angle can be dialled in against the real linkage without
# editing car.py and restarting, which is otherwise a reflash-speed
# loop for a one-number question.
#
# v lets the servo go limp. That is the thing to reach for if holding
# the mast up is what starts browning out the 5V rail — a servo under
# load draws current for as long as it is asked to hold.
MAST_STEP = 5
MAST = {
    "t": "up",
    "g": "down",
    "v": "limp",
    "+": MAST_STEP,  "=": MAST_STEP,     # = so it works without shift
    "-": -MAST_STEP, "_": -MAST_STEP,
}


def mast_key(car: Car, action) -> str:
    """Apply one mast keypress and describe where the mast ended up."""
    if action == "up":
        # settle=0 because the operator is watching it move and does
        # not need the key loop to go deaf for most of a second.
        car.camera_up(settle=0.0)
    elif action == "down":
        car.camera_down()
    elif action == "limp":
        car.mast(-1)
    else:
        # Nudging from limp, or from never-commanded, starts at the
        # raised preset — dialling that angle in is the only reason to
        # nudge at all.
        base = car.mast_angle
        if base is None or base < 0:
            base = Car.MAST_UP
        car.mast(max(0, min(180, base + action)))

    angle = car.mast_angle
    if angle is None or angle < 0:
        return "mast limp (drawing nothing)"
    if angle in (Car.MAST_UP, Car.MAST_DOWN):
        return f"mast {angle} deg"
    # Off-preset means it is being calibrated, so say what to keep.
    return f"mast {angle} deg   (Car.MAST_UP is {Car.MAST_UP})"

HELP = """
  w  forward        s  backward

  q  arc left       e  arc right      held until you press something else
  a  soft arc left  d  soft arc right --arc-time seconds, then back

  space  stop       x  quit
  r      unstick    i  show config and relay operation counts

  t  mast up        g  mast down       v  mast limp
  +/-  nudge the mast 5 deg, to dial in the raised angle

Both arcs steer the same way — the inside track idles and the outside
one pulls the nose round. They differ only in who ends them.

  q / e   latch. The tank keeps turning until you press w, space, or
          something else. Use them to come round a long way.

  a / d   end themselves. One track drops out for --arc-time, then
          both go back to exactly what they were doing. Use them to
          correct a heading without giving up your course.

There is no spin. Opposing the tracks reverses a motor under load,
which is the harshest thing this drivetrain does to its contacts, and
an arc reaches the same heading without it.

Reversals wait --reverse-cooldown before they are allowed again;
arcs, stops and starts wait only --cooldown, nothing by default. A
press inside a window shows HELD. Space is never held.

r drives backwards briefly, then stops. That clears a relay that has
stayed engaged. Expect the rover to move.

relays[...] is what the bridge reports as APPLIED. Pressing space
should show relays[0 0 0 0]. If it does and the tracks keep turning,
try r. If that does not free it either, cut the motor power.
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


def relay_ops(info_line: str) -> list[int]:
    """Per-relay actuation counts out of an info reply."""
    for field in info_line.split():
        if field.startswith("ops="):
            try:
                return [int(x) for x in field[4:].split(",")]
            except ValueError:
                break
    return []


def selftest(car: Car) -> int:
    """Click each relay on its own, with a human as the sensor.

    Nothing reports back from the contacts, so the firmware cannot
    tell a relay that opened from one that welded shut. It can only
    energise them one at a time and leave you to listen. Four relays,
    four clicks, four releases. A relay that stays silent is stuck,
    and a stuck relay is what keeps a tank driving after it has been
    told to stop.
    """
    print("\n  DISCONNECT THE MOTOR BATTERY FIRST.")
    print("  This energises relays on purpose. With motors attached,")
    print("  the tank will move.\n")
    try:
        input("  Motors disconnected? Enter to run, ctrl-c to abort. ")
    except (KeyboardInterrupt, EOFError):
        print("\n  aborted")
        return 1

    before = relay_ops(car.info())
    print()

    for i in range(4):
        state = [0, 0, 0, 0]
        state[i] = 1
        print(f"  relay {i + 1} (IN{i + 1})   energise ...", end="", flush=True)
        car.relays(*state)
        time.sleep(0.6)
        print(" release ...", end="", flush=True)
        car.relays(0, 0, 0, 0)
        time.sleep(0.5)
        print(" done")

    after = relay_ops(car.info())
    print()
    if before and after and len(before) == len(after) == 4:
        print("  lifetime operations, per relay:")
        for i, (b, a) in enumerate(zip(before, after)):
            print(f"    IN{i + 1}  {a:6d}   (+{a - b} this test)")
        print()
        print("  Contacts are a consumable. Log these each session — welding")
        print("  gets likelier as they climb, and nothing warns you first.")
        print()

    print("  You should have heard EIGHT clicks: four energise, four release.")
    print("  A relay that never clicked, or clicked once and not again, is")
    print("  stuck. With the motor battery still disconnected, meter COM to")
    print("  NO on that relay — it must read open. Continuity means welded.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="serial device")
    ap.add_argument("--selftest", action="store_true",
                    help="click each relay in turn, then stop (motors off!)")
    ap.add_argument("--reverse-cooldown", type=float, default=0.5,
                    help="seconds before a motor may be reversed again; the "
                         "one change that fights the motor's own momentum")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds before any gentler change — arc, stop, "
                         "start — is allowed")
    ap.add_argument("--arc-time", type=float, default=1.0,
                    help="seconds to hold an arc before the tracks go back")
    ap.add_argument("--unstick-time", type=float, default=0.5,
                    help="seconds to drive backwards when r is pressed")
    args = ap.parse_args()

    try:
        car = Car(port=args.port,
                  command_cooldown=args.cooldown,
                  reverse_cooldown=args.reverse_cooldown)
    except BridgeError as e:
        print(f"could not open the bridge: {e}", file=sys.stderr)
        return 1

    print(f"connected on {car.port}")
    print(car.info())

    warning = boot_warning(car.boot_reason)
    if warning:
        print(f"\n  !! {warning}\n")

    if args.selftest:
        try:
            return selftest(car)
        finally:
            car.close()

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
                if key == "r":
                    print("\runsticking — reversing briefly ...", end="")
                    sys.stdout.flush()
                    try:
                        car.unstick(args.unstick_time)
                        print(" done, stopped        ")
                    except BridgeError as e:
                        print(f"\rerror: {e}")
                    continue
                if key in ARCS:
                    side = ARCS[key]
                    print(f"\rsoft arc {side:<6} inside track idling ...", end="")
                    sys.stdout.flush()
                    try:
                        if car.soft_arc(side, args.arc_time):
                            print(" back on course      ")
                        else:
                            print(f" HELD {car.cooldown_remaining():.1f}s      ")
                    except BridgeError as e:
                        print(f"\rerror: {e}")
                    continue
                if key in MAST:
                    try:
                        print(f"\r{mast_key(car, MAST[key]):<44}", end="")
                        sys.stdout.flush()
                    except BridgeError as e:
                        print(f"\rerror: {e}")
                    continue

                if key not in KEYS:
                    continue

                label, (left, right) = KEYS[key]
                try:
                    reply = car.drive(left, right)
                except BridgeError as e:
                    print(f"\rerror: {e}")
                    continue

                # Show what the bridge says is APPLIED, not what was
                # asked for. If this reads 0 0 0 0 while the tracks are
                # still turning, the firmware did as it was told and the
                # fault is a contact that will not open.
                applied = " ".join(reply.split()[1:5])

                if car.last_command_held:
                    # Say why nothing happened. A key that appears to do
                    # nothing is indistinguishable from a broken one.
                    status = f"HELD {car.cooldown_remaining():.1f}s   "
                else:
                    status = f"{label:<12} left={left:+d} right={right:+d}"

                # \r and padding keep the status on one line in cbreak mode
                print(f"\r{status}  relays[{applied}]      ", end="")
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
