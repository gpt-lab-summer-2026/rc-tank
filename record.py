#!/usr/bin/env python3
"""
Phase 1 — record camera frames paired with drive commands.

Every frame is saved next to the command being given at that
moment.

    python3 record.py                    drive with the keyboard
    python3 record.py --no-car           camera only, nothing moves
    python3 record.py --lock-exposure    fix AE/AWB after warmup
    python3 record.py --width 640 --height 480

Writes to  dataset/session_YYYYmmdd_HHMMSS/
    frames/000001.jpg ...
    log.csv        frame,timestamp,left,right,label
    meta.json      settings this session ran with

Needs car.py in the same directory.

    sudo apt install -y python3-picamera2 python3-opencv
    pip3 install pyserial

-------------------------------------------------------------
WHAT THIS IS FOR

Not training labels. Cloning drive commands does not work on
roaming data — roam.py's docstring explains why, and that
argument still holds.

These sessions exist so that perception can be replayed offline
against recorded frames, and so that a floor model has something
to be trained and measured on. Both want frames, not actions.

Which changes how you should drive. Do not chase class balance.
Cover ground instead: different rooms, different flooring,
daylight and lamplight, obstacles near and far. Then drive
slowly into things on purpose — nothing else in this project
produces a picture of an obstacle at the moment of contact, and
those frames are worth more than another lap.

Record at the size roam.py runs at, not the default. Its
morphology kernels are in pixels, so a half-size frame does not
behave the same, and replay stops matching what the car did.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import select
import sys
import termios
import threading
import time
import tty
from collections import Counter
from datetime import datetime
from pathlib import Path

from car import BridgeError, Car

# (left, right) -> label
LABELS = {
    (1, 1): "forward",
    (-1, -1): "backward",
    (1, 0): "arc_right",
    (0, 1): "arc_left",
    (1, -1): "spin_right",
    (-1, 1): "spin_left",
    (0, 0): "stop",
    (-1, 0): "rev_arc_right",
    (0, -1): "rev_arc_left",
}


def label_of(left: int, right: int) -> str:
    return LABELS.get((left, right), f"{left}_{right}")


def mix(steer: float, throttle: float, dz: float) -> tuple[int, int]:
    """Analogue stick to a discrete pair the relays can actually do."""
    base = 1 if throttle > dz else (-1 if throttle < -dz else 0)
    turn = 1 if steer > dz else (-1 if steer < -dz else 0)

    if base and turn:                       # arc: inner wheel stops
        return (base, 0) if turn > 0 else (0, base)
    if turn:                                # spin on the spot
        return (turn, -turn)
    return (base, base)


# ----------------------------------------------------------- camera


class Camera:
    def __init__(self, size=(640, 480), fps=30, lock_exposure=False):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise SystemExit(
                "picamera2 not found.\n"
                "  sudo apt install -y python3-picamera2\n"
                "In a venv, create it with --system-site-packages."
            )

        self.cam = Picamera2()

        cfg = {
            # picamera2's "RGB888" hands back a BGR array, which is
            # what OpenCV wants. Not a typo.
            "main": {"size": size, "format": "RGB888"},
            "controls": {"FrameRate": fps},
        }

        # Several sensor modes are cropped rather than binned — on the
        # IMX219 (camera v2) the 1080p and 640x480 modes read only part
        # of the array and narrow the field of view. Asking for a small
        # main size can land on one of those. Pinning the raw stream to
        # a full-array mode avoids losing FOV we cannot spare.
        mode = self._full_fov_mode()
        if mode:
            cfg["raw"] = {"size": mode}
            print(f"sensor mode {mode[0]}x{mode[1]} (full field of view)")

        self.cam.configure(self.cam.create_video_configuration(**cfg))
        self.cam.start()
        time.sleep(2.0)                     # let AE and AWB settle

        if lock_exposure:
            md = self.cam.capture_metadata()
            self.cam.set_controls(
                {
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ExposureTime": md["ExposureTime"],
                    "AnalogueGain": md["AnalogueGain"],
                    "ColourGains": md["ColourGains"],
                }
            )
            print(
                f"exposure locked at {md['ExposureTime']}us "
                f"gain {md['AnalogueGain']:.2f}"
            )

    def _full_fov_mode(self):
        """Smallest sensor mode that still reads the whole array.

        crop_limits gives the sensor region a mode covers. Modes that
        read the full width see the full field of view; the rest are
        crops. Among the full ones, the smallest is the fastest.
        """
        modes = getattr(self.cam, "sensor_modes", None)
        if not modes:
            return None
        try:
            widest = max(m["crop_limits"][2] for m in modes)
            full = [m for m in modes if m["crop_limits"][2] == widest]
            return min(full, key=lambda m: m["size"][0] * m["size"][1])["size"]
        except (KeyError, TypeError, ValueError):
            return None

    def frame(self):
        return self.cam.capture_array()

    def close(self):
        self.cam.stop()


# ------------------------------------------------------------ input


class KeyboardInput:
    """Latched keys. Works with no extra hardware, produces worse data."""

    KEYS = {
        "w": (1, 1),
        "s": (-1, -1),
        "a": (-1, 1),
        "d": (1, -1),
        "q": (0, 1),
        "e": (1, 0),
        " ": (0, 0),
    }
    HELP = "w/s drive  a/d spin  q/e arc  space stop  x quit"

    def __init__(self):
        self.left = 0
        self.right = 0
        self.quit = False
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def poll(self) -> None:
        while select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1).lower()
            if key == "x":
                self.quit = True
            elif key in self.KEYS:
                self.left, self.right = self.KEYS[key]

    def close(self) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)


class GamepadInput:
    """Analogue sticks via evdev. Left stick steers and throttles."""

    HELP = "left stick to drive  any button stops  ctrl-c quits"

    def __init__(self, deadzone: float = 0.35):
        from evdev import InputDevice, ecodes, list_devices

        self.ecodes = ecodes
        self.dev = None
        for path in list_devices():
            dev = InputDevice(path)
            caps = dev.capabilities()
            if ecodes.EV_ABS in caps and ecodes.EV_KEY in caps:
                axes = [c for c, _ in caps[ecodes.EV_ABS]]
                if ecodes.ABS_X in axes and ecodes.ABS_Y in axes:
                    self.dev = dev
                    break
        if self.dev is None:
            raise RuntimeError("no gamepad found")

        self.dev.grab()
        self._range = {}
        for code in (ecodes.ABS_X, ecodes.ABS_Y):
            info = self.dev.absinfo(code)
            self._range[code] = (info.min, info.max)

        self.deadzone = deadzone
        self.steer = 0.0
        self.throttle = 0.0
        self.left = 0
        self.right = 0
        self.quit = False

    def _norm(self, code: int, value: int) -> float:
        lo, hi = self._range[code]
        mid = (lo + hi) / 2.0
        span = (hi - lo) / 2.0
        return 0.0 if span == 0 else (value - mid) / span

    def poll(self) -> None:
        ec = self.ecodes
        try:
            # read_one, not read: read() hands back a generator and
            # never returns None, so iter()'s sentinel never matches
            # and the loop yields generators instead of events.
            for event in iter(self.dev.read_one, None):
                if event.type == ec.EV_ABS:
                    if event.code == ec.ABS_X:
                        self.steer = self._norm(ec.ABS_X, event.value)
                    elif event.code == ec.ABS_Y:
                        # sticks report negative when pushed up
                        self.throttle = -self._norm(ec.ABS_Y, event.value)
                elif event.type == ec.EV_KEY and event.value == 1:
                    self.steer = self.throttle = 0.0     # any button stops
        except OSError:
            pass                          # nothing pending, or it unplugged

        self.left, self.right = mix(self.steer, self.throttle, self.deadzone)

    def close(self) -> None:
        try:
            self.dev.ungrab()
        except Exception:
            pass


# --------------------------------------------------------- recorder


class Recorder:
    """Frames go to a queue; a worker thread does the encoding."""

    def __init__(self, root: Path, save_size, quality: int = 85):
        import cv2

        self.cv2 = cv2
        self.save_size = save_size
        self.quality = quality

        self.frames_dir = root / "frames"
        self.frames_dir.mkdir(parents=True)
        self.csv_path = root / "log.csv"

        self.count = 0
        self.dropped = 0
        self.stats: Counter[str] = Counter()

        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def submit(self, frame, ts: float, left: int, right: int) -> None:
        label = label_of(left, right)
        self.count += 1
        self.stats[label] += 1
        try:
            self._q.put_nowait((self.count, frame, ts, left, right, label))
        except queue.Full:
            # Better to lose a frame than to stall the control loop.
            self.dropped += 1
            self.count -= 1
            self.stats[label] -= 1

    def _writer(self) -> None:
        with open(self.csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "timestamp", "left", "right", "label"])

            while not (self._stop.is_set() and self._q.empty()):
                try:
                    n, frame, ts, left, right, label = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue

                name = f"{n:06d}.jpg"
                small = self.cv2.resize(frame, self.save_size)
                self.cv2.imwrite(
                    str(self.frames_dir / name),
                    small,
                    [int(self.cv2.IMWRITE_JPEG_QUALITY), self.quality],
                )
                w.writerow([name, f"{ts:.3f}", left, right, label])
                fh.flush()

    def balance(self) -> str:
        if not self.count:
            return "no frames yet"
        parts = []
        for label, n in self.stats.most_common():
            parts.append(f"{label} {100 * n / self.count:.0f}%")
        return "  ".join(parts)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)


# -------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset", help="dataset root directory")
    ap.add_argument("--port", default=None, help="ESP32 serial device")
    ap.add_argument("--fps", type=float, default=12.0, help="frames recorded per second")
    ap.add_argument("--width", type=int, default=320, help="saved frame width")
    ap.add_argument("--height", type=int, default=240, help="saved frame height")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality")
    ap.add_argument("--gamepad", action="store_true",
                    help="use an analogue gamepad instead of the keyboard "
                         "(needs evdev; not used by default)")
    ap.add_argument("--no-car", action="store_true", help="do not drive, camera only")
    ap.add_argument("--lock-exposure", action="store_true", help="fix AE/AWB after warmup")
    ap.add_argument(
        "--idle-skip",
        type=float,
        default=1.5,
        help="stop recording after this many seconds stationary (0 disables)",
    )
    args = ap.parse_args()

    # Input first — if the input source is going to fail we want to
    # know before spinning up the camera and opening the serial port.
    source = None
    if args.gamepad:
        try:
            source = GamepadInput()
            print("using gamepad")
        except Exception as e:
            print(f"no gamepad ({e}), falling back to keyboard")
    if source is None:
        source = KeyboardInput()
        print("using keyboard")

    car = None
    if not args.no_car:
        try:
            car = Car(port=args.port)
            print(f"bridge on {car.port}")
        except BridgeError as e:
            source.close()
            print(f"could not open the bridge: {e}", file=sys.stderr)
            return 1

    # KeyboardInput has already put the terminal in cbreak mode, and
    # Camera bails with SystemExit when picamera2 is missing. Landing
    # in the shell with cbreak still set leaves it unusable.
    try:
        cam = Camera(lock_exposure=args.lock_exposure)
    except BaseException:
        source.close()
        if car is not None:
            car.close()
        raise

    session = Path(args.out) / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    rec = Recorder(session, (args.width, args.height), args.quality)

    with open(session / "meta.json", "w") as fh:
        json.dump(
            {
                "started": datetime.now().isoformat(timespec="seconds"),
                "input": type(source).__name__,
                "record_fps": args.fps,
                "save_size": [args.width, args.height],
                "jpeg_quality": args.quality,
                "lock_exposure": args.lock_exposure,
                "idle_skip_s": args.idle_skip,
            },
            fh,
            indent=2,
        )

    print(f"\nrecording to {session}")
    print(source.HELP)
    print()

    period = 1.0 / args.fps
    next_frame = time.monotonic()
    last_summary = time.monotonic()
    last_moving = time.monotonic()
    sent = (None, None)
    t0 = time.monotonic()

    try:
        while not getattr(source, "quit", False):
            source.poll()
            left, right = source.left, source.right

            # Only talk to the bridge when something changes. Relays
            # have a limited number of operations in them.
            if car is not None and (left, right) != sent:
                try:
                    car.drive(left, right)
                    sent = (left, right)
                except BridgeError as e:
                    print(f"\rbridge error: {e}")

            now = time.monotonic()
            if (left, right) != (0, 0):
                last_moving = now

            if now >= next_frame:
                next_frame += period
                if next_frame < now:            # fell behind, resync
                    next_frame = now + period

                idle = args.idle_skip > 0 and (now - last_moving) > args.idle_skip
                if not idle:
                    rec.submit(cam.frame(), time.time(), left, right)

                elapsed = now - t0
                print(
                    f"\r{rec.count:6d} frames  "
                    f"{rec.count / elapsed:5.1f}/s  "
                    f"{label_of(left, right):<14}"
                    f"{'  IDLE' if idle else '      '}",
                    end="",
                )
                sys.stdout.flush()

            if now - last_summary > 5.0:
                last_summary = now
                print(f"\n  balance: {rec.balance()}")

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        if car is not None:
            car.close()
        cam.close()
        rec.close()

        print("\n")
        print(f"saved {rec.count} frames to {session}")
        if rec.dropped:
            print(f"dropped {rec.dropped} frames — lower --fps or --quality")
        print(f"balance: {rec.balance()}")
        print()
        print("Balance does not matter here — these are not action labels.")
        print("Coverage does: another room, another floor, another light,")
        print("and some slow deliberate contacts.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
