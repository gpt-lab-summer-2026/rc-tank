#!/usr/bin/env python3
"""
Phase 2 — roam and avoid obstacles, with no training data.

Learns what the floor looks like from the patch directly in front
of the car, then treats anything that does not match as an
obstacle. Steers toward whichever side has the most free floor.

    python3 roam.py --dry-run          decide but do not move
    python3 roam.py                    drive
    python3 roam.py --debug-image /tmp/roam.jpg

Keep car.py and record.py in the same directory.

-------------------------------------------------------------
WHY NOT A TRAINED MODEL YET

Roaming has no single correct action — clear floor ahead makes
both "forward" and "turn" reasonable. Cloning actions from such
data averages the alternatives into "straight ahead", which
drives into things.

Learning perception sidesteps that. "Where is the floor" has one
right answer per pixel, and the steering policy below is code you
can read rather than weights that fail quietly.

-------------------------------------------------------------
START IT POINTING AT CLEAR FLOOR

The floor model is learned once at startup from a patch at the
bottom-centre of the frame. If the car is nose-to-wall when you
launch it, it learns that the wall is floor and will drive
straight into things. Give it a metre of clear ground.

Press the reference patch region with --show-patch to see exactly
what it samples.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque

import cv2
import numpy as np

from car import BridgeError, Car
from record import Camera, label_of

# ------------------------------------------------------- floor model


class FloorModel:
    """Hue/saturation histogram of the ground, back-projected per pixel.

    Value is deliberately excluded — brightness varies with shadow
    and vignetting, while hue and saturation stay comparatively
    stable across the same carpet or lino.
    """

    BINS = (24, 24)
    RANGES = (0, 180, 0, 256)

    def __init__(self, smooth: float = 0.0):
        self.hist = None
        self.smooth = smooth      # 0 = learn once and never change

    @staticmethod
    def patch_box(shape) -> tuple[int, int, int, int]:
        """The trapezoid of ground the camera sees right in front."""
        h, w = shape[:2]
        return (int(w * 0.35), int(h * 0.82), int(w * 0.65), h)

    def learn(self, bgr) -> None:
        x0, y0, x1, y1 = self.patch_box(bgr.shape)
        hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, self.BINS, self.RANGES)
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        if self.hist is None or self.smooth <= 0:
            self.hist = hist
        else:
            self.hist = (1 - self.smooth) * self.hist + self.smooth * hist

    def mask(self, bgr, threshold: int = 40):
        """255 where the pixel looks like floor."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        back = cv2.calcBackProject([hsv], [0, 1], self.hist, self.RANGES, 1)

        # Blur first so isolated speckles do not cut a column short,
        # then close small holes from carpet texture and specular dots.
        cv2.filter2D(back, -1, np.ones((5, 5), np.float32) / 25, back)
        _, m = cv2.threshold(back, threshold, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k)


def free_profile(mask) -> np.ndarray:
    """Unbroken floor height per column, counting up from the bottom.

    A column is only free as far as its first non-floor pixel — an
    obstacle blocks everything behind it regardless of what the
    floor does further up the frame.
    """
    floor = mask > 0
    flipped = floor[::-1]                     # bottom row first
    blocked = np.argmax(~flipped, axis=0)     # first non-floor going up
    blocked[np.all(flipped, axis=0)] = mask.shape[0]   # never blocked
    return blocked


def regions(profile: np.ndarray, pct: int = 25) -> tuple[float, float, float]:
    """Free height for left, centre and right thirds.

    A low percentile rather than the mean, so one thin chair leg
    still counts as blocking that side.
    """
    w = len(profile)
    third = w // 3
    return (
        float(np.percentile(profile[:third], pct)),
        float(np.percentile(profile[third : 2 * third], pct)),
        float(np.percentile(profile[2 * third :], pct)),
    )


# ------------------------------------------------------------ policy


class Policy:
    """Free space in, relay directions out."""

    def __init__(self, go: float, turn_margin: float, stuck_after: float):
        self.go = go                      # centre clearance to keep going
        self.turn_margin = turn_margin    # how much better a side must be
        self.stuck_after = stuck_after
        self._turning_since = None

    def decide(self, left, centre, right, now) -> tuple[int, int]:
        if centre >= self.go:
            self._turning_since = None
            return (1, 1)

        if self._turning_since is None:
            self._turning_since = now
        elif now - self._turning_since > self.stuck_after:
            # Pivoting has not found a way out. Reverse instead.
            return (-1, -1)

        # Blocked ahead: spin toward the side with more room. Ties and
        # near-ties break left so the car does not dither in a corner.
        if right > left + self.turn_margin:
            return (1, -1)
        return (-1, 1)


class Smoother:
    """Majority vote plus a floor on how often relays may change.

    Contacts are rated for a limited number of operations, and a
    single misclassified frame should not cost one.
    """

    def __init__(self, window: int, min_interval: float):
        self.buf: deque = deque(maxlen=window)
        self.min_interval = min_interval
        self.current = (0, 0)
        self.changed_at = 0.0

    def update(self, decision, now) -> tuple[int, int]:
        self.buf.append(decision)
        if len(self.buf) < self.buf.maxlen:
            return self.current

        winner, _ = Counter(self.buf).most_common(1)[0]
        if winner != self.current and now - self.changed_at >= self.min_interval:
            self.current = winner
            self.changed_at = now
        return self.current


# ------------------------------------------------------------- debug


def annotate(bgr, mask, prof, regs, decision):
    out = bgr.copy()
    green = np.zeros_like(out)
    green[:, :, 1] = mask
    out = cv2.addWeighted(out, 0.7, green, 0.3, 0)

    h, w = mask.shape
    for x in range(0, w, 4):
        y = h - int(prof[x])
        cv2.circle(out, (x, y), 1, (0, 0, 255), -1)

    x0, y0, x1, y1 = FloorModel.patch_box(bgr.shape)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 2)

    for i, v in enumerate(regs):
        cv2.putText(
            out, f"{v:.0f}", (int(w * (i / 3 + 0.12)), 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
    cv2.putText(
        out, label_of(*decision), (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    return out


# -------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="ESP32 serial device")
    ap.add_argument("--dry-run", action="store_true", help="decide but do not drive")
    ap.add_argument("--fps", type=float, default=10.0, help="decisions per second")
    ap.add_argument("--go", type=float, default=0.45,
                    help="centre clearance to keep going, as a fraction of frame height")
    ap.add_argument("--turn-margin", type=float, default=0.05,
                    help="how much clearer one side must be, as a fraction")
    ap.add_argument("--threshold", type=int, default=40, help="floor match strictness 0-255")
    ap.add_argument("--window", type=int, default=5, help="frames in the majority vote")
    ap.add_argument("--min-interval", type=float, default=0.2,
                    help="seconds between relay changes")
    ap.add_argument("--stuck-after", type=float, default=3.0,
                    help="seconds of turning before reversing")
    ap.add_argument("--adapt", type=float, default=0.0,
                    help="floor model blend rate, 0 learns once")
    ap.add_argument("--debug-image", default=None, help="write an annotated frame here")
    args = ap.parse_args()

    cam = Camera()
    car = None
    if not args.dry_run:
        try:
            car = Car(port=args.port)
            print(f"bridge on {car.port}")
        except BridgeError as e:
            print(f"could not open the bridge: {e}", file=sys.stderr)
            return 1

    print("\nlearning the floor — keep a metre of clear ground ahead")
    time.sleep(1.0)
    floor = FloorModel(smooth=args.adapt)
    floor.learn(cam.frame())
    print("learned\n")

    frame = cam.frame()
    h = frame.shape[0]
    go_px = args.go * h
    margin_px = args.turn_margin * h

    policy = Policy(go_px, margin_px, args.stuck_after)
    smoother = Smoother(args.window, args.min_interval)

    period = 1.0 / args.fps
    next_tick = time.monotonic()
    last_debug = 0.0
    sent = (None, None)

    try:
        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(0.005)
                continue
            next_tick = max(now + period, next_tick + period)

            frame = cam.frame()
            mask = floor.mask(frame, args.threshold)
            prof = free_profile(mask)
            regs = regions(prof)

            raw = policy.decide(*regs, now)
            decision = smoother.update(raw, now)

            if car is not None and decision != sent:
                try:
                    car.drive(*decision)
                    sent = decision
                except BridgeError as e:
                    print(f"\rbridge error: {e}")

            # Only adapt when the view ahead is clearly open, or the
            # model slowly learns that obstacles are floor.
            if args.adapt > 0 and regs[1] > go_px * 1.2:
                floor.learn(frame)

            print(
                f"\rL {regs[0]:5.0f}  C {regs[1]:5.0f}  R {regs[2]:5.0f}   "
                f"go>{go_px:.0f}   {label_of(*decision):<12}",
                end="",
            )
            sys.stdout.flush()

            if args.debug_image and now - last_debug > 1.0:
                last_debug = now
                cv2.imwrite(args.debug_image, annotate(frame, mask, prof, regs, decision))

    except KeyboardInterrupt:
        pass
    finally:
        if car is not None:
            car.close()
        cam.close()
        print("\nstopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())


# =============================================================
# TUNING
#
# Run with --dry-run --debug-image /tmp/roam.jpg first and look at
# the image. Green overlay is what it thinks is floor, red dots are
# where each column stops being free, the cyan box is the patch it
# learned from.
#
# Obstacles shown as floor
#   Raise --threshold. If a specific object is the same colour as
#   your carpet, hue and saturation cannot separate them and no
#   threshold will fix it — that is where the ToF sensor earns its
#   place.
#
# Floor shown as obstacle
#   Lower --threshold. Common on patterned rugs and near strong
#   shadow edges. More even lighting helps more than tuning does.
#
# Turns too late
#   Raise --go. It is a fraction of frame height, so 0.45 means the
#   centre must be free almost halfway up the frame to continue.
#
# Dithers in corners
#   Raise --turn-margin or --min-interval.
#
# Circles forever in open space
#   Lower --go, or check the camera is angled down far enough that
#   the far wall is not permanently in view.
#
# Drives into things the moment it starts
#   It learned the floor while pointing at an obstacle. Restart on
#   clear ground.
# =============================================================
