#!/usr/bin/env python3
"""
Why is the floor model calling clear floor an obstacle?

    python3 floorcheck.py                  capture a frame and report
    python3 floorcheck.py --save raw.png   also keep the frame, lossless
    python3 floorcheck.py --image raw.png  replay one saved earlier

Point the car at floor you believe is clear, exactly as you would to
start roam.py, and run it. Nothing moves.

-------------------------------------------------------------
THE ONE NUMBER THAT MATTERS

"patch matches itself". The model is built from the pixels inside the
cyan box and from nothing else, so those pixels are the easiest test
it will ever be given. If they do not come back as floor, the failure
is in the model and no amount of pointing it at better ground will
help. If they do, but the rest of the frame does not, the failure is
that the rest of the floor does not look like the patch — a different
problem with different fixes, listed at the bottom of the output.

Save the frame with --save and keep it. A raw frame is the only
artefact worth having here; the annotated debug image has the green
overlay blended into its pixels and cannot be measured.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from roam import FloorModel, free_profile, regions


def describe(name, hsv):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    print(f"  {name:<14} H {h.mean():6.1f} +/-{h.std():5.1f}   "
          f"S {s.mean():6.1f} +/-{s.std():5.1f}   "
          f"V {v.mean():6.1f} +/-{v.std():5.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="replay a saved frame instead of capturing")
    ap.add_argument("--save", default=None, help="write the raw frame here (use .png)")
    ap.add_argument("--rotate", type=int, default=None, choices=(0, 180))
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"could not read {args.image}", file=sys.stderr)
            return 1
        print(f"replaying {args.image}")
    else:
        from record import CAMERA_ROTATION, Camera
        rot = CAMERA_ROTATION if args.rotate is None else args.rotate
        cam = Camera(lock_exposure=True, rotate=rot)
        frame = cam.frame()
        cam.close()

    if args.save:
        # PNG, not JPEG. Chroma subsampling in a JPEG moves hue and
        # saturation around, which is exactly what is being measured.
        cv2.imwrite(args.save, frame)
        print(f"raw frame written to {args.save}")

    h, w = frame.shape[:2]
    x0, y0, x1, y1 = FloorModel.patch_box(frame.shape)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    print(f"\nframe {w}x{h}   patch x {x0}-{x1}  y {y0}-{y1}\n")
    print("colour, mean and spread per channel:")
    describe("whole frame", hsv)
    describe("patch", hsv[y0:y1, x0:x1])

    sat = hsv[:, :, 1].mean()
    print()
    if sat < 30:
        print(f"  NOTE: mean saturation {sat:.0f} is very low — this surface is")
        print("  close to grey. Hue is numerically unstable on grey pixels, so")
        print("  the H,S histogram has little real signal to work with.")

    model = FloorModel()
    model.learn(frame)
    hist = model.hist
    occupied = int((hist > 0).sum())
    total = hist.size
    vals = hist[hist > 0]

    print(f"\nhistogram: {occupied}/{total} bins populated")
    print(f"  values after normalise:  min {vals.min():5.1f}   "
          f"median {np.median(vals):5.1f}   max {vals.max():5.1f}")
    for t in (40, 10, 2):
        print(f"    bins above threshold {t:>2}: {int((hist > t).sum())}")

    back = cv2.calcBackProject([hsv], [0, 1], hist, FloorModel.RANGES, 1)
    print(f"\nback-projection before blur:")
    print(f"  patch       min {back[y0:y1, x0:x1].min():3d}  "
          f"mean {back[y0:y1, x0:x1].mean():6.1f}  max {back[y0:y1, x0:x1].max():3d}")
    print(f"  whole frame min {back.min():3d}  mean {back.mean():6.1f}  max {back.max():3d}")
    print(f"  fraction of frame that is exactly 0 (colour never seen): "
          f"{(back == 0).mean()*100:.1f}%")

    print(f"\n{'thresh':>6} {'%frame floor':>13} {'patch matches itself':>21} "
          f"{'L':>6} {'C':>6} {'R':>6}")
    for t in (40, 20, 10, 5, 2):
        m = model.mask(frame, t)
        L, C, R = regions(free_profile(m))
        print(f"{t:>6} {(m>0).mean()*100:>12.1f}% {(m[y0:y1, x0:x1]>0).mean()*100:>20.1f}% "
              f"{L:>6.0f} {C:>6.0f} {R:>6.0f}")

    print(f"\n  go> is {0.45*h:.0f} at the default --go 0.45")

    # ---------------------------------------------------------- verdict
    m40 = model.mask(frame, 40)
    self40 = (m40[y0:y1, x0:x1] > 0).mean()
    print("\nverdict:")
    if self40 < 0.8:
        print("  The patch does not match itself. The model is the problem, not")
        print("  the ground. Most likely the patch spans more than one surface —")
        print("  check the saved frame for the car's own chassis, a shadow edge,")
        print("  or a bright streak inside the cyan box — which splits the")
        print("  histogram so that no single colour dominates.")
    elif (m40 > 0).mean() < 0.5:
        print("  The patch matches itself but the rest of the floor does not, so")
        print("  the floor is not one colour to this model. On a grey surface")
        print("  that usually means hue is noise and H,S cannot separate floor")
        print("  from anything else. Value would carry the signal here, and it")
        print("  is deliberately excluded — see FloorModel's docstring.")
    else:
        print("  Perception looks reasonable on this frame. If roam.py still")
        print("  refuses to drive, compare C above against go> — the clearance")
        print("  needed may simply be more than this camera pitch can ever see.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
