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


def sweep(frame) -> int:
    """What each coarseness knob does to THIS frame.

    Clearances are printed as fractions of frame height so the rows
    stay comparable across --scale, and so they can be read straight
    against --go, which is the same units.
    """
    from roam import shrink

    def run(scale=1.0, bins=FloorModel.BINS, close=11, minobs=1, pct=25,
            thresh=40, flatten=0.0):
        im = shrink(frame, scale)
        h = im.shape[0]
        model = FloorModel(bins=bins, flatten=flatten)
        model.learn(im)
        m = model.mask(im, thresh, close)
        prof = free_profile(m, minobs)
        L, C, R = regions(prof, pct)
        return L / h, C / h, R / h, (m > 0).mean() * 100, model.chosen

    def row(label, **kw):
        L, C, R, floor, mode = run(**kw)
        flag = "  <- would drive" if C >= 0.45 else ""
        print(f"  {label:<24} L{L:5.2f} C{C:5.2f} R{R:5.2f}   floor {floor:5.1f}%{flag}")

    print("\nclearance as a fraction of frame height; --go default is 0.45\n")
    print("  baseline")
    row("as configured")
    print("\n  --scale   (averages texture away before the histogram sees it)")
    for v in (0.5, 0.33, 0.25):
        row(f"scale {v}", scale=v)
    print("\n  --bins    (coarser idea of the same colour)")
    for v in (16, 12, 8, 6):
        row(f"bins {v}", bins=v)
    print("\n  --percentile  (how much of a third must be blocked)")
    for v in (40, 50, 60):
        row(f"percentile {v}", pct=v)
    print("\n  --close   (fills gaps thinner than this)")
    for v in (15, 21, 31):
        row(f"close {v}", close=v)
    print("\n  --min-obstacle  (ignore blockers shorter than this)")
    for v in (10, 25, 40):
        row(f"min-obstacle {v}", minobs=v)
    print("\n  --flatten  (divides the brightness gradient out; the shadow knob)")
    for v in (30, 60, 120, 200):
        row(f"flatten {v}", flatten=v)
    print("\n  combined")
    row("scale .33 bins 8", scale=0.33, bins=8)
    row("scale .25 bins 8", scale=0.25, bins=8)
    row("scale .25 bins 8 pct 50", scale=0.25, bins=8, pct=50)
    row("scale .25 bins 8 flatten 60", scale=0.25, bins=8, flatten=60)

    print("\n  Pick the loosest row that still leaves a real obstacle showing.")
    print("  Put a box or a bag in view and run this again to check that.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="replay a saved frame instead of capturing")
    ap.add_argument("--save", default=None, help="write the raw frame here (use .png)")
    ap.add_argument("--rotate", type=int, default=None, choices=(0, 180))
    ap.add_argument("--port", default=None, help="ESP32 serial device")
    ap.add_argument("--no-servo", action="store_true",
                    help="do not open the bridge to raise the camera mast")
    ap.add_argument("--sweep", action="store_true",
                    help="try the coarseness knobs on this frame and print what "
                         "each does, instead of the channel diagnosis")
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

        # Only here to work the mast — nothing in this script drives.
        mast = None
        if not args.no_servo:
            try:
                from car import Car
                mast = Car(port=args.port)
            except Exception as e:
                print(f"  !! no bridge, so no mast: {e}", file=sys.stderr)

        cam = Camera(lock_exposure=True, rotate=rot, mast=mast)
        frame = cam.frame()
        cam.close()
        if mast is not None:
            mast.close()

    if args.save:
        # PNG, not JPEG. Chroma subsampling in a JPEG moves hue and
        # saturation around, which is exactly what is being measured.
        cv2.imwrite(args.save, frame)
        print(f"raw frame written to {args.save}")

    if args.sweep:
        return sweep(frame)

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

    # Both channel pairs, on the same frame, so the choice is measured
    # rather than argued about.
    results = {}
    for mode in ("hs", "sv"):
        model = FloorModel(mode=mode)
        model.learn(frame)
        hist = model.hist
        vals = hist[hist > 0]

        back = cv2.calcBackProject([hsv], model.channels, hist, model.ranges, 1)
        zeros = (back == 0).mean() * 100

        print(f"\n=== {mode.upper()} "
              f"({'hue+saturation, the original' if mode == 'hs' else 'saturation+value'}) ===")
        print(f"  {int((hist > 0).sum()):>3}/{hist.size} bins populated; "
              f"after normalise min {vals.min():.1f} median {np.median(vals):.1f} max {vals.max():.1f}")
        print(f"  back-projects to exactly 0 over {zeros:5.1f}% of the frame "
              f"(a colour the patch never held)")
        print(f"  {'thresh':>6} {'%frame floor':>13} {'patch self':>11} "
              f"{'L':>6} {'C':>6} {'R':>6}")
        for t in (40, 20, 10, 5, 2):
            m = model.mask(frame, t)
            L, C, R = regions(free_profile(m))
            if t == 40:
                results[mode] = ((m > 0).mean(), (m[y0:y1, x0:x1] > 0).mean(), C)
            print(f"  {t:>6} {(m>0).mean()*100:>12.1f}% {(m[y0:y1,x0:x1]>0).mean()*100:>10.1f}% "
                  f"{L:>6.0f} {C:>6.0f} {R:>6.0f}")

    print(f"\n  go> is {0.45*h:.0f} at the default --go 0.45")
    print(f"  roam.py would pick '{'hs' if sat >= FloorModel.ACHROMATIC_S else 'sv'}' "
          f"here on --channels auto (patch saturation {sat:.0f}, "
          f"cutoff {FloorModel.ACHROMATIC_S})")

    # ---------------------------------------------------------- verdict
    hs_frame, hs_self, _ = results["hs"]
    sv_frame, _, _ = results["sv"]
    print("\nverdict:")
    if hs_self < 0.8:
        print("  The patch does not match itself even on the frame it was built")
        print("  from. That is a broken model, not a hard scene — check the saved")
        print("  frame for the car's own chassis or a shadow edge inside the box.")
    elif hs_frame < 0.7 and sv_frame > hs_frame + 0.15:
        print(f"  H,S calls {(1-hs_frame)*100:.0f}% of this frame an obstacle; S,V calls")
        print(f"  {(1-sv_frame)*100:.0f}%. Hue is carrying noise on this floor and value is")
        print("  carrying the signal. Run roam.py with --channels sv, or leave it")
        print("  on auto, which picks the same thing from the saturation alone.")
    else:
        print("  Both channel pairs behave similarly here, so the channel choice")
        print("  is not what is stopping it. Compare C above against go>: the")
        print("  clearance being demanded may exceed what this camera pitch sees.")

    print("\n  CAVEAT: this frame is all floor, so the numbers above only show")
    print("  false obstacles. They say nothing about whether real obstacles are")
    print("  still rejected. Put a shoe in view and run it again before driving.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
