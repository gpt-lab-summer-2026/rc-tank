#!/usr/bin/env python3
"""
Build one floor profile from a whole recorded session.

    python3 record.py --lock-exposure          drive the room by hand
    python3 floorprofile.py dataset/session_*  build the profile
    python3 roam.py --floor-model floor.npz    use it

Pass roam's own perception flags here too — --bins, --flatten,
--channels, --scale. A histogram built under one set of them means
nothing read back under another, so they are stored in the file and
checked on load.

-------------------------------------------------------------
WHY THIS EXISTS

roam.py learns the floor from ONE patch of ONE frame at startup. In a
room with a window at one end that is a coin toss: started facing the
light it learns a bright reflective floor and reads the shaded half as
obstacle, started facing away it does the reverse. Either way it can
only drive in the direction it happened to boot in, and no threshold
fixes it — the far half of the room is a colour the patch never held,
and an unseen bin back-projects to exactly 0.

Driving the room by hand first and summing the patch across every
heading asks a better question: not what the floor looks like here,
but what it looks like in this ROOM. Bright, shaded, reflective and
matt all end up in the histogram because all of them were shown to it.

WHICH FRAMES COUNT

Only ones recorded while going forward. record.py logs the command
next to each frame, and a driver holding forward is a driver who
believed the floor ahead was clear — which is exactly the judgement
needed and impossible to recover any other way. Frames recorded while
turning or reversing are skipped: the patch during a turn often holds
the thing being turned away from.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from roam import FloorModel, add_perception_args, free_profile, regions, shrink

# Frames are resized to this before anything looks at them, so a
# session recorded at 320x240 and a camera running at 640x480 produce
# the same profile. --flatten is a sigma in PIXELS; without this the
# same number would mean different things in the two places.
WORKING = (640, 480)


def forward_frames(session: Path, every: int = 1):
    """Paths of frames recorded while driving forward."""
    log = session / "log.csv"
    frames = session / "frames"
    if not log.exists():
        raise SystemExit(f"no log.csv in {session}")

    out = []
    with open(log) as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if row.get("label") != "forward":
                continue
            if i % every:
                continue
            p = frames / row["frame"] if not row["frame"].startswith("frames") \
                else session / row["frame"]
            if not p.exists():
                p = frames / Path(row["frame"]).name
            if p.exists():
                out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="dataset/session_* directories")
    ap.add_argument("--out", default="floor.npz", help="where to write the profile")
    ap.add_argument("--every", type=int, default=1,
                    help="use every Nth forward frame; consecutive frames are "
                         "nearly identical so there is little gained below 3")
    ap.add_argument("--all-labels", action="store_true",
                    help="use every frame, not only ones recorded going forward")
    add_perception_args(ap)
    args = ap.parse_args()

    paths = []
    for d in args.sessions:
        session = Path(d)
        if not session.is_dir():
            print(f"  skipping {session}: not a directory", file=sys.stderr)
            continue
        got = (sorted((session / "frames").glob("*.jpg")) if args.all_labels
               else forward_frames(session, args.every))
        print(f"  {session.name}: {len(got)} frames")
        paths += got

    if not paths:
        print("no usable frames — was the session recorded while driving forward?",
              file=sys.stderr)
        return 1

    model = FloorModel(mode=args.channels, bins=args.bins, flatten=args.flatten)
    used = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, WORKING, interpolation=cv2.INTER_AREA)
        model.accumulate(shrink(img, args.scale))
        used += 1
    model.finalise()

    populated = int((model.hist > 0).sum())
    print(f"\nprofile from {used} frames: {model.why}")
    print(f"  {populated}/{model.hist.size} bins populated")

    # The number that says whether this was worth doing: how much of
    # the recorded floor each model accepts, measured over the SAME
    # frames. A one-patch model is scored the way roam builds it.
    single = FloorModel(mode=args.channels, bins=args.bins, flatten=args.flatten)
    first = cv2.resize(cv2.imread(str(paths[0])), WORKING, interpolation=cv2.INTER_AREA)
    single.learn(shrink(first, args.scale))

    sample = paths[:: max(1, len(paths) // 40)]
    scores = {"one patch (what roam does now)": [], "this profile": []}
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = shrink(cv2.resize(img, WORKING, interpolation=cv2.INTER_AREA), args.scale)
        scores["one patch (what roam does now)"].append(
            (single.mask(img, args.threshold, args.close) > 0).mean())
        scores["this profile"].append(
            (model.mask(img, args.threshold, args.close) > 0).mean())

    print(f"\n  floor accepted across {len(sample)} frames of the session:")
    for name, vals in scores.items():
        v = np.array(vals)
        print(f"    {name:<32} mean {v.mean()*100:5.1f}%   worst frame {v.min()*100:5.1f}%")
    print("\n  The worst frame is the one that matters. That is the heading")
    print("  roam would refuse to drive in.")

    model.save(args.out, threshold=args.threshold, close=args.close,
               scale=args.scale)
    print(f"\nwrote {args.out}\n")
    print(f"  python3 roam.py --floor-model {args.out} \\")
    print(f"      --bins {args.bins} --flatten {args.flatten:.0f} "
          f"--threshold {args.threshold} --close {args.close}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
