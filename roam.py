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

Run with --debug-image to see exactly what it samples: the cyan box
is the reference patch.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque

import cv2
import numpy as np

from car import BridgeError, Car, SoftArc, boot_warning
from record import CAMERA_ROTATION, Camera
from stream import MJPEGStreamer

# ------------------------------------------------------- floor model


class FloorModel:
    """Two-channel histogram of the ground, back-projected per pixel.

    WHICH TWO CHANNELS DEPENDS ON THE FLOOR, and getting it wrong is
    not a matter of degree — it decides whether the model has any
    signal at all.

    Hue and saturation are the right pair on a floor with actual
    colour in it. Value is excluded there because brightness moves
    with shadow and vignetting while hue does not.

    That reasoning inverts on a pale or grey floor. Saturation is
    what makes hue meaningful: HSV computes hue from which channel
    is largest, so as saturation falls towards zero the ordering is
    decided by sensor noise and lens shading rather than by colour.
    Measured on a whitewashed pine floor, mean saturation was 9 and
    hue spread +/-55 over its 0-180 range — that is not a colour, it
    is a random number. Meanwhile value sat at 192 +/-7, the tightest
    signal in the frame.

    Feeding that floor an H,S histogram put 41% of the frame in bins
    the reference patch had never seen, so two fifths of clear ground
    read as obstacle and no threshold could rescue it: back-projection
    returns exactly 0 for an unseen bin, and 0 fails every threshold
    including 0.

    So the pair is chosen from the patch rather than fixed in advance.
    """

    # Default bin count per channel. Fewer bins is a coarser idea of
    # what counts as the same colour, which is exactly what a floor
    # with texture in it wants: at 24 bins a V axis spanning 0-256 is
    # quantised every 10 levels, so grain 20 levels darker than the
    # boards lands two bins away and reads as an obstacle. At 8 bins
    # it is the same bin as the floor and disappears.
    BINS = 6

    # channels into HSV, and the range of each, per mode.
    MODES = {
        "hs": ([0, 1], (0, 180, 0, 256)),
        "sv": ([1, 2], (0, 256, 0, 256)),
    }

    # Below this mean saturation, treat the surface as grey and stop
    # trusting hue. Well above the 9 measured on bare pine and well
    # below anything that deserves to be called a colour, so the
    # decision is not close in either direction.
    ACHROMATIC_S = 40

    def __init__(self, smooth: float = 0.0, mode: str = "auto",
                 bins: int = BINS, flatten: float = 0.0):
        self.bins = (max(2, int(bins)), max(2, int(bins)))
        self.flatten = float(flatten)
        self.hist = None
        self.smooth = smooth      # 0 = learn once and never change
        self.mode = mode          # "auto" resolves on the first learn()
        self.chosen: str | None = None if mode == "auto" else mode

    @staticmethod
    def patch_box(shape) -> tuple[int, int, int, int]:
        """The trapezoid of ground the camera sees right in front."""
        h, w = shape[:2]
        return (int(w * 0.35), int(h * 0.82), int(w * 0.65), h)

    def _illumination(self, v):
        """The slow spatial brightness field, estimated cheaply.

        A large-sigma Gaussian over a full frame is expensive, and it
        is also wasted precision: what is being estimated is smooth by
        definition, so it survives being computed at an eighth of the
        resolution and scaled back up.
        """
        small = cv2.resize(v, None, fx=0.125, fy=0.125,
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(1.0, self.flatten * 0.125))
        return cv2.resize(small, (v.shape[1], v.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    def hsv_of(self, bgr):
        """HSV, with the illumination gradient optionally divided out.

        A shadow does not repaint the floor, it scales how much light
        reaches it — every channel drops by roughly the same factor.
        Dividing brightness by a blurred copy of itself therefore
        cancels the shadow and leaves reflectance, which is the thing
        that actually distinguishes floor from an object sitting on it.

        Measured on a soft-edged shadow that dropped V from 192 to
        114: the absolute model kept 0% of the shaded floor at every
        bin count and every threshold, because an unseen bin
        back-projects to 0 and 0 fails every comparison. Divided, 79%
        of it came back as floor with the obstacle still 96% detected.

        flatten is the sigma, and it is a genuine trade-off rather
        than a free win. It must be LARGER than the obstacles, or the
        blur follows an obstacle into its own interior, normalises it
        away and the tank drives into it — at sigma 25 obstacle
        detection fell to 37%. It must be SMALLER than the lighting
        variation, or there is nothing left to cancel. Somewhere near
        the size of the largest thing worth dodging.

        This runs on the WHOLE frame before any cropping, on purpose.
        Estimate the illumination from the reference patch alone and
        the ratio is 1 everywhere inside it by construction, which
        would teach the model nothing and disagree with what mask()
        computes over the full frame.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if self.flatten > 0:
            v = hsv[:, :, 2].astype(np.float32)
            hsv[:, :, 2] = np.clip(v / (self._illumination(v) + 1e-3) * 128.0,
                                   0, 255).astype(np.uint8)
        return hsv

    @property
    def channels(self):
        return self.MODES[self.chosen][0]

    @property
    def ranges(self):
        return self.MODES[self.chosen][1]

    def learn(self, bgr) -> None:
        x0, y0, x1, y1 = self.patch_box(bgr.shape)
        hsv = self.hsv_of(bgr)[y0:y1, x0:x1]

        # Decided once, from the first patch, and then left alone. A
        # model that swapped channels mid-run would be comparing
        # histograms with different axes.
        if self.chosen is None:
            sat = float(hsv[:, :, 1].mean())
            self.chosen = "hs" if sat >= self.ACHROMATIC_S else "sv"
            self.why = (f"patch saturation {sat:.0f} — "
                        + ("has colour, using hue+saturation" if self.chosen == "hs"
                           else "effectively grey, hue is noise, using saturation+value"))

        hist = cv2.calcHist([hsv], self.channels, None, self.bins, self.ranges)
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        if self.hist is None or self.smooth <= 0:
            self.hist = hist
        else:
            self.hist = (1 - self.smooth) * self.hist + self.smooth * hist

    def accumulate(self, bgr) -> int:
        """Add this frame's patch to the profile instead of replacing it.

        learn() answers "what does the floor look like HERE, NOW",
        which is the wrong question in a room with a window at one end.
        Started facing the light it learns a bright reflective floor
        and calls the shaded half an obstacle; started facing away it
        does the reverse. Either way it can only drive in the direction
        it happened to boot in.

        Accumulating raw counts over many frames, from many headings,
        asks instead what the floor looks like in this ROOM. The
        histogram ends up covering every appearance the floor actually
        has, because it was shown every one of them.

        Counts are summed before normalising, deliberately: normalising
        each frame first would let a patch of near-uniform floor —
        which is most of them — carry the same weight as one spanning
        a bright-to-dark gradient, and the gradient is the part worth
        having.
        """
        x0, y0, x1, y1 = self.patch_box(bgr.shape)
        hsv = self.hsv_of(bgr)[y0:y1, x0:x1]
        if self.chosen is None:
            sat = float(hsv[:, :, 1].mean())
            self.chosen = "hs" if sat >= self.ACHROMATIC_S else "sv"
            self.why = f"patch saturation {sat:.0f} across the corpus"
        hist = cv2.calcHist([hsv], self.channels, None, self.bins, self.ranges)
        self._raw = hist if getattr(self, "_raw", None) is None else self._raw + hist
        self.frames_seen = getattr(self, "frames_seen", 0) + 1
        return self.frames_seen

    def finalise(self) -> None:
        """Turn accumulated counts into the histogram mask() uses."""
        if getattr(self, "_raw", None) is None:
            raise ValueError("nothing accumulated")
        hist = self._raw.copy()
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self.hist = hist

    def save(self, path: str, **extra) -> None:
        """Write the profile and the settings it only makes sense under.

        The settings travel with it because a histogram built at one
        bin count, channel pair or flatten sigma is meaningless read
        back under another, and nothing about the numbers themselves
        would reveal the mismatch.
        """
        np.savez(path, hist=self.hist, mode=self.chosen,
                 bins=self.bins[0], flatten=self.flatten,
                 frames=getattr(self, "frames_seen", 1), **extra)

    @classmethod
    def load(cls, path: str, smooth: float = 0.0):
        d = np.load(path, allow_pickle=False)
        m = cls(smooth=smooth, mode=str(d["mode"]),
                bins=int(d["bins"]), flatten=float(d["flatten"]))
        m.hist = d["hist"].astype(np.float32)
        m.frames_seen = int(d["frames"])
        m.why = (f"loaded from {path}: {m.frames_seen} frames, "
                 f"{m.chosen}, bins {m.bins[0]}, flatten {m.flatten:.0f}")
        return m

    def mask(self, bgr, threshold: int = 40, close_px: int = 11,
             horizon: float = 0.0):
        """255 where the pixel looks like floor.

        close_px is what separates floor texture from obstacles, and it
        does it by SHAPE rather than by colour. Wood grain, grout lines
        and carpet pile read as thin elongated not-floor; a thing worth
        stopping for reads as compact. Closing by more than the texture
        is wide fills the first and leaves the second alone.

        The alternative — widening the histogram until the texture
        falls inside it — was measured and is worse. Grain on a pale
        floor dips into the same brightness band as a pale obstacle, so
        a histogram loose enough to swallow the grain went blind to a
        shoe as well: detection of it fell from 88% to nothing, while
        closing at 11 kept both it and a phone above 87%.

        The cost is that anything thinner than close_px is erased with
        the grain. At 640 wide that is under 2% of the frame, so cables
        and chair spindles seen edge-on are gone. They were already
        marginal here; this makes it certain.
        """
        hsv = self.hsv_of(bgr)
        back = cv2.calcBackProject([hsv], self.channels, self.hist, self.ranges, 1)

        # Blur first so isolated speckles do not cut a column short,
        # then close texture-sized holes, then drop floor specks that
        # are too small to stand on. The two kernels are deliberately
        # different sizes: the first is sized to the floor's texture,
        # the second to noise.
        cv2.filter2D(back, -1, np.ones((5, 5), np.float32) / 25, back)
        _, m = cv2.threshold(back, threshold, 255, cv2.THRESH_BINARY)
        close_px = max(3, close_px | 1)                   # odd, and never degenerate
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)

        # Above the horizon nothing is driveable, whatever colour it
        # is. That is geometry, not appearance, and it is the only
        # thing here a colour model cannot argue with.
        #
        # It needs saying explicitly because both of the tools that
        # widen the model's idea of floor will otherwise swallow the
        # wall. --flatten divides brightness by a local blur, which
        # makes ANY locally-uniform surface normalise to the same
        # value — a flat wall and a flat floor come out identical. And
        # a profile accumulated across a room lit from one end spans
        # far enough that the floor's shaded end reaches the wall's
        # brightness. Measured with a wall at 118 and floor from 115 to
        # 210, both routes called the wall floor over 96% of the time.
        #
        # A column that keeps counting past the horizon reports
        # clearance to infinity and the tank drives into the wall.
        if horizon > 0:
            m[: int(m.shape[0] * horizon)] = 0
        return m


def shrink(bgr, scale: float):
    """Downscale before anything looks at the frame.

    INTER_AREA averages the pixels it discards rather than sampling
    one of them, so floor texture is gone before the histogram is
    built instead of being argued with afterwards. Everything
    downstream — patch, mask, morphology, profile — then works on a
    frame that never had the grain in it.

    Clearances stay honest because --go and friends are fractions of
    frame height. Kernel sizes do not: --close and --min-obstacle are
    pixels, and shrink with the frame.
    """
    if scale >= 1.0:
        return bgr
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def free_profile(mask, min_obstacle: int = 1) -> np.ndarray:
    """Unbroken floor height per column, counting up from the bottom.

    A column is only free as far as the first thing blocking it — an
    obstacle hides everything behind it regardless of what the floor
    does further up the frame.

    min_obstacle is how tall a run of not-floor has to be before it
    counts as blocking. At 1 this is the original rule, where the
    first stray pixel ends the column, and on a textured floor that
    is brutal: a few pixels of wood grain lying across the bottom row
    truncate a column that is otherwise clear to the horizon. That is
    how a mask which is visibly almost all floor still reports
    L 0 C 14 R 0 — the clearance is real, the profile just never gets
    to see it.

    Set it to the smallest vertical extent an obstacle worth avoiding
    would occupy in frame. Shorter than that is read as texture and
    driven straight over, which is the right call when the things
    being dodged are large and the floor is patterned.
    """
    floor = mask > 0

    if min_obstacle > 1:
        # Keep only not-floor runs at least min_obstacle tall. The
        # kernel is anchored at its BOTTOM row, so a pixel survives
        # only when the run continues upward from it — which makes the
        # surviving pixel the lowest one of the obstacle, exactly the
        # height the search below wants to stop at.
        k = np.ones((min_obstacle, 1), np.uint8)
        solid = cv2.erode((~floor).astype(np.uint8), k,
                          anchor=(0, min_obstacle - 1))
        floor = solid == 0

    flipped = floor[::-1]                     # bottom row first
    blocked = np.argmax(~flipped, axis=0)     # first blocker going up
    blocked[np.all(flipped, axis=0)] = mask.shape[0]   # never blocked
    return blocked


def regions(profile: np.ndarray, pct: int = 25) -> tuple[float, float, float]:
    """Free height for left, centre and right thirds.

    A low percentile rather than the mean, so one thin chair leg
    still counts as blocking that side. That cuts both ways: at 25 a
    quarter of the columns speckled with texture is enough to call a
    third blocked. Raise it when the things being dodged are wide
    enough to block most of a third on their own.
    """
    w = len(profile)
    third = w // 3
    return (
        float(np.percentile(profile[:third], pct)),
        float(np.percentile(profile[third : 2 * third], pct)),
        float(np.percentile(profile[2 * third :], pct)),
    )


# ------------------------------------------------------------ policy


# Track pair for each move that is a single held state. Soft arcs are
# not in here — they are not one state, see SoftArc.
#
# There are no single-track reverse states here. Backing one track
# against a stopped one digs this chassis in instead of pivoting it,
# so a reverse turn is soft_back_left / soft_back_right: both tracks
# reversing, briefly interrupted to swing the nose. Those are not in
# this table because they are not one state — see tracks_for().
TRACKS = {
    "forward":       (1, 1),
    "reverse":       (-1, -1),
    "stop":          (0, 0),
    "arc_left":      (0, 1),
    "arc_right":     (1, 0),
}


def tracks_for(move: str, now: float, soft: SoftArc) -> tuple[int, int]:
    """Resolve a move name to the pair of track directions for now."""
    if move.startswith("soft_"):
        # soft_back_* run the cycle in reverse: both tracks backing
        # between pulses rather than both driving.
        return soft.tracks(move.endswith("right"), now,
                           reverse="back" in move)
    return TRACKS[move]


class Policy:
    """Free space in, a named move out.

    Turns are arcs, not spins. Driving one track against the other
    pivots on the spot and is the obvious way to turn a tank, but this
    chassis does it badly — and it is the hardest move there is on the
    contacts, since both pairs reverse at once.

    An arc drives one track and leaves the other stopped. That turns
    this tank better and halves the relay work, at the cost of no
    longer turning on the spot: a forward arc travels while it turns.
    Which is fine when there is room ahead and precisely wrong when
    there is not, so a badly blocked centre gets a REVERSE arc — still
    one track, still a proper turn, but retreating from the thing it
    is turning away from rather than creeping into it.
    """

    # Every move that turns. Forward is the only one that does not,
    # and the only one allowed to run without a deadline.
    TURNS = frozenset((
        "soft_arc_left", "soft_arc_right",
        "arc_left", "arc_right",
        "soft_back_left", "soft_back_right",
    ))

    def __init__(self, go: float, commit: float, even: float,
                 turn_margin: float, stuck_after: float,
                 reverse_for: float, soft_margin: float = 0.0,
                 back_below: float = 0.35, dodge_min: float = 0.5,
                 max_turn: float = 4.0, straighten_for: float = 1.0):
        self.go = go                      # centre clearance to keep going
        # Above this the centre is not merely passable but plainly
        # open, and nothing either side is worth turning for.
        self.commit = commit
        # And above this the WORST of the three is still fine, so
        # there is no better direction to turn toward.
        self.even = even
        self.turn_margin = turn_margin    # how much better a side must be
        self.stuck_after = stuck_after
        self.reverse_for = reverse_for
        # How lopsided the sides must be before nudging away from the
        # closer one while still going forward. 0 disables it.
        self.soft_margin = soft_margin
        # Below this fraction of go, a forward arc would drive into
        # whatever is ahead, so turn while backing instead.
        self.back_below = back_below
        # How much room a side needs, as a fraction of go, before it
        # counts as somewhere to dodge INTO rather than just the less
        # bad wall. This is the whole difference between the two cases:
        # above it there is a way past, below it there is not.
        self.dodge_min = dodge_min
        # How long any turning move may run before it is made to stop
        # turning, whether or not it looks like it is getting
        # anywhere. See _turn_expired.
        self.max_turn = max_turn
        self.straighten_for = straighten_for
        self._turning_since = None
        self._reverse_until = None
        self._escape_move = None
        self._dodge_side = None
        self._turn_since = None
        self._straighten_until = None

        # Why the last move was chosen. There is no way to argue with
        # "it reversed for no reason" without this: every branch below
        # is a decision about numbers nobody wrote down, and by the
        # time the tank has backed up the numbers are gone.
        self.reason = "start"
        self.reversals = 0

    def _say(self, reason: str, move: str) -> str:
        self.reason = reason
        return move

    def _escape(self, left_is_better: bool, now) -> str:
        """Back out of somewhere with no way through, turning as it goes.

        The heading is chosen once, here, and held for the whole
        retreat. Re-deciding it every frame is what a straight reverse
        effectively did: the camera is pointing the wrong way, so the
        numbers it is re-deciding from are about ground already behind
        the tank, and the left/right call flickers on noise. Averaged
        over a second that is no turn at all — reversing in a line,
        arriving with the same view, and doing it again.

        Committing means the nose is genuinely somewhere else when the
        retreat ends, which is the only thing that stops the loop.
        """
        self._escape_move = ("soft_back_left" if left_is_better
                             else "soft_back_right")
        self._reverse_until = now + self.reverse_for
        self._turning_since = None
        self.reversals += 1
        return self._escape_move

    def decide(self, left, centre, right, now) -> str:
        # A retreat runs for a fixed time on the heading it picked, then
        # hands back to the normal rules. Backing until the view ahead
        # clears reads as the sensible version and is not: the camera
        # faces the other way, so the longer it runs the less anything
        # knows about where the tank is going.
        if self._reverse_until is not None:
            if now < self._reverse_until:
                return self._say(self.reason, self._escape_move)
            self._reverse_until = None
            self._escape_move = None
            self._turning_since = None    # turning gets a fresh chance
            self._dodge_side = None       # and a fresh look before choosing

        # Straightening out after a turn overran its limit. Cut it
        # short if the view closes in: a forced forward is a recovery,
        # not a licence to drive into something.
        if self._straighten_until is not None:
            if now < self._straighten_until and centre >= self.go * self.back_below:
                return self._say("straightening", "forward")
            self._straighten_until = None
            self._dodge_side = None

        move = self._choose(left, centre, right, now)

        # An escape carries its own deadline, so it is not the cap's
        # business. Neither is anything that is not a turn.
        if self._reverse_until is not None or move not in self.TURNS:
            self._turn_since = None
            return move

        if self._turn_since is None:
            self._turn_since = now
        elif now - self._turn_since > self.max_turn:
            return self._turn_expired(centre, now)
        return move

    def _turn_expired(self, centre, now) -> str:
        """A turn has been held for as long as it is safe to hold one.

        One track idling cannot always move this chassis. On carpet,
        against a skirting board, or nosed into anything at all, the
        driving track stalls instead of turning — full current, no
        motion, and on this hardware that is also how contacts weld
        and how the 5V rail sags. Nothing in the picture says it is
        happening, because a stalled tank and a slowly turning one
        look much the same from a camera.

        So turning gets a deadline and driving does not. Both tracks
        pulling together is the thing most likely to break a stall one
        track alone could not, which is why the way out of here is to
        drive straight for a moment rather than to stop.
        """
        self._turn_since = None
        if centre >= self.go * self.back_below:
            self._straighten_until = now + self.straighten_for
            self._dodge_side = None
            return self._say("turn-capped", "forward")
        # Nothing ahead worth driving into. Give ground instead.
        return self._say("turn-capped-blocked",
                         self._escape(self._dodge_side != "right", now))

    def _choose(self, left, centre, right, now) -> str:
        # Plainly open ahead. Commit to it and stop looking sideways.
        #
        # Below this the drift rule is worth having: a side closing in
        # is worth easing away from before it becomes a hard turn.
        # Above it that same rule is what keeps the tank fiddling in a
        # corner — both sides read as interesting, it corrects toward
        # one, the correction changes the view, it corrects back, and
        # it works its way around the room without ever crossing it.
        #
        # Going somewhere requires being willing to ignore a better
        # option to either side. This is that willingness, expressed
        # as a number.
        if centre >= self.commit:
            self._turning_since = None
            self._dodge_side = None
            return self._say("committed", "forward")

        # Nothing is notably worse than anything else, and none of it
        # is tight. A turn is only ever worth making because some
        # direction is BETTER; when the worst of the three is still
        # comfortable there is no better, so turning buys nothing and
        # costs the heading.
        #
        # This is checked before the centre-clearance rule on purpose,
        # so it holds even when --go is set high enough that an evenly
        # clearish view would otherwise read as blocked and get dodged.
        if min(left, centre, right) >= self.even:
            self._turning_since = None
            self._dodge_side = None
            return self._say("all-clear", "forward")

        if centre >= self.go:
            self._turning_since = None
            self._dodge_side = None
            # Room ahead. Drift away from whichever side is closer
            # rather than holding course until it becomes an obstacle
            # and forces a hard turn. Costs contact operations that
            # going straight does not, which is why soft_margin has
            # to be asked for.
            # Only when there is something to drift away FROM, though.
            # A lopsided view of two clear sides — one wall three
            # metres off, the other five — is not a reason to turn,
            # and steering on the difference alone had the tank
            # correcting its way across empty floor. If both sides are
            # clearer than the distance wanted straight ahead, nothing
            # needs avoiding yet: hold course.
            if self.soft_margin > 0 and min(left, right) < self.go:
                if left - right > self.soft_margin:
                    return self._say("drift", "soft_arc_left")
                if right - left > self.soft_margin:
                    return self._say("drift", "soft_arc_right")
            return self._say("clear", "forward")

        # Blocked ahead. The question is now which of two situations
        # this is, and they want opposite things:
        #
        #   somewhere to go   -> dodge, keep the ground already made
        #   nowhere to go     -> give the ground up and turn while
        #                        doing it, so the next look is at
        #                        something different
        #
        # Ties and near-ties break left so the tank does not dither.
        #
        # Chosen once, then held until the way ahead opens or the
        # retreat resets it. left and right are percentiles of a noisy
        # mask and near an obstacle they sit close together, so
        # re-deciding every frame lets that noise steer: the tank
        # weaves at the thing rather than going round it, never
        # commits far enough either way to find the gap, and ends up
        # backing out of somewhere it could have driven past.
        # Committing is most of what turns a dodge into a swerve.
        if self._dodge_side is None:
            self._dodge_side = "right" if right > left + self.turn_margin else "left"
        left_is_better = self._dodge_side == "left"

        # Nowhere to go: neither side has enough room to be a way past,
        # so this is a corner or a wall rather than an obstacle with a
        # gap beside it. Straight back would return to this same spot
        # facing the same way.
        if max(left, right) < self.go * self.dodge_min:
            return self._say("boxed-in", self._escape(left_is_better, now))

        # Almost nothing ahead. A forward arc here would turn while
        # driving into the thing being turned away from, so give up the
        # ground instead and yaw the same way going backwards.
        #
        # Soft, not a held single-track reverse: both tracks back out
        # together and the turn comes from interrupting one of them.
        # Held, this chassis barely turns backwards at all.
        if centre < self.go * self.back_below:
            # Only this branch counts toward being stuck, because it is
            # the only one that is not getting anywhere. If backing and
            # yawing has not opened the view in stuck_after seconds it
            # is not going to, and the committed retreat is the
            # fallback.
            if self._turning_since is None:
                self._turning_since = now
            elif now - self._turning_since > self.stuck_after:
                return self._say("no-progress", self._escape(left_is_better, now))
            return self._say("too-close",
                             "soft_back_left" if left_is_better else "soft_back_right")

        # Room on one side and room to move: go round. This is the
        # move the tank should spend most of its blocked time in.
        #
        # The stuck timer is cleared rather than left running. A
        # forward arc IS progress — it covers ground while it turns —
        # and timing it out was what turned every obstacle wider than
        # a few seconds of arc into a retreat, however well the swerve
        # was going. A tank arcing along a wall gets to the end of the
        # wall; dodge_min above catches the corner when it arrives.
        self._turning_since = None
        return self._say("dodge", "arc_left" if left_is_better else "arc_right")


class Smoother:
    """Majority vote plus a floor on how often relays may change.

    Contacts are rated for a limited number of operations, and a
    single misclassified frame should not cost one.
    """

    def __init__(self, window: int, min_interval: float):
        self.buf: deque = deque(maxlen=window)
        self.min_interval = min_interval
        self.current = "stop"
        self.changed_at = 0.0

    def update(self, decision: str, now: float) -> str:
        self.buf.append(decision)
        if len(self.buf) < self.buf.maxlen:
            return self.current

        winner, _ = Counter(self.buf).most_common(1)[0]
        if winner != self.current and now - self.changed_at >= self.min_interval:
            self.current = winner
            self.changed_at = now
        return self.current


# ------------------------------------------------------------- debug


def annotate(bgr, mask, prof, regs, move, marks=(), dets=(),
             small_max: float = 1.0, det_age: float = 0.0):
    """Draw what the policy is looking at, over the frame it looked at.

    marks are horizontal reference lines — the thresholds the numbers
    are being compared against. Without them the red profile is just a
    squiggle; with them you can see at a glance whether a column
    clears the bar, which is the entire question the policy asks.

    dets are drawn in two shades, because which ones get REPORTED is a
    tuning decision and it should be visible. Anything at or under
    small_max of the frame is something on the floor and is announced,
    so it is drawn bright. Anything larger is a wall or a piece of
    furniture that roam is already steering around, so it is drawn dim
    and its area is labelled — which is what you read when deciding
    whether --small-object is set where you want it.

    det_age says how long ago the boxes were computed. Detection runs
    on a worker thread and takes longer than a tick, so the boxes are
    ALWAYS from an older frame than the pixels under them. Saying how
    much older is the difference between a lagging box and a wrong one.
    """
    out = bgr.copy()
    green = np.zeros_like(out)
    green[:, :, 1] = mask
    out = cv2.addWeighted(out, 0.7, green, 0.3, 0)

    h, w = mask.shape
    for x in range(0, w, 4):
        y = h - int(prof[x])
        cv2.circle(out, (x, y), 1, (0, 0, 255), -1)

    # Thresholds, drawn where the profile has to reach to clear them.
    for value, label, colour in marks:
        y = int(h - value)
        if 0 <= y < h:
            cv2.line(out, (0, y), (w, y), colour, 1)
            cv2.putText(out, label, (w - 78, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    x0, y0, x1, y1 = FloorModel.patch_box(bgr.shape)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 2)

    for i, v in enumerate(regs):
        cv2.putText(
            out, f"{v:.0f}", (int(w * (i / 3 + 0.12)), 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
    for d in dets:
        reported = d.area_frac <= small_max
        colour = (0, 140, 255) if reported else (130, 130, 130)
        bx0, by0, bx1, by1 = d.box
        cv2.rectangle(out, (bx0, by0), (bx1, by1), colour, 2 if reported else 1)

        tag = f"{d.label} {d.confidence:.0%}"
        if not reported:
            tag += f"  {d.area_frac:.0%} of frame"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Above the box, or inside it when the box is against the top.
        ty = by0 - 4 if by0 - th - 6 >= 0 else min(by1, by0 + th + 6)
        tx = max(0, min(bx0, w - tw - 4))
        cv2.rectangle(out, (tx, ty - th - 4), (tx + tw + 4, ty + 2), colour, -1)
        cv2.putText(out, tag, (tx + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if dets:
        note = f"{len(dets)} detected, {det_age:.1f}s ago"
        cv2.putText(out, note, (w - 190, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

    cv2.putText(
        out, move, (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    return out


# -------------------------------------------------------------- main


def add_perception_args(ap) -> None:
    """Flags for anything that runs the floor model.

    Defined here and shared rather than repeated per script, so a
    default tuned in one place cannot quietly disagree with the
    same flag somewhere else.
    """
    ap.add_argument("--commit-above", type=float, default=0.625,
                    help="centre clearance, as a fraction of frame height, above "
                         "which roam drives straight and ignores both sides "
                         "entirely. Stops it grooming its way around a room "
                         "instead of crossing it. 0.625 is 300px at 480 high")
    ap.add_argument("--even-above", type=float, default=0.5625,
                    help="when the WORST of left, centre and right is above this "
                         "fraction of frame height, go straight. Nothing is worse "
                         "than anything else, so no turn improves matters. "
                         "0.5625 is 270px at 480 high")
    ap.add_argument("--go", type=float, default=0.45,
                    help="centre clearance to keep going, as a fraction of frame height")
    ap.add_argument("--threshold", type=int, default=40, help="floor match strictness 0-255")
    ap.add_argument("--min-obstacle", type=int, default=1,
                    help="pixels of continuous not-floor a column must hit "
                         "before it counts as blocked. 1 is the old rule, where "
                         "one speck of grain truncates a clear column. Raise it "
                         "to the smallest obstacle you actually care about")
    ap.add_argument("--close", type=int, default=11,
                    help="fill not-floor gaps thinner than this many pixels — wood "
                         "grain and grout, not obstacles. Raise until the floor "
                         "stops speckling; anything thinner is erased with it")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink each frame by this factor before looking at it. "
                         "0.25 turns 640x480 into 160x120, which averages floor "
                         "texture away before the histogram ever sees it and runs "
                         "far faster. NOTE --close and --min-obstacle are in "
                         "pixels, so they shrink with it")
    ap.add_argument("--bins", type=int, default=FloorModel.BINS,
                    help="histogram bins per channel. Fewer is a coarser idea of "
                         "the same colour: 8 puts wood grain in the same bin as "
                         "the boards, 24 puts it two bins away")
    ap.add_argument("--flatten", type=float, default=0.0,
                    help="divide the slow brightness gradient out before matching, "
                         "sigma in pixels. This is what stops shadows reading as "
                         "obstacles; no other knob can. Set it larger than the "
                         "things you dodge and smaller than the lighting changes. "
                         "0 is off")
    ap.add_argument("--horizon", type=float, default=0.0,
                    help="rows above this fraction of the frame are never floor, "
                         "whatever they look like. Set it to where the horizon "
                         "sits when the camera is pitched to see past the floor. "
                         "Only the navigation mask is clipped — the object "
                         "detector still sees the whole frame")
    ap.add_argument("--percentile", type=int, default=25,
                    help="how blocked a third must be to count as blocked. 25 "
                         "means a quarter of its columns; raise it when obstacles "
                         "are wide and texture is what is truncating columns")
    ap.add_argument("--channels", default="auto", choices=("auto", "hs", "sv"),
                    help="which two HSV channels model the floor. auto picks sv on a "
                         "grey floor, where hue is noise, and hs where there is colour")
    ap.add_argument("--floor-model", default=None, metavar="FILE.npz",
                    help="load a profile built by floorprofile.py from a recorded "
                         "session, instead of learning from one patch at startup. "
                         "This is the fix for a room lit from one end")
    ap.add_argument("--adapt", type=float, default=0.0,
                    help="floor model blend rate, 0 learns once")
    ap.add_argument("--no-lock-exposure", action="store_true",
                    help="leave AE/AWB running (the floor model will drift)")
    ap.add_argument("--rotate", type=int, default=CAMERA_ROTATION, choices=(0, 180),
                    help="turn each frame before anything looks at it "
                         f"(default {CAMERA_ROTATION}, this camera is mounted upside down)")


def add_detect_args(ap) -> None:
    """Flags for anything that runs the object detector.

    Defined here and shared rather than repeated per script, so a
    default tuned in one place cannot quietly disagree with the
    same flag somewhere else.
    """
    ap.add_argument("--detect", default=None, metavar="MODEL.onnx",
                    help="YOLOv8/v11 ONNX model. Detection runs on a worker "
                         "thread and never holds the driving loop up")
    ap.add_argument("--detect-every", type=float, default=1.5,
                    help="seconds between frames offered to the detector while "
                         "roaming")
    ap.add_argument("--detect-conf", type=float, default=0.40,
                    help="confidence below which a detection is not reported")
    ap.add_argument("--detect-threads", type=int, default=2,
                    help="cores the model may use. All four starves the control "
                         "loop, which is what this design exists to prevent")
    ap.add_argument("--small-object", type=float, default=0.08,
                    help="while driving, only report things smaller than this "
                         "fraction of the frame. Bigger ones are the walls and "
                         "furniture roam is already steering around")
    ap.add_argument("--report-cooldown", type=float, default=25.0,
                    help="seconds before the same label is announced again")


def add_view_args(ap) -> None:
    """Flags for anything that serves a live picture.

    Defined here and shared rather than repeated per script, so a
    default tuned in one place cannot quietly disagree with the
    same flag somewhere else.
    """
    ap.add_argument("--stream", type=int, default=0, metavar="PORT",
                    help="serve the annotated frame as MJPEG on this port, so "
                         "the thing being tuned can be watched while it runs. "
                         "0 is off")
    ap.add_argument("--stream-fps", type=float, default=5.0,
                    help="how often to publish a frame; annotating and encoding "
                         "cost loop time, so this is deliberately below --fps")


def marks_for(args, h):
    """The threshold lines annotate() draws, in profile pixels.

    getattr with roam's own defaults, because the policy flags these
    come from are roam's and teleop does not register them — but the
    lines are just as useful when driving by hand, which is when you
    are deciding where to put them.
    """
    go_px = args.go * h
    return [
        (getattr(args, "commit_above", 0.625) * h, "commit", (255, 255, 255)),
        (go_px, "go", (0, 255, 255)),
        (go_px * getattr(args, "dodge_min", 0.35), "dodge", (255, 160, 0)),
        (go_px * getattr(args, "back_below", 0.22), "back", (200, 0, 255)),
    ]


def perceive(raw, floor, args):
    """Frame in, everything the policy reads out.

    Shared so teleop shows exactly what roam would decide from, rather
    than an approximation of it that drifts the first time either is
    tuned.
    """
    frame = shrink(raw, args.scale)
    mask = floor.mask(frame, args.threshold, args.close, args.horizon)
    prof = free_profile(mask, args.min_obstacle)
    return frame, mask, prof, regions(prof, args.percentile)




def main() -> int:
    ap = argparse.ArgumentParser()
    add_perception_args(ap)
    add_detect_args(ap)
    add_view_args(ap)
    ap.add_argument("--port", default=None, help="ESP32 serial device")
    ap.add_argument("--dry-run", action="store_true", help="decide but do not drive")
    ap.add_argument("--fps", type=float, default=10.0, help="decisions per second")
    ap.add_argument("--turn-margin", type=float, default=0.05,
                    help="how much clearer one side must be, as a fraction")
    ap.add_argument("--no-servo", action="store_true",
                    help="do not raise the camera mast, and do not open the bridge "
                         "just to raise it when dry running")
    ap.add_argument("--soft-margin", type=float, default=0.10,
                    help="how much clearer one side must be before drifting away "
                         "from the other while still going forward, as a fraction "
                         "of frame height. 0 disables the soft arc")
    ap.add_argument("--soft-period", type=float, default=1.2,
                    help="seconds per soft-arc cycle; shorter turns harder and "
                         "costs contact life faster")
    ap.add_argument("--soft-duty", type=float, default=0.30,
                    help="fraction of each soft-arc cycle spent arcing")
    ap.add_argument("--dodge-min", type=float, default=0.35,
                    help="room a side needs, as a fraction of --go, before it "
                         "counts as a way past rather than the less bad wall. "
                         "Below it on both sides, roam backs out and turns "
                         "instead of trying to squeeze through")
    ap.add_argument("--back-below", type=float, default=0.22,
                    help="turn while REVERSING when centre clearance falls below "
                         "this fraction of --go, instead of arcing forward into it")
    ap.add_argument("--survey-every", type=float, default=0.0,
                    help="seconds between survey stops: halt on clear floor, "
                         "raise the mast to look at the room, report, carry on. "
                         "0 is off")
    ap.add_argument("--window", type=int, default=5, help="frames in the majority vote")
    ap.add_argument("--min-interval", type=float, default=0.2,
                    help="seconds between relay changes")
    ap.add_argument("--stuck-after", type=float, default=3.0,
                    help="seconds of turning before reversing")
    ap.add_argument("--max-turn", type=float, default=4.0,
                    help="longest any turning move may run. One track idling "
                         "cannot always move this chassis, and a stalled "
                         "track draws full current with nothing to show for "
                         "it. Forward and reverse are not capped")
    ap.add_argument("--straighten-for", type=float, default=1.0,
                    help="seconds of driving straight after a turn hits "
                         "--max-turn; both tracks together is what breaks a "
                         "stall one track could not")
    ap.add_argument("--reverse-for", type=float, default=1.0,
                    help="seconds to reverse before trying to turn again")
    ap.add_argument("--command-timeout", type=float, default=None,
                    help="seconds of silence from this loop before the bridge "
                         "watchdog is allowed to release the relays "
                         "(default: five ticks, at least 0.5s)")
    ap.add_argument("--reverse-cooldown", type=float, default=2.0,
                    help="seconds before a motor may be reversed again — the "
                         "one change that fights the motor's own momentum")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds before any gentler change — arc, stop, "
                         "start. Raising this delays every swerve")
    ap.add_argument("--debug-image", default=None, help="write an annotated frame here")
    args = ap.parse_args()

    # The floor model is a histogram of colours, learned once. Leaving
    # auto-exposure and auto-white-balance running means the camera
    # keeps changing those colours underneath it, and the model rots
    # over minutes for no reason that appears in any log.
    # Tied to the tick rate rather than fixed: too short and a slow
    # frame drops the motors for no reason, too long and a loop that
    # has genuinely stopped keeps driving. Five ticks tolerates a
    # stall without tolerating a death.
    command_timeout = args.command_timeout
    if command_timeout is None:
        command_timeout = max(0.5, 5.0 / args.fps)

    # The bridge opens BEFORE the camera now, which is the reverse of
    # what it used to do. The mast servo hangs off the bridge, and the
    # mast has to be up before the sensor spends two seconds settling
    # its exposure on whatever it can see.
    #
    # It is also why --dry-run opens the bridge at all. Dry run means
    # "decide but do not drive", not "do not touch the hardware": with
    # the mast stowed there is nothing worth deciding about. No relay
    # command is sent while dry running, so the firmware watchdog holds
    # the motors released throughout — nothing refreshes them.
    driving = not args.dry_run
    car = None
    if driving or not args.no_servo:
        try:
            car = Car(port=args.port, command_timeout=command_timeout,
                      command_cooldown=args.cooldown,
                      reverse_cooldown=args.reverse_cooldown)
            print(f"bridge on {car.port}  (releases after {command_timeout:.1f}s silent)")
            car.chirp(Car.TUNE_ROAM)
            warning = boot_warning(car.boot_reason)
            if warning:
                print(f"\n  !! {warning}\n")
        except BridgeError as e:
            if driving:
                print(f"could not open the bridge: {e}", file=sys.stderr)
                return 1
            # Dry running is still worth doing without a mast, as long
            # as it is clear the camera is wherever it was left.
            print(f"  !! no bridge, so no mast: {e}", file=sys.stderr)

    cam = Camera(lock_exposure=not args.no_lock_exposure, rotate=args.rotate,
                 mast=None if args.no_servo else car)

    if args.floor_model:
        # A profile built from a driven session already covers every
        # heading in the room, so there is nothing to learn here and
        # nothing to point at first.
        floor = FloorModel.load(args.floor_model, smooth=args.adapt)
        if (floor.bins[0], floor.flatten) != (args.bins, args.flatten):
            print(f"  !! profile was built with bins {floor.bins[0]} flatten "
                  f"{floor.flatten:.0f}, you passed bins {args.bins} flatten "
                  f"{args.flatten:.0f} — using the profile's", file=sys.stderr)
    else:
        print("\nlearning the floor — keep a metre of clear ground ahead")
        time.sleep(1.0)
        floor = FloorModel(smooth=args.adapt, mode=args.channels, bins=args.bins,
                           flatten=args.flatten)
        floor.learn(shrink(cam.frame(), args.scale))
    print(f"learned: {getattr(floor, 'why', floor.chosen)}\n")

    # A cooldown longer than the arc pulse turns every pulse into a
    # resend of the previous state: the tank drives straight and
    # nothing says why. Cheaper to hear about it here.
    if args.soft_margin > 0 and car is not None:
        probe = SoftArc(period=args.soft_period, duty=args.soft_duty)
        if probe.swallowed_by(getattr(car, "command_cooldown", 0.0)):
            print(f"  !! command_cooldown {car.command_cooldown:.2f}s is longer than the "
                  f"soft-arc pulse\n     ({args.soft_period*args.soft_duty:.2f}s) — "
                  f"the arc would never reach the relays. Soft arc disabled.\n")
            args.soft_margin = 0.0
        else:
            print(f"  soft arc on: ~{probe.ops_per_second:.1f} contact ops/sec "
                  f"while correcting\n")

    frame = shrink(cam.frame(), args.scale)
    h = frame.shape[0]
    go_px = args.go * h
    margin_px = args.turn_margin * h

    # A camera pitched to see over the horizon caps how much floor any
    # column can ever show: nothing above the horizon is floor, so the
    # profile stops there no matter how empty the room is. A threshold
    # set above that cap can never be met, and the branch behind it is
    # dead code that reads as "permanently blocked" instead.
    probe = free_profile(floor.mask(shrink(cam.frame(), args.scale),
                                    args.threshold, args.close),
                         args.min_obstacle)
    reach = float(probe.max())
    print(f"most clearance any column can show right now: {reach:.0f}px "
          f"({reach / h:.2f} of frame)")
    for name, want in (("--go", go_px),
                       ("--even-above", args.even_above * h),
                       ("--commit-above", args.commit_above * h)):
        if want > reach:
            print(f"  !! {name} wants {want:.0f}px, which is more than anything "
                  f"here can reach.\n     That test can never pass — lower it "
                  f"below {reach:.0f} ({reach / h:.2f}).", file=sys.stderr)
    if go_px >= args.commit_above * h:
        print(f"  !! --go ({go_px:.0f}) is not below --commit-above "
              f"({args.commit_above * h:.0f}).\n     They are checked in the "
              f"order commit, even, go, so go never gets a turn.",
              file=sys.stderr)
    print()

    policy = Policy(go_px, args.commit_above * h, args.even_above * h,
                    margin_px,
                    args.stuck_after, args.reverse_for,
                    soft_margin=args.soft_margin * h, back_below=args.back_below,
                    dodge_min=args.dodge_min, max_turn=args.max_turn,
                    straighten_for=args.straighten_for)
    soft = SoftArc(period=args.soft_period, duty=args.soft_duty)
    smoother = Smoother(args.window, args.min_interval)

    view = None
    if args.stream:
        try:
            view = MJPEGStreamer(port=args.stream)
            print(f"live view on http://<this-pi>:{view.port}/")
            print(f"  if the network will not route it, tunnel instead:")
            print(f"    ssh -N -L {view.port}:localhost:{view.port} "
                  f"{__import__('getpass').getuser()}@<this-pi>")
            print(f"  then open http://localhost:{view.port}/\n")
        except OSError as e:
            print(f"  !! no live view: {e}", file=sys.stderr)

    # Thresholds the policy compares against, in profile pixels, so
    # annotate can draw them where they actually sit in the frame.
    marks = marks_for(args, h)

    last_view = 0.0
    view_period = 1.0 / max(0.1, args.stream_fps)

    detector = reporter = None
    if args.detect:
        try:
            from detect import Detector, Reporter
            detector = Detector(args.detect, conf=args.detect_conf,
                                threads=args.detect_threads)
            reporter = Reporter(cooldown=args.report_cooldown)
            print(f"detector {args.detect} at {detector.size}x{detector.size}, "
                  f"{args.detect_threads} threads")
            if args.survey_every:
                print(f"  survey stop every {args.survey_every:.0f}s, "
                      f"mast to {Car.MAST_SURVEY}")
            print()
        except Exception as e:
            print(f"  !! no detector: {e}\n", file=sys.stderr)

    last_detect = 0.0
    last_survey = time.monotonic()
    # Latest floor-context detections, kept so the view can draw them
    # between runs rather than flashing them for one frame in six.
    # Survey results are deliberately NOT kept: they were taken with
    # the mast at a different angle, so their boxes mean nothing over
    # a driving frame.
    shown_dets: list = []
    shown_at = 0.0      # do not survey the instant it starts

    def say(line: str) -> None:
        """Print above the status line without shredding it."""
        print("\r" + " " * 78 + "\r" + line, flush=True)

    period = 1.0 / args.fps
    next_tick = time.monotonic()
    last_debug = 0.0
    last_error = None

    try:
        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(0.005)
                continue
            next_tick = max(now + period, next_tick + period)

            frame, mask, prof, regs = perceive(cam.frame(), floor, args)

            move = smoother.update(policy.decide(*regs, now), now)
            decision = tracks_for(move, now, soft)

            # Resent every tick, not only when it changes. The firmware
            # ignores a state it has already applied, so this costs no
            # relay operations, and it makes the bridge watchdog track
            # whether this loop is still running. Send only on change
            # and a hung loop looks exactly like a healthy one holding
            # course — which, with nothing watching for a collision, is
            # the difference between stopping and driving into a wall.
            if car is not None and driving:
                try:
                    car.drive(*decision)
                    last_error = None
                except BridgeError as e:
                    if str(e) != last_error:     # do not flood at 10 Hz
                        print(f"\rbridge error: {e}")
                        last_error = str(e)

            # Only adapt when the view ahead is clearly open, or the
            # model slowly learns that obstacles are floor.
            if args.adapt > 0 and regs[1] > go_px * 1.2:
                floor.learn(frame)

            # PATH 2. Stop somewhere clear, look up, say what is in the
            # room, then carry on. Blocking here is fine and nowhere
            # else is: the tank is stopped, so the watchdog releasing
            # the relays is exactly what should happen while a survey
            # runs long.
            if (detector is not None and args.survey_every
                    and move == "forward"
                    and now - last_survey > args.survey_every):
                last_survey = now
                say("  survey: stopping to look around ...")
                try:
                    if car is not None and driving:
                        car.stop()
                        time.sleep(0.4)          # let it actually settle
                    if car is not None:
                        car.camera_survey()
                    shot = shrink(cam.frame(), args.scale)
                    found = detector.run_sync(shot, "survey")
                    if found:
                        for d in reporter.fresh(found, time.monotonic()):
                            say(f"  room: {d.label} ({d.confidence:.0%})")
                        say(f"  survey done, {len(found)} object(s), "
                            f"{detector.last_ms:.0f} ms")
                    else:
                        say("  survey done, nothing recognised")
                except Exception as e:
                    say(f"  survey failed: {e}")
                finally:
                    if car is not None:
                        car.mast_hold(Car.MAST_UP)
                    # Force a fresh decision rather than resuming on a
                    # vote taken before the tank stopped.
                    smoother.buf.clear()
                    smoother.current = "stop"
                    next_tick = time.monotonic() + period
                    last_survey = time.monotonic()

            # Restarts are shown inline rather than only at the end.
            # A bridge that reboots mid-run is the difference between
            # a car that is being driven and one that is coasting on
            # its last order, and you want to see that as it happens.
            resets = f"  RESETS {car.resets}" if car is not None and car.resets else ""
            if car is not None and car.last_command_held:
                resets += f"  HELD {car.cooldown_remaining():.1f}s"
            # The reason matters more than the move. A reverse is only
            # ever one of four decisions, and which one says whether to
            # go and look at the mask or at the tank's speed.
            print(
                f"\rL {regs[0]:5.0f}  C {regs[1]:5.0f}  R {regs[2]:5.0f}   "
                f"go>{go_px:.0f}   {move:<15} {policy.reason:<18}"
                f"back x{policy.reversals}{resets}",
                end="",
            )
            sys.stdout.flush()

            # PATH 1. Offer the driving view to the detector and print
            # anything small enough to be a thing ON the floor rather
            # than a wall. Submitting never blocks; a frame arriving
            # while the model is busy replaces the waiting one.
            if detector is not None:
                if now - last_detect > args.detect_every:
                    last_detect = now
                    detector.submit(frame, "floor")
                for found in detector.poll():
                    floor_dets = [d for d in found if d.context == "floor"]
                    shown_dets, shown_at = floor_dets, now
                    small = [d for d in floor_dets
                             if d.area_frac <= args.small_object]
                    for d in reporter.fresh(small, now):
                        say(f"  saw {d.label} ({d.confidence:.0%}) on the floor"
                            f"  [{detector.last_ms:.0f} ms]")

                # Expire them rather than leaving a box hanging over
                # ground the tank drove past ten seconds ago.
                if shown_dets and now - shown_at > max(2.0, args.detect_every * 2):
                    shown_dets = []

            # One annotate() serves both the file and the stream, since
            # drawing it twice would double the cost for no gain.
            want_file = args.debug_image and now - last_debug > 1.0
            want_view = view is not None and now - last_view > view_period
            if want_file or want_view:
                shown = annotate(frame, mask, prof, regs, move, marks,
                                 dets=shown_dets, small_max=args.small_object,
                                 det_age=now - shown_at)
                if want_file:
                    last_debug = now
                    cv2.imwrite(args.debug_image, shown)
                if want_view:
                    last_view = now
                    view.update(shown)

    except KeyboardInterrupt:
        pass
    finally:
        # Camera first: lowering the mast is a bridge command, so the
        # bridge has to still be open when it goes out.
        if detector is not None:
            detector.close()
        if view is not None:
            view.close()
        cam.close()
        if car is not None:
            car.close()
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
# Backs into things
#   Lower --reverse-for. Reversing is blind — the camera faces the
#   other way and there is no rear sensor — so it is capped at a
#   fixed time rather than run until the way ahead clears.
#
# Sees the obstacle, decides to turn, and hits it anyway
#   Look at --cooldown before anything else. It defaults to 2s
#   because the contacts need that long to settle under teleop, and
#   during it the car holds its last direction no matter what the
#   camera says. Two seconds at 0.4 m/s is 80 cm of committed
#   travel, which is further than this can see in front of itself.
#
#   Perception cannot fix that; it is not a perception problem. Cut
#   --cooldown as far as the relays tolerate, and cut the speed to
#   match whatever is left. HELD in the status line shows when a
#   decision is being sat on.
#
# Drifts from working to useless over several minutes
#   The floor model is colours, learned once. If you passed
#   --no-lock-exposure, auto-exposure is changing those colours out
#   from under it. Drop the flag.
#
# Circles forever in open space
#   Lower --go, or check the camera is angled down far enough that
#   the far wall is not permanently in view.
#
# Drives into things the moment it starts
#   It learned the floor while pointing at an obstacle. Restart on
#   clear ground.
# =============================================================
