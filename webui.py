#!/usr/bin/env python3
"""
One page that drives the tank, roams it, and tunes it.

    python3 webui.py --args-from cmd.txt
    python3 webui.py --threshold 14 --go 0.62 --flatten 75 ...

Then open http://<pi>:8080/ , or over the tunnel already in use:

    ssh -N -L 8080:localhost:8080 kitchen-helper@rover.local
    http://localhost:8080/

-------------------------------------------------------------
WHY THIS EXISTS

roam.py and teleop.py each open the camera and the serial port, so
only one of them can run. Driving somewhere by hand and then letting
it roam meant killing one process and starting the other from a
different command line, and any tuning learned in between had to be
retyped from memory.

This owns the hardware once and switches MODE instead. The control
loop never stops: the same perception runs in every mode and the same
frame is published, so the picture on the page is the picture the
policy is deciding from, whether or not the policy is being listened
to. Watching it while driving by hand is how you find out what roam
would have done there.

-------------------------------------------------------------
EVERY MANUAL CONTROL IS DEAD-MAN

A browser is a worse place to drive from than a terminal on the same
machine: the tab can close, the phone can sleep, the WiFi can drop,
and none of those tell the server anything at all. So a manual order
is not a state that latches, it is a lease. The page renews it while
the control is held and the tank stops on its own the moment the
renewals stop arriving.

Same reasoning as the firmware watchdog one layer down, and it is
here for the same reason: this tank has no bumper, so nothing else
notices that the operator has gone.

A browser CAN see a key release, unlike a terminal, so holding to
drive here is exact rather than inferred. teleop.py's --arc-hold
guesswork is not needed and not used.

-------------------------------------------------------------
ROAM IS NOT ARMED BY THE CLICK THAT SELECTS IT

Choosing roam only selects it. It starts on a second, separate press,
because the mode buttons sit next to each other on a phone screen and
the difference between two of them is a tank driving off across the
room.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from car import BridgeError, Car, SoftArc, boot_warning
from record import CAMERA_ROTATION, Camera
from roam import (LowBattery, add_lowbattery_args,
                  FloorModel, Policy, Smoother, add_detect_args,
                  add_perception_args, add_view_args, annotate, blocking_boxes,
                  free_profile, marks_for, perceive, shrink, tracks_for)

# Moves the page may ask for by hand. A deliberate subset of what the
# policy can produce: no soft_back_*, because a reverse arc is what a
# tank does when it has decided it is stuck, not something to hand to
# someone who can see the picture and back out themselves.
MANUAL_MOVES = ("forward", "reverse", "arc_left", "arc_right",
                "soft_arc_left", "soft_arc_right", "stop")

MODES = ("stop", "manual", "roam")

# Changing these cannot take effect on the next tick. The first four
# describe a histogram that would have to be rebuilt; the rest are
# fixed when the camera or the bridge is opened.
NEEDS_RELEARN = {"bins", "flatten", "channels", "floor_model"}
NEEDS_RESTART = {"rotate", "no_lock_exposure", "detect", "detect_threads",
                 "port", "web", "no_servo", "dry_run"}


def _coerce(raw, like):
    """Turn a form string into whatever the existing value already is."""
    if isinstance(like, bool):
        return str(raw).lower() in ("1", "true", "on", "yes")
    if raw is None or raw == "":
        return None
    if isinstance(like, int) and not isinstance(like, bool):
        return int(float(raw))
    if isinstance(like, float):
        return float(raw)
    if like is None:
        # Was unset, so there is nothing to copy the type from. Numbers
        # are far more common here than paths, so try that first.
        try:
            return float(raw) if "." in str(raw) else int(raw)
        except ValueError:
            return str(raw)
    return str(raw)


class FrameSlot:
    """One latest-frame slot, served to any number of browsers.

    The same single-slot design as stream.py, reimplemented here for
    one reason: MJPEGStreamer owns an HTTP server of its own, and two
    servers means two ports, which means a second SSH tunnel. The page
    and the picture belong on the same one.

    Encoding happens on the CALLER's thread and update() returns as
    soon as the bytes exist. A viewer on a slow phone misses frames; it
    cannot slow the control loop, and therefore cannot cause the bridge
    watchdog to release the relays mid-drive.
    """

    def __init__(self, quality: int = 70):
        self.quality = quality
        self._jpeg = None
        self._seq = 0
        self._closing = False
        self._cv = threading.Condition()

    def update(self, bgr) -> None:
        ok, buf = cv2.imencode(".jpg", bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self._cv:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cv.notify_all()

    def wait(self, since: int):
        with self._cv:
            while self._seq <= since and not self._closing:
                self._cv.wait(timeout=1.0)
            if self._closing:
                return None, since
            return self._jpeg, self._seq

    def close(self) -> None:
        with self._cv:
            self._closing = True
            self._cv.notify_all()


class Controller:
    """Owns the hardware and runs one loop for every mode."""

    def __init__(self, args, car, cam, floor, driving: bool):
        self.args = args
        self.car = car
        self.cam = cam
        self.floor = floor
        self.driving = driving

        self.slot = FrameSlot(quality=args.stream_quality)
        self.lock = threading.Lock()
        self._stop = threading.Event()

        self.mode = "stop"
        self.manual_move = None
        self.manual_until = 0.0

        self._pending = None
        self._relearn = False
        self.notice = ""

        self.h = shrink(cam.frame(), args.scale).shape[0]
        self._build()

        self.detector = self.reporter = None
        if args.detect:
            try:
                from detect import Detector, Reporter
                self.detector = Detector(args.detect, conf=args.detect_conf,
                                         threads=args.detect_threads)
                self.reporter = Reporter(cooldown=args.report_cooldown)
            except Exception as e:
                self.notice = f"no detector: {e}"

        self.stat = {"move": "stop", "reason": "idle", "regs": [0, 0, 0],
                     "relays": "", "loop_fps": 0.0, "detections": []}
        self._thread = threading.Thread(target=self._loop, name="control",
                                        daemon=True)

    # ------------------------------------------------------- config

    def _build(self) -> None:
        """(Re)make everything derived from a number the page can change."""
        a, h = self.args, self.h
        self.go_px = a.go * h
        self.lowbat = LowBattery(a.low_battery, a.lowbat_runup)
        self.policy = Policy(self.go_px, a.commit_above * h, a.even_above * h,
                             a.turn_margin * h, a.stuck_after, a.reverse_for,
                             soft_margin=a.soft_margin * h,
                             back_below=a.back_below, dodge_min=a.dodge_min,
                             max_turn=a.max_turn, straighten_for=a.straighten_for)
        self.soft = SoftArc(period=a.soft_period, duty=a.soft_duty)
        self.smoother = Smoother(a.window, a.min_interval)
        self.marks = marks_for(a, h)
        if self.car is not None:
            self.car.command_cooldown = a.cooldown
            self.car.reverse_cooldown = max(a.reverse_cooldown, a.cooldown)

    def submit_config(self, values: dict) -> None:
        with self.lock:
            self._pending = values

    def request_relearn(self) -> None:
        with self.lock:
            self._relearn = True

    def _apply(self, values: dict) -> None:
        changed, deferred = [], []
        for dest, raw in values.items():
            if not hasattr(self.args, dest):
                continue
            old = getattr(self.args, dest)
            try:
                new = _coerce(raw, old)
            except (TypeError, ValueError):
                continue
            if new == old:
                continue
            setattr(self.args, dest, new)
            (deferred if dest in NEEDS_RESTART else changed).append(dest)
        self._build()

        bits = []
        if changed:
            bits.append(f"applied {len(changed)}: {', '.join(changed[:8])}")
        if deferred:
            bits.append(f"needs a restart to take effect: {', '.join(deferred)}")
        self.notice = "  |  ".join(bits) if bits else "nothing changed"

    # -------------------------------------------------------- modes

    def set_mode(self, mode: str) -> str:
        if mode not in MODES:
            return self.mode
        with self.lock:
            self.mode = mode
            self.manual_move = None
            self.manual_until = 0.0
            # Never carry a vote across a mode change. The buffer was
            # filled while something else was driving, so letting it
            # decide roaming's first tick means starting on evidence
            # about a situation that no longer exists.
            self.smoother.buf.clear()
            self.smoother.current = "stop"
        if self.car is not None and self.driving:
            try:
                self.car.stop()
            except BridgeError:
                pass
        return mode

    def hold(self, move: str) -> bool:
        """Renew a manual lease. False if it was refused."""
        if move not in MANUAL_MOVES:
            return False
        with self.lock:
            if self.mode != "manual":
                return False
            self.manual_move = None if move == "stop" else move
            self.manual_until = time.monotonic() + self.args.manual_hold
        return True

    MAST_STEP = 5

    def mast(self, action) -> dict:
        """Aim the camera. Never touches the tracks.

        Allowed in every mode including roam, deliberately: the mast
        angle is the one perception setting that cannot be judged from
        a number, and having to stop roaming to correct it is how it
        stays wrong.
        """
        if self.car is None:
            return {"ok": False, "why": "no bridge"}
        try:
            if action == "up":
                self.car.camera_up(settle=0.0)
            elif action == "limp":
                self.car.mast(-1)
            elif action in ("raise", "lower"):
                self.car.mast_nudge(self.MAST_STEP if action == "raise"
                                    else -self.MAST_STEP)
            else:
                return {"ok": False, "why": f"unknown action {action!r}"}
        except BridgeError as e:
            return {"ok": False, "why": str(e)}
        angle = self.car.mast_angle
        limp = angle is None or angle < 0
        self.notice = "camera limp" if limp else f"camera {angle} deg"
        # None, matching state(), so the page has one thing to test for
        # rather than two spellings of the same condition.
        return {"ok": True, "mast": None if limp else angle}

    def release(self) -> None:
        with self.lock:
            self.manual_move = None
            self.manual_until = 0.0

    # --------------------------------------------------------- loop

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.slot.close()

    def _decide(self, regs, now):
        with self.lock:
            mode, move, until = self.mode, self.manual_move, self.manual_until

        if mode == "roam":
            move = self.smoother.update(self.policy.decide(*regs, now), now)
            move = self.lowbat.filter(move, now)
            return move, (self.lowbat.reason or self.policy.reason)

        if mode == "manual":
            if move is None:
                return "stop", "released"
            if now > until:
                # The page stopped renewing. It may have been closed,
                # slept, or lost the network — all of which look the
                # same from here, and all of which mean stop.
                return "stop", "LEASE EXPIRED"
            return move, "held"

        return "stop", "idle"

    def _loop(self) -> None:
        period = 1.0 / self.args.fps
        next_tick = time.monotonic()
        last_view = last_detect = 0.0
        last_error = None
        shown, shown_at = [], 0.0
        ticks, ticked_at, loop_fps = 0, time.monotonic(), 0.0

        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(0.005)
                continue
            next_tick = max(now + period, next_tick + period)

            # Config is applied HERE rather than in the HTTP thread, so
            # a tick never reads half of one setting and half of the
            # next. It also means a bad number cannot land between
            # perceiving and deciding.
            with self.lock:
                pending, self._pending = self._pending, None
                relearn, self._relearn = self._relearn, False
            if pending:
                self._apply(pending)
                period = 1.0 / self.args.fps

            if self.detector is not None:
                for found in self.detector.poll():
                    shown = [d for d in found if d.context == "floor"]
                    shown_at = now
                if shown and now - shown_at > max(2.0, self.args.detect_every * 2):
                    shown = []

            blocks = (blocking_boxes(shown, now - shown_at, self.args)
                      if self.detector is not None else [])
            frame, mask, prof, regs = perceive(self.cam.frame(), self.floor,
                                               self.args, blocks)

            if relearn:
                self.floor.learn(frame)
                self.notice = f"floor relearned — {getattr(self.floor, 'why', '')}"

            if (self.detector is not None
                    and now - last_detect > self.args.detect_every):
                last_detect = now
                self.detector.submit(frame, "floor")

            move, reason = self._decide(regs, now)
            decision = tracks_for(move, now, self.soft)

            relays = ""
            if self.car is not None and self.driving:
                try:
                    reply = self.car.drive(*decision)
                    relays = " ".join(reply.split()[1:5])
                    last_error = None
                except BridgeError as e:
                    if str(e) != last_error:
                        last_error = str(e)
                        self.notice = f"bridge: {e}"

            ticks += 1
            if now - ticked_at >= 1.0:
                loop_fps = ticks / (now - ticked_at)
                ticks, ticked_at = 0, now

            self.stat = {
                "move": move, "reason": reason,
                "regs": [round(float(x)) for x in regs],
                "relays": relays, "loop_fps": round(loop_fps, 1),
                "detections": [f"{d.label} {d.confidence:.0%}" for d in shown],
            }

            if now - last_view >= 1.0 / max(0.1, self.args.stream_fps):
                last_view = now
                try:
                    self.slot.update(annotate(
                        frame, mask, prof, regs, move, self.marks, shown,
                        self.args.small_object, now - shown_at, blocks,
                        getattr(self.policy, "reason", "")))
                except TypeError:
                    # annotate's tail arguments have moved before now.
                    # A live view is worth more than an exact overlay,
                    # so fall back rather than kill the control loop.
                    self.slot.update(frame)

    # ------------------------------------------------------- report

    def state(self) -> dict:
        with self.lock:
            mode, until = self.mode, self.manual_until
            notice, self.notice = self.notice, ""
        car = self.car
        return {
            "mode": mode,
            "driving": self.driving,
            "lease": round(max(0.0, until - time.monotonic()), 2),
            "go_px": round(self.go_px),
            "reversals": getattr(self.policy, "reversals", 0),
            "resets": getattr(car, "resets", 0) if car else 0,
            "held": bool(car and car.last_command_held),
            "lowbat": self.lowbat.enabled,
            "mast": (car.mast_angle if car and car.mast_angle is not None
                     and car.mast_angle >= 0 else None),
            "notice": notice,
            **self.stat,
        }


# ------------------------------------------------------------- config form


def describe(parser, args) -> list:
    """Every flag, its current value, and which group it belongs in."""
    groups = {}
    for fn, name in ((add_perception_args, "perception"),
                     (add_detect_args, "detector"),
                     (add_view_args, "view")):
        probe = argparse.ArgumentParser(add_help=False)
        fn(probe)
        for a in probe._actions:
            groups[a.dest] = name

    out = []
    for a in parser._actions:
        # args_from is how this page was started, not something it can
        # change afterwards. stream comes in with the shared view flags
        # and would be a dead field here: this page already serves the
        # picture on --web, and a second port is the thing it exists to
        # avoid.
        if a.dest in ("help", "args_from", "stream"):
            continue
        value = getattr(args, a.dest, None)
        kind = "bool" if isinstance(value, bool) else "text"
        out.append({
            "dest": a.dest,
            "flag": a.option_strings[0] if a.option_strings else a.dest,
            "group": groups.get(a.dest, "driving"),
            "value": "" if value is None else value,
            "kind": kind,
            "choices": list(a.choices) if a.choices else None,
            "help": (a.help or "").replace("  ", " "),
            "live": a.dest not in NEEDS_RESTART,
            "relearn": a.dest in NEEDS_RELEARN,
        })
    out.sort(key=lambda d: (d["group"] != "driving", d["group"], d["dest"]))
    return out


PAGE = Path(__file__).with_name("webui.html")


class Handler(BaseHTTPRequestHandler):
    ctl: Controller = None
    fields: list = None

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                          # never scribble over the console

    # ------------------------------------------------------ helpers

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    # ---------------------------------------------------------- GET

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                body = PAGE.read_bytes()
            except OSError as e:
                self._send(500, f"cannot read {PAGE.name}: {e}".encode())
                return
            self._send(200, body, "text/html; charset=utf-8")
            return

        if self.path == "/api/state":
            self._json(self.ctl.state())
            return

        if self.path == "/api/config":
            self._json({"fields": self.fields})
            return

        if self.path == "/stream.mjpg":
            self._stream()
            return

        self._send(404, b"no")

    def _stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        last = -1
        try:
            while True:
                jpeg, seq = self.ctl.slot.wait(last)
                if jpeg is None:
                    return
                last = seq
                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                      # viewer closed the tab, which is fine

    # --------------------------------------------------------- POST

    def do_POST(self):
        body = self._body()

        if self.path == "/api/mode":
            self._json({"mode": self.ctl.set_mode(body.get("mode", "stop"))})
            return

        if self.path == "/api/hold":
            ok = self.ctl.hold(body.get("move", "stop"))
            self._json({"ok": ok, **self.ctl.state()})
            return

        if self.path == "/api/lowbat":
            on = bool(body.get("on", False))
            self.ctl.lowbat.enabled = on
            # Momentum cannot be assumed across the switch: turn it on
            # mid-roll and the tank may or may not still be moving, so
            # make the next turn earn its run-up.
            self.ctl.lowbat.rolling_since = None
            self.ctl.notice = (f"low battery ON — turns need "
                               f"{self.ctl.lowbat.run_up:.1f}s of straight first"
                               if on else "low battery off")
            self._json({"ok": True, "lowbat": on})
            return

        if self.path == "/api/mast":
            self._json(self.ctl.mast(body.get("action", "")))
            return

        if self.path == "/api/release":
            self.ctl.release()
            self._json({"ok": True})
            return

        if self.path == "/api/config":
            self.ctl.submit_config(body.get("values", {}))
            self._json({"ok": True})
            return

        if self.path == "/api/relearn":
            self.ctl.request_relearn()
            self._json({"ok": True})
            return

        self._send(404, b"no")


# ------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    """roam's own flags, plus the few this page adds.

    Deliberately the same parser roam.main() builds. A page that
    offered a different set, or the same names with different
    defaults, would be a second place for the tuning to live and the
    two would disagree within a week.
    """
    ap = argparse.ArgumentParser()
    add_perception_args(ap)
    add_detect_args(ap)
    add_view_args(ap)
    ap.add_argument("--args-from", default=None, metavar="FILE",
                    help="read defaults from a file of flags, such as cmd.txt. "
                         "A leading 'python roam.py' is ignored, so the line "
                         "already in use can be pasted unedited")
    ap.add_argument("--web", type=int, default=8080, help="port for this page")
    ap.add_argument("--stream-quality", type=int, default=70,
                    help="JPEG quality of the live view")
    ap.add_argument("--manual-hold", type=float, default=0.6,
                    help="seconds a manual order survives without being "
                         "renewed by the page. The tank stops when a browser "
                         "goes quiet, whatever the reason")
    add_lowbattery_args(ap)
    ap.add_argument("--port", default=None, help="ESP32 serial device")
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and show, but never drive")
    ap.add_argument("--fps", type=float, default=10.0, help="decisions per second")
    ap.add_argument("--turn-margin", type=float, default=0.05,
                    help="how much clearer one side must be, as a fraction")
    ap.add_argument("--no-servo", action="store_true",
                    help="do not raise the camera mast")
    ap.add_argument("--soft-margin", type=float, default=0.10,
                    help="how much clearer one side must be before drifting "
                         "away from the other while still going forward")
    ap.add_argument("--soft-period", type=float, default=1.2,
                    help="seconds per soft-arc cycle")
    ap.add_argument("--soft-duty", type=float, default=0.30,
                    help="fraction of each soft-arc cycle spent arcing")
    ap.add_argument("--dodge-min", type=float, default=0.35,
                    help="room a side needs, as a fraction of --go, before it "
                         "counts as a way past rather than the less bad wall")
    ap.add_argument("--back-below", type=float, default=0.22,
                    help="turn while reversing when centre clearance falls "
                         "below this fraction of --go")
    ap.add_argument("--window", type=int, default=5,
                    help="frames in the majority vote")
    ap.add_argument("--min-interval", type=float, default=0.2,
                    help="seconds between relay changes")
    ap.add_argument("--stuck-after", type=float, default=3.0,
                    help="seconds of turning before reversing")
    ap.add_argument("--max-turn", type=float, default=4.0,
                    help="longest any turning move may run; forward and "
                         "reverse are not capped")
    ap.add_argument("--straighten-for", type=float, default=1.0,
                    help="seconds of driving straight after a turn hits "
                         "--max-turn")
    ap.add_argument("--reverse-for", type=float, default=1.0,
                    help="seconds to reverse before trying to turn again")
    ap.add_argument("--command-timeout", type=float, default=None,
                    help="seconds of silence from the loop before the bridge "
                         "watchdog may release the relays")
    ap.add_argument("--reverse-cooldown", type=float, default=2.0,
                    help="seconds before a motor may be reversed again")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds before any gentler change is allowed")
    return ap


def load_defaults(path: str) -> list:
    """Flags out of a cmd.txt-style file.

    Everything up to and including a .py is dropped, so the line
    already being used can be pasted in whole. Blank lines and
    anything after them are ignored, which is what makes it safe to
    keep the ssh tunnel command in the same file.
    """
    tokens = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line)
        if not parts:
            continue
        if parts[0] in ("python", "python3") or parts[0].endswith(".py"):
            while parts and not parts[0].startswith("-"):
                parts.pop(0)
        elif not parts[0].startswith("-"):
            continue              # ssh line, or anything else not for us
        tokens += parts
    return tokens


def main() -> int:
    ap = build_parser()

    # Two passes: the file supplies defaults, the command line then
    # overrides them. That way --args-from cmd.txt reproduces the last
    # session exactly, and one flag can still be changed on top of it
    # without editing the file.
    pre, _ = ap.parse_known_args()
    if pre.args_from:
        try:
            tokens = load_defaults(pre.args_from)
        except OSError as e:
            print(f"could not read {pre.args_from}: {e}", file=sys.stderr)
            return 1

        # parse_KNOWN_args, because the file is a roam command line and
        # roam has flags this page does not — --debug-image being one,
        # since the live view replaces it. Refusing to start over a
        # flag that simply does not apply here would mean keeping a
        # second copy of the line, which is what --args-from exists to
        # avoid. They are named rather than dropped in silence.
        try:
            seed, unknown = ap.parse_known_args(tokens)
        except SystemExit:
            print(f"{pre.args_from} has a flag this page cannot read",
                  file=sys.stderr)
            return 1
        if unknown:
            print(f"  ignoring, not used here: {' '.join(unknown)}")

        # A file that names a stream port is naming the port a tunnel
        # is already pointed at. Serve the page there too, so the
        # tunnel already open keeps working without being retyped.
        if getattr(seed, "stream", 0) and "--web" not in sys.argv:
            seed.web = seed.stream
        ap.set_defaults(**vars(seed))

    args = ap.parse_args()

    command_timeout = args.command_timeout
    if command_timeout is None:
        command_timeout = max(0.5, 5.0 / args.fps)

    driving = not args.dry_run
    car = None
    if driving or not args.no_servo:
        try:
            car = Car(port=args.port, command_timeout=command_timeout,
                      command_cooldown=args.cooldown,
                      reverse_cooldown=args.reverse_cooldown)
            print(f"bridge on {car.port}")
            warning = boot_warning(car.boot_reason)
            if warning:
                print(f"\n  !! {warning}\n")
        except BridgeError as e:
            if driving:
                print(f"could not open the bridge: {e}", file=sys.stderr)
                return 1
            print(f"  !! no bridge, so no mast: {e}", file=sys.stderr)

    cam = Camera(lock_exposure=not args.no_lock_exposure, rotate=args.rotate,
                 mast=None if args.no_servo else car)

    if args.floor_model:
        floor = FloorModel.load(args.floor_model, smooth=args.adapt)
    else:
        print("\nlearning the floor — keep a metre of clear ground ahead")
        time.sleep(1.0)
        floor = FloorModel(smooth=args.adapt, mode=args.channels,
                           bins=args.bins, flatten=args.flatten)
        floor.learn(shrink(cam.frame(), args.scale))
    print(f"learned: {getattr(floor, 'why', floor.chosen)}")

    ctl = Controller(args, car, cam, floor, driving)
    ctl.start()

    Handler.ctl = ctl
    Handler.fields = describe(ap, args)

    try:
        server = ThreadingHTTPServer(("", args.web), Handler)
    except OSError as e:
        print(f"cannot serve on port {args.web}: {e}", file=sys.stderr)
        ctl.close()
        return 1
    server.daemon_threads = True

    port = server.server_address[1]
    print(f"\n  control page on http://<this-pi>:{port}/")
    print(f"  over the tunnel already in use:")
    print(f"    ssh -N -L {port}:localhost:{port} <user>@<this-pi>")
    print(f"    http://localhost:{port}/")
    print(f"\n  mode starts at STOP. Nothing moves until the page says so.")
    print(f"  ctrl-c here stops the tank and closes the port.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        ctl.close()
        if car is not None:
            car.close()
        cam.close()
        print("\nstopped, port closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
