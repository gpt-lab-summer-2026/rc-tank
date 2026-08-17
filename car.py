"""
Client for the ESP32 relay bridge.

The ESP32 sets relay states and nothing else. Everything about how
the car drives is decided here.

    from car import Car

    with Car() as car:
        car.forward()
        time.sleep(1)
        car.spin_left()
        time.sleep(0.5)
        car.stop()

If a turn comes out mirrored, the sides are wired the other way
round to what the code assumes — see SWAP_SIDES below.

Needs pyserial:  pip install pyserial
"""

from __future__ import annotations

import atexit
import sys
import threading
import time
from typing import Callable, Optional

import serial
from serial.tools import list_ports

__all__ = ["Car", "BridgeError", "find_port", "boot_warning"]

# Opening the serial port resets most ESP32 boards, so one of these
# is expected at the start of every run.
_NORMAL_BOOT = ("power-on", "external-pin", "software")


def boot_warning(reason: Optional[str]) -> Optional[str]:
    """Say something if the bridge's last restart was not routine.

    Anything outside _NORMAL_BOOT happened on its own. On this
    hardware that almost always means the motors pulled the supply
    down far enough to take the ESP32 with them.
    """
    if not reason or reason in _NORMAL_BOOT:
        return None
    if reason == "BROWNOUT":
        return (
            "bridge last restarted from BROWNOUT — the supply sagged far enough\n"
            "  to reset the ESP32. While it reboots nothing is driving the relay\n"
            "  pins, so the motors hold whatever they were last given and no\n"
            "  watchdog is running to release them. Fit the 10k pull-ups, and\n"
            "  stop powering the relay coils from the same 5V as the ESP32."
        )
    return f"bridge last restarted from {reason} — it did not do that on request."


def _default_event(msg: str) -> None:
    """Where bridge events go when nobody says otherwise.

    Loud on purpose. These are the lines that tell you the ESP32
    restarted underneath you, and the previous version of this file
    threw them away before anyone could read them.
    """
    print(f"\n[bridge] {msg}", file=sys.stderr, flush=True)


class BridgeError(RuntimeError):
    """The bridge replied with an error, or did not reply at all."""


# USB-serial chips commonly found on ESP32 dev boards.
_LIKELY = ("CP210", "CH340", "CH910", "FTDI", "FT232", "ESP32", "USB Serial")


def find_port() -> Optional[str]:
    """Best guess at which serial device is the ESP32."""
    ports = list(list_ports.comports())
    for p in ports:
        blob = f"{p.description} {p.manufacturer} {p.product}"
        if any(k.lower() in blob.lower() for k in _LIKELY):
            return p.device
    for p in ports:                      # fall back to the usual suspects
        if "ttyUSB" in p.device or "ttyACM" in p.device:
            return p.device
    return None


# Per motor: +1 forward, 0 stop, -1 reverse -> (relay A, relay B)
_DIR = {1: (1, 0), 0: (0, 0), -1: (0, 1)}

# ------------------------------------------------------------------
# WHICH RELAY PAIR IS WHICH TRACK
#
# The firmware calls IN1/IN2 "motor 1" and IN3/IN4 "motor 2". On this
# tank motor 1 is the RIGHT track, so drive(left, right) has to hand
# its arguments over the other way round.
#
# Without the swap, forward and backward still look perfectly correct
# — both tracks get the same command — and only turns come out
# mirrored. That is what makes this worth a named constant rather
# than a quiet fix: the symptom points at the steering logic, and the
# cause is two connectors.
#
# Set False if the motor leads are ever swapped at the relay board,
# which fixes the same problem in copper.
SWAP_SIDES = True


class SoftArc:
    """A turn gentler than the relays can express in any single state.

    Each track has three states, so the widest turn available as a
    held command is a full arc — one track stopped — and on this
    chassis that is still a hard turn. Anything gentler has to be
    built in TIME rather than in voltage: hold forward for most of a
    cycle, interrupt it with a short arc pulse, repeat.

    -------------------------------------------------------------
    THIS SPENDS CONTACT LIFE, AND PLAN.md ALREADY SAID SO

    PLAN.md rejects pulse-and-coast outright, pricing it at roughly
    triple the contact wear and a dead relay board inside three hours.
    That verdict stands for what it judged: duty-cycling every move,
    continuously, to fake speed control.

    A soft arc is a narrower bargain than the one it refused. Only the
    inner track's pair switches, so a cycle costs two contact
    operations rather than eight, and it runs only while actually
    correcting course rather than the whole time the tank is moving.
    At the default 1.2 s period that is about 1.7 operations per
    second while active, against zero for a hard arc simply held.

    Narrower, but not free, and bought to paper over something the
    hardware does not have. Use it to correct course on open floor.
    Do not use it to go slower — that is the exact case PLAN.md
    already costed and turned down.

    -------------------------------------------------------------
    The phase free-runs off the clock rather than being stepped, so it
    holds no state, cannot drift, and answers identically for every
    caller asking about the same instant.
    """

    def __init__(self, period: float = 1.2, duty: float = 0.30):
        self.period = period
        self.duty = duty

    def tracks(self, turn_right: bool, now: float,
               reverse: bool = False) -> tuple[int, int]:
        """The track pair for this instant of the cycle.

        reverse runs the same cycle backwards: both tracks reversing
        between pulses instead of both driving. That is the only way
        this chassis turns while backing — holding ONE track in
        reverse against a stopped one barely turns it at all, whereas
        interrupting a reverse both tracks are already committed to
        carries enough momentum through the pulse to swing the nose.

        Note the sides invert. Going forward the inside track idles;
        going backwards the OTHER one does, because the nose swings
        the opposite way when the tracks pull the opposite way.
        """
        base = -1 if reverse else 1
        if (now % self.period) < self.period * self.duty:
            return (base, 0) if turn_right != reverse else (0, base)
        return (base, base)

    @property
    def ops_per_second(self) -> float:
        """Contact operations per second while this is running."""
        return 2.0 / self.period

    def swallowed_by(self, cooldown: float) -> bool:
        """Would Car's cooldown hold the pulses instead of passing them?

        A cooldown at least as long as the shorter half of the cycle
        turns every pulse into a resend of the previous state, so the
        arc silently never happens and the tank drives straight. Worth
        catching at startup rather than in a corridor.
        """
        shorter = self.period * min(self.duty, 1.0 - self.duty)
        return cooldown > 0 and cooldown >= shorter


class Car:
    """Serial link to the relay bridge.

    Opening the port resets most ESP32 boards, so the constructor
    waits for it to come back before talking to it.

    command_timeout is what makes the firmware watchdog mean anything.
    Set it, and the keepalive stops refreshing the bridge once that
    many seconds have passed without a command from the caller, so a
    control loop that hangs takes the motors down with it. Leave it
    None and the keepalive holds the last state indefinitely, which is
    what you want when a human is driving and nothing else is.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 115200,
        timeout: float = 0.3,
        keepalive: bool = True,
        boot_delay: float = 2.0,
        command_timeout: Optional[float] = None,
        command_cooldown: float = 0.0,
        reverse_cooldown: float = 0.0,
        swap_sides: bool = SWAP_SIDES,
        on_event: Optional[Callable[[str], None]] = _default_event,
    ):
        port = port or find_port()
        if port is None:
            raise BridgeError("no serial port found — pass port= explicitly")

        self.port = port
        self._lock = threading.Lock()

        # Anything the bridge says without being asked, and a count of
        # how many times it has restarted. A restart mid-session is the
        # single most useful thing this class can tell you.
        self.on_event = on_event
        self.events: list[str] = []
        self.resets = 0
        self.boot_reason: Optional[str] = None

        self._ser = serial.Serial(port, baud, timeout=timeout)

        time.sleep(boot_delay)            # board resets when the port opens

        now = time.monotonic()
        self._last_tx = now
        self._last_command_at = now
        self._last_state: Optional[tuple[int, int, int, int]] = None
        self._last_uptime: Optional[int] = None
        self._command_timeout = command_timeout

        # Contacts take real time to settle, and commands arriving
        # faster than that leave the state machine chasing itself.
        # Changes are refused inside the cooldown; unchanged states
        # always pass, so the keepalive still reaches the bridge.
        #
        # Two limits, because not every change costs the same. A
        # reversal gets reverse_cooldown; everything gentler — arcs,
        # stops, starts — gets command_cooldown, which can sensibly
        # be zero.
        self.command_cooldown = command_cooldown
        self.reverse_cooldown = max(reverse_cooldown, command_cooldown)
        self.held = 0
        self.last_command_held = False
        self._changed_at = 0.0
        self.mast_angle: Optional[int] = None
        self._mast_atexit = False
        self._reversed_at = 0.0

        self.swap_sides = swap_sides
        self._watchdog_s = 0.4            # replaced by what the bridge reports
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # The boot banner is sitting in the buffer right now. Read it
        # rather than flushing it — resets is meant to count restarts
        # that happen while running, not the one we just caused.
        self._drain_async()
        self.resets = 0

        info = self.info()
        if "bridge" not in info:
            raise BridgeError(f"unexpected greeting: {info!r}")

        if keepalive:
            self._thread = threading.Thread(
                target=self._keepalive, name="car-keepalive", daemon=True
            )
            self._thread.start()

    # ---------------------------------------------------------- io

    # Lines the bridge sends unprompted. They start with their own
    # words precisely so a restart cannot be mistaken for the answer
    # to whatever was sent a moment earlier.
    _ASYNC = ("boot", "evt")

    def _emit(self, msg: str) -> None:
        self.events.append(msg)
        if self.on_event is not None:
            self.on_event(msg)

    def _note_async(self, line: str) -> None:
        if line.startswith("boot"):
            self.resets += 1
            self._emit(f"*** BRIDGE RESTARTED *** {line}")
        elif line.startswith("evt"):
            self._emit(line)
        else:
            self._emit(f"unexpected line: {line!r}")

    def _drain_async(self) -> None:
        """Read whatever is already waiting. Caller holds the lock."""
        while self._ser is not None and self._ser.in_waiting:
            stray = self._ser.readline().decode(errors="replace").strip()
            if stray:
                self._note_async(stray)

    def _note_uptime(self, reply: str) -> None:
        """Catch a restart the boot banner did not tell us about.

        The banner can be missed — swallowed by a flush, mangled by
        line noise, or sent while nothing was reading. Uptime cannot
        be missed: it only ever counts up, so it counting down means
        the chip started again.
        """
        for field in reply.split():
            if not field.startswith("up="):
                continue
            try:
                up = int(field[3:])
            except ValueError:
                return
            if self._last_uptime is not None and up < self._last_uptime:
                self.resets += 1
                self._emit(
                    f"*** BRIDGE RESTARTED *** uptime went "
                    f"{self._last_uptime} -> {up} ms (banner missed)"
                )
            self._last_uptime = up
            return

    def _send(self, line: str, expect: str = "ok") -> str:
        with self._lock:
            if self._ser is None:
                raise BridgeError("port is closed")
            try:
                # Whatever is already waiting was not asked for, so it
                # is either a restart or a watchdog notice. Both are
                # worth more than the microsecond saved by discarding
                # them, which is what this used to do.
                self._drain_async()

                self._ser.write((line + "\n").encode())
                self._ser.flush()

                reply = ""
                for _ in range(5):
                    reply = self._ser.readline().decode(errors="replace").strip()
                    if not reply:
                        break
                    if reply.startswith(self._ASYNC):
                        self._note_async(reply)
                        continue          # that was not the reply, keep reading
                    break
                self._last_tx = time.monotonic()
            except serial.SerialException as e:
                # pyserial raises this, not BridgeError, so without
                # this every caller's `except BridgeError` misses an
                # unplugged cable entirely.
                raise BridgeError(f"serial link failed: {e}") from e

        if not reply:
            raise BridgeError(f"no reply to {line!r}")
        if reply.startswith("err"):
            raise BridgeError(f"{line!r} -> {reply}")
        if not reply.startswith(expect):
            raise BridgeError(f"{line!r} -> wanted {expect!r}, got {reply!r}")
        self._note_uptime(reply)
        return reply

    def _note_info(self, info_line: str) -> None:
        """Pick the settings we care about out of an info reply."""
        for field in info_line.split():
            key, _, val = field.partition("=")
            if key == "watchdog":
                try:
                    self._watchdog_s = int(val) / 1000.0
                except ValueError:
                    pass
            elif key == "boot":
                self.boot_reason = val

    def _caller_is_alive(self) -> bool:
        """Has whoever owns this Car issued a command recently?"""
        if self._command_timeout is None:
            return True
        return time.monotonic() - self._last_command_at <= self._command_timeout

    def _keepalive(self) -> None:
        """Refresh the bridge when idle — but only while the caller lives.

        Without this, a slow inference step between commands looks
        to the ESP32 like the Pi has gone away.

        The catch is that a thread which refreshes regardless of what
        the rest of the program is doing defeats the watchdog it is
        working around. Nothing on this car watches for a collision,
        so the watchdog releasing the relays is the only thing between
        a hung control loop and driving until something stops it. Hence
        command_timeout: past it, this goes quiet and lets the watchdog
        do its job.

        It resends the last commanded state rather than a bare ping.
        The firmware ignores a state it has already applied, so this
        costs no contact operations, and a bridge that reset underneath
        us comes back doing what it was last told.
        """
        while not self._stop_evt.is_set():
            state = self._last_state
            if self._watchdog_s > 0 and state is not None and self._caller_is_alive():
                idle = time.monotonic() - self._last_tx
                if idle > self._watchdog_s / 3:
                    try:
                        self._apply(state)
                    except Exception:
                        pass              # close() will surface real failures
            self._stop_evt.wait(0.05)

    # ----------------------------------------------------- commands

    def _apply(self, state: tuple[int, int, int, int]) -> str:
        """Send a relay state without it counting as caller activity."""
        return self._send("R {} {} {} {}".format(*state))

    @staticmethod
    def _is_reversal(old: tuple[int, int, int, int],
                     new: tuple[int, int, int, int]) -> bool:
        """Does either motor turn one way and then the other?

        The distinction the cooldown cares about. Reversing a turning
        motor fights its own momentum and breaks a contact carrying
        the current that fight draws. Dropping a track to idle, or
        starting one from rest, does neither — an arc is a gentle
        thing and does not deserve the same wait.

        Matches how the firmware decides to insert its dead-time, so
        the two layers agree on what counts as harsh.
        """
        turning = ((1, 0), (0, 1))          # the two driving states
        for i in (0, 2):                    # each relay pair, one per motor
            was, now_ = old[i:i + 2], new[i:i + 2]
            if was in turning and now_ in turning and was != now_:
                return True
        return False

    def cooldown_remaining(self, state: Optional[tuple] = None) -> float:
        """Seconds until a change would be accepted.

        Each limit runs off its own clock, which is the point. The
        reversal limit is recovery time *after* a reversal, not a
        readiness check before one — a motor is no easier to reverse
        for having been driving a while, so making it wait on the
        last arc protects nothing and only makes the controls sticky.
        What is worth preventing is reversing again immediately, back
        and forth, while the contacts are still hot.

        Without an argument, reports the worst case.
        """
        now = time.monotonic()
        wait = self.command_cooldown - (now - self._changed_at)

        reversing = True
        if state is not None and self._last_state is not None:
            reversing = self._is_reversal(self._last_state, state)
        if reversing:
            wait = max(wait, self.reverse_cooldown - (now - self._reversed_at))
        return max(0.0, wait)

    def _set(self, state: tuple[int, int, int, int], force: bool = False) -> str:
        """Apply a relay state, subject to the cooldown.

        A held command is not silently dropped — the current state is
        resent instead. That keeps the bridge hearing from us, so the
        watchdog does not mistake a rate-limited operator for a dead
        one and release the relays mid-drive.
        """
        now = time.monotonic()
        self._last_command_at = now

        changing = self._last_state is not None and state != self._last_state
        if changing and not force and self.cooldown_remaining(state) > 0:
            self.held += 1
            self.last_command_held = True
            return self._apply(self._last_state)   # type: ignore[arg-type]

        self.last_command_held = False
        if changing and self._is_reversal(self._last_state, state):  # type: ignore[arg-type]
            self._reversed_at = now
        if state != self._last_state:
            self._changed_at = now
        self._last_state = state
        return self._apply(state)

    def relays(self, a: int, b: int, c: int, d: int) -> str:
        """Set IN1..IN4 directly. Returns the applied state line."""
        return self._set((int(a), int(b), int(c), int(d)))

    def _drive(self, left: int, right: int, force: bool = False) -> str:
        if left not in _DIR or right not in _DIR:
            raise ValueError("left and right must be -1, 0 or 1")
        if self.swap_sides:
            left, right = right, left
        a, b = _DIR[left]
        c, d = _DIR[right]
        return self._set((a, b, c, d), force=force)

    def drive(self, left: int, right: int) -> str:
        """Set each track to -1 (reverse), 0 (stop) or +1 (forward).

        left and right mean the tank's own left and right. See
        SWAP_SIDES for how those reach the relay pairs.
        """
        return self._drive(left, right)

    def soft_arc(self, side: str, seconds: float = 1.0,
                 reverse: bool = False) -> bool:
        """Steer by dropping one track briefly, then put it back.

        Cutting the inside track lets the outside one pull the nose
        round. Nothing reverses, so no contact has to break a
        reversing current and no gearbox is asked to change direction
        under load — the gentlest turn this drivetrain can make.

        The tracks go back to whatever they were doing before, so a
        turn is a correction to a course rather than a new state to
        remember to cancel. Returns False if the cooldown held it,
        leaving nothing changed.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        before = self._last_state

        # Idle the track that must travel less. Forward that is the
        # inside one; backing it is the other, since the nose swings
        # the other way when the tracks pull the other way.
        base = -1 if reverse else 1
        if (side == "right") != reverse:
            self.drive(base, 0)
        else:
            self.drive(0, base)
        if self.last_command_held:
            return False

        time.sleep(seconds)

        # Forced: the arc itself just started the clock, and making
        # the tracks wait it out would leave one of them idling with
        # nobody having asked for that.
        if before is None:
            self.stop()
        else:
            self._set(before, force=True)
        return True

    def forward(self) -> str:
        return self.drive(1, 1)

    def backward(self) -> str:
        return self.drive(-1, -1)

    def spin_left(self) -> str:
        """Both wheels opposed — turns on the spot, hardest on the contacts."""
        return self.drive(-1, 1)

    def spin_right(self) -> str:
        return self.drive(1, -1)

    def arc_left(self) -> str:
        """Inner track stopped — yaws left, and travels forward doing it."""
        return self.drive(0, 1)

    def arc_right(self) -> str:
        """Yaws right, travelling forward."""
        return self.drive(1, 0)

    # There is deliberately no held back_arc_left/right. Driving one
    # track in reverse against a stopped one barely turns this chassis
    # — it digs in rather than pivoting — so a reverse turn is a soft
    # arc instead: both tracks reversing, briefly interrupted. See
    # SoftArc.tracks(reverse=True) and soft_arc(reverse=True).
    #
    # The single-track reverse states still exist as relay states and
    # drive(-1, 0) will still produce one. Nothing here asks for one.

    # ------------------------------------------------- camera mast

    # Angles, not directions. The mast is at 0 stowed and 90 raised;
    # anything else in 0-180 is reachable and unused.
    MAST_DOWN = 0
    MAST_UP = 125

    # ------------------------------------------------------- buzzer

    # Tune indices, matching the TUNES table in the firmware. Keep the
    # two in step: the bridge validates the range but cannot tell a
    # wrong-but-valid index from a right one, so a mismatch here plays
    # the wrong program's tune rather than reporting anything.
    TUNE_TELEOP = 0
    TUNE_ROAM = 1
    TUNE_RECORD = 2

    def tune(self, n: int) -> str:
        """Play one of the firmware's tunes. Returns as it starts.

        The bridge plays it from its main loop rather than blocking in
        one, so this does not stall relay timing and does not hold the
        caller up either — by the time the reply lands the first note
        is sounding and the rest are the bridge's problem.
        """
        return self._send(f"T {int(n)}")

    def chirp(self, n: int) -> None:
        """tune(), for callers who would rather not be interrupted.

        A startup jingle is decoration. Nothing about a program that
        wants to drive should fail because a buzzer did not, so this
        swallows what tune() would raise.
        """
        try:
            self.tune(n)
        except Exception:
            pass

    # ------------------------------------------------- camera mast

    def mast(self, angle: int) -> str:
        """Point the camera mast servo. Negative lets it go limp.

        Not rate-limited and not a relay operation — it goes straight
        out rather than through _set, which exists to protect contacts
        the servo does not have.
        """
        angle = int(angle)

        # Recorded BEFORE the write, not after. If Ctrl+C lands between
        # the write and the reply the mast is probably already moving,
        # and a redundant stow at exit costs nothing while a missed one
        # leaves the camera up.
        self.mast_angle = angle

        # Whatever else happens now, something will put it down. atexit
        # runs on a normal return, on sys.exit, and while unwinding an
        # unhandled exception — which is what a Ctrl+C becomes wherever
        # it happens to land, including inside Camera's constructor
        # before any caller holds a reference to anything.
        if not self._mast_atexit:
            atexit.register(self._stow_mast)
            self._mast_atexit = True

        return self._send(f"V {angle}")

    def _stow_mast(self) -> None:
        """Lower the mast if it is raised. Safe to call repeatedly.

        Deliberately forgiving: this runs during interpreter shutdown
        and while cleaning up after failures, where raising would
        replace whatever actually went wrong with a serial error.
        """
        try:
            if (self._ser is not None and self.mast_angle is not None
                    and self.mast_angle > self.MAST_DOWN):
                self.mast(self.MAST_DOWN)
        except Exception:
            pass

    def camera_up(self, settle: float = 0.6) -> str:
        """Raise the mast and wait for it to actually get there.

        The servo reports nothing back, so the wait is open-loop and
        generous. Returning early would hand the camera to a caller
        that then locks exposure on a view still swinging through
        frame.
        """
        reply = self.mast(self.MAST_UP)
        time.sleep(settle)
        return reply

    def camera_down(self, settle: float = 0.0) -> str:
        reply = self.mast(self.MAST_DOWN)
        if settle:
            time.sleep(settle)
        return reply

    def stop(self) -> str:
        # Never held. Whatever else is rate-limited, stopping is not.
        now = time.monotonic()
        self._last_command_at = now
        self._changed_at = now
        self._last_state = (0, 0, 0, 0)
        self.last_command_held = False
        return self._send("S")

    def unstick(self, reverse_s: float = 0.5) -> None:
        """Jolt relays that have stayed closed, then stop.

        Found by hand: when a contact stays engaged after an arc, a
        short burst the other way clears it and normal driving
        resumes. Reversing energises the opposite relay of each pair,
        which takes the current off the stuck contact and lets it let
        go.

        The rover moves during this — backwards, briefly. That is the
        cost of the recovery, not a side effect to be tuned out.

        Ignores the cooldown, being the thing you reach for when the
        relays are already in a state nobody asked for.
        """
        a, b = _DIR[-1]
        self._set((a, b, a, b), force=True)
        time.sleep(reverse_s)
        self.stop()

    def ping(self) -> str:
        """Report applied state without changing anything."""
        return self._send("P")

    def applied(self) -> tuple[int, int, int, int]:
        """Relay states as actually set right now.

        During a dead-time pause this lags what you asked for.
        """
        parts = self.ping().split()
        return tuple(int(x) for x in parts[1:5])  # type: ignore[return-value]

    # ------------------------------------------------------- config

    def info(self) -> str:
        reply = self._send("?", expect="info")
        self._note_info(reply)
        return reply

    def config(
        self,
        deadtime_ms: Optional[int] = None,
        watchdog_ms: Optional[int] = None,
        active_low: Optional[bool] = None,
        stagger_ms: Optional[int] = None,
    ) -> str:
        """Change firmware behaviour at runtime — no reflashing.

        deadtime_ms=0 lets reversals happen instantly, so sequence
        the stop yourself if you still want one.
        watchdog_ms=0 means a crash here leaves the motors running.
        stagger_ms starts the two motors that many milliseconds
        apart, so their inrush does not land on the rail at once.
        """
        reply = self.info()
        if deadtime_ms is not None:
            reply = self._send(f"C D {int(deadtime_ms)}", expect="info")
        if watchdog_ms is not None:
            reply = self._send(f"C W {int(watchdog_ms)}", expect="info")
        if active_low is not None:
            reply = self._send(f"C A {1 if active_low else 0}", expect="info")
        if stagger_ms is not None:
            reply = self._send(f"C S {int(stagger_ms)}", expect="info")

        self._note_info(reply)
        return reply

    # ----------------------------------------------------- lifecycle

    def close(self) -> None:
        # The mast comes down before the port closes. Every caller that
        # tears down in the other order — bridge first, camera second —
        # would otherwise be asking a shut serial port to move a servo,
        # and both scripts here used to do exactly that.
        self._stow_mast()

        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.stop()
        except Exception:
            pass
        with self._lock:
            if self._ser is not None:
                self._ser.close()
                self._ser = None

    def __enter__(self) -> "Car":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Car {self.port}>"
