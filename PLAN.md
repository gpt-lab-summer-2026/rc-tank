# Roadmap — a camera-only tank that roams

A phase-by-phase plan for the rest of this project. **For the
step-by-step version with tick boxes, see [CHECKLIST.md](CHECKLIST.md)**
— this document holds the reasoning, that one holds the doing.

Phase 0 is proven on hardware. Phases 1 and 2 are written but have
never run, so this plan treats them as bring-up work with real
unknowns. Everything from Phase 3 on is new.

**Two constraints shape everything below.**

1. **The camera is the only sensor.** No rangefinders, no bumper, no
   IMU, no encoders. Every number the tank acts on is derived from
   pixels.
2. **The relays stay.** Each motor is full-forward, off, or
   full-reverse. No speed control, 300 ms of dead-time on reversals,
   and a finite number of contact operations.

Together these remove every fallback the earlier draft leaned on. The
vision model is no longer an upgrade to a system that is already safe
— it *is* the system. Section 3 says what that costs and what pays for
it.

---

## 1. Where this stands today

| Phase | File | Status |
|---|---|---|
| 0 | [teleop.py](teleop.py) | **Works on hardware.** Keyboard driving, proven end to end. |
| — | [car.py](car.py) | **Works,** with one dangerous behaviour — see §3. |
| — | [esp32_relay_bridge.ino](esp32_relay_bridge/esp32_relay_bridge.ino) | **Works.** Line protocol in, four relay coils out. Dead-time on reversals, watchdog on silence. |
| 1 | [record.py](record.py) | **Written, never run.** Frames + commands to `dataset/session_*/`. Also owns `Camera`. |
| 2 | [roam.py](roam.py) | **Written, never run.** HSV floor histogram, back-projected per pixel, steers toward the freest third. |

The proven part is: *the Pi can tell the ESP32 what to do, and the
motors do it.* The protocol, dead-time, watchdog and port handling are
all exercised. Nothing above that has met a camera.

### Bugs found by reading — all fixed

Six, found before any of this ran. Listed because the reasoning
behind the first one shapes §3, and because the last two are the kind
that only bite once you are standing in the room.

- **The watchdog did not protect against a hung control loop.** ✅
  `Car` ran a daemon thread sending `P` pings every ~0.13 s, and the
  firmware resets `lastRx` on *any* line including `P`
  ([esp32_relay_bridge.ino:155](esp32_relay_bridge/esp32_relay_bridge.ino#L155)).
  So a deadlocked control loop — camera stall, inference hang, GC
  pause — still looked like traffic, the watchdog never fired, and the
  relays held their last state. The tank drove forward indefinitely.

  With a bumper that is an annoyance. Camera-only it is the failure
  mode with nothing behind it. Fixed with `Car(command_timeout=…)`:
  the keepalive now resends the *last commanded state* rather than a
  bare ping, and goes quiet once the caller stops issuing commands, so
  the watchdog measures control-loop liveness. `roam.py` re-sends every
  tick, which is free — the firmware's `tick()` returns early when
  target equals applied
  ([esp32_relay_bridge.ino:108](esp32_relay_bridge/esp32_relay_bridge.ino#L108)),
  so no coil is written and no contact life is spent. Measured ~550 ms
  from hang to release. `teleop.py` and `record.py` keep the old
  hold-indefinitely behaviour, which is correct with a human present.

- **`record.py` crashed on the first gamepad event.** ✅
  `iter(self.dev.read, None)` — evdev's `read()` returns a *generator*
  and never returns `None`, so `event` bound to a generator object and
  `event.type` raised `AttributeError`. The sentinel-`None` pattern
  belongs to `read_one()`.

- **`roam.py` never locked exposure.** ✅ It built `Camera()` with
  defaults, so AE and AWB kept changing the colours under a floor
  histogram learned once at startup. Now locked by default, with
  `--no-lock-exposure` to opt out.

- **`roam.py` reversed blind and unboundedly.** ✅ `Policy.decide`
  returned `(-1, -1)` after `--stuck-after` seconds of turning and
  never reset `_turning_since`, so it reversed until the centre
  cleared — into whatever was behind it. Now a bounded state with
  `--reverse-for` (default 1.0 s) that hands back to the turning
  rules and restarts the stuck timer.

- **`record.py` left the terminal unusable.** ✅ `KeyboardInput` puts
  the tty in cbreak mode, and `Camera` exits with `SystemExit` when
  picamera2 is missing — which is exactly what happens on first run
  in a venv built without `--system-site-packages`. The shell was left
  in cbreak. Now restored on the way out.

- **`roam.py` documented a flag that does not exist.** ✅ The docstring
  pointed at `--show-patch` for inspecting the reference patch; the
  actual flag is `--debug-image`.

### Design gaps the camera has to close by itself

1. **One colour cue.** An obstacle at the carpet's hue and saturation
   is invisible to `roam.py`, and no threshold fixes it. This was
   going to be the rangefinder's job. It is now the model's job, which
   is what makes Phase 6 load-bearing rather than an improvement.
2. **Everything is in pixels.** `--go 0.45` means "free almost halfway
   up the frame" — no physical meaning, does not survive a change of
   camera pitch or resolution.
3. **No memory.** Every frame is judged alone, so the tank cannot know
   it has swept this corner four times. That is exactly what "roam
   freely" requires it to know.
4. **No sense of its own motion.** No odometry at all, so it cannot
   tell a completed turn from a stalled one, or driving from being
   wedged against a chair leg.
5. **No way to measure a change.** No replay, no metric, no fixed
   course.

### Unknowns only running it will settle

- Does picamera2 import, and does the camera enumerate at all?
- Does `_full_fov_mode()` find a sane mode, and is the FOV really full?
- Is the `RGB888`-means-BGR assumption right on this stack?
- Can the Pi sustain capture + JPEG encode + serial at 12 fps?
- What does the floor histogram look like on *this* floor, and how far
  must the threshold move between rooms?
- **How fast does the tank actually go, and how big is the strip in
  front of it that the camera cannot see?** These two numbers decide
  whether camera-only avoidance is feasible at all. See §3.
- How badly do the tracks shake the camera, and does rolling shutter
  make frames unusable while moving?

---

## 2. What "done" means

The tank runs for **30 minutes in an unfamiliar furnished room, with
no human touching it, and makes no damaging contact.** It covers most
of the accessible floor rather than orbiting one clear patch, and it
recovers from being wedged without help.

Three numbers track that:

- **MDBC** — mean distance between contacts, in metres. Contact means
  a stall the tank did not predict, or a human intervening.
- **Coverage** — fraction of a taped arena's floor cells visited in
  10 minutes. This is what separates roaming from circling.
- **Stuck fraction** — share of runtime where commanded motion and
  *visually measured* motion disagree.

Baselines are recorded at the **end of Phase 2**, on the first working
`roam.py`, before any Phase 3+ work.

Note the softened wording: "no damaging contact," not "no contact."
Camera-only with no bumper means occasional contact is a design
outcome, not a bug. §3 explains why, and what makes it survivable.

---

## 3. The safety argument, since nothing catches a mistake now

This is the section to disagree with if any section deserves it.

With no rangefinder and no bumper, **there is no independent layer
below the perception stack.** If the model says the floor is clear and
it is not, the sequence runs all the way to the motors. The earlier
draft put a reflex layer on the ESP32 precisely so that a Pi-side
mistake could not reach the wheels; that layer needed sensors and is
now gone.

What is left has to come from four places.

**Fix the watchdog so it means what it appears to mean.** Today the
keepalive thread pets the firmware watchdog independently of whether
the control loop is alive (§1). The fix is to delete the thread and
have the control loop **re-send its current command every tick**.

This is free at the relay level: the firmware's `tick()` returns
immediately when target equals applied
([esp32_relay_bridge.ino:108](esp32_relay_bridge/esp32_relay_bridge.ino#L108)),
so re-sending an identical `R 1 0 1 0` writes no coils and costs no
contact life. At 10 Hz it is about 20 bytes per tick on a 115200 link
— nothing. And the 400 ms watchdog then genuinely tracks control-loop
liveness: if perception hangs, the motors release. Optionally also
make `P` not reset `lastRx` in firmware, but note that `teleop.py`
depends on the current behaviour and would need the same re-send
treatment.

**Detect contact visually, since there is no switch.** When the tank
is commanded forward but ground-plane optical flow is near zero, it
has hit something or is wedged. That is a real, reliable camera-only
collision detector at roughly 200–300 ms latency. It does not prevent
the contact, but it stops the tank grinding into a wall for ten
seconds, and it supplies the negative training labels the bumper would
have. Phase 4.

**Slow the tank down, which with relays means the battery pack.** PWM
is off the table, so the one remaining lever on speed is supply
voltage: a lower-voltage pack, or motors run under their rated
voltage. No new electronics, one-time change. Whether it is needed
depends on a five-minute measurement in Phase 2 — see the design
equation below.

**Make contact harmless mechanically.** Foam or a sprung skid plate
across the front is not a sensor and adds nothing to the software. If
occasional contact is a design outcome, the cheap move is to make
occasional contact cost nothing.

### The design equation

Camera-only avoidance works if and only if:

```
   near-field blind wedge   >   stopping distance
```

The **blind wedge** is the strip in front of the tracks that a
downward-pitched camera cannot see. The **stopping distance** is
`speed × total latency`, plus coast.

Total latency, camera-only with relays:

| Stage | Estimate |
|---|---|
| Camera exposure + ISP | 30–60 ms |
| Preprocess | ~5 ms |
| Inference | 20–70 ms |
| Arbiter | ~5 ms |
| **`Smoother` majority vote + min-interval** | **300–500 ms** |
| Serial + firmware + relay actuation | ~15 ms |
| **Dead-time, if the move is a reversal** | **300 ms** |

Roughly 400–600 ms. At 0.4 m/s that is 16–24 cm of blind travel.

So there are three knobs, and the plan uses all three:

- **Shrink the latency.** The smoother dominates everything else
  combined, and it exists to protect relay contacts. Make it
  **asymmetric**: changing to a *safer* action — stop, turn away —
  bypasses the vote and the minimum interval; changing to a *less
  safe* action, meaning forward, keeps the full smoothing. Braking is
  urgent, accelerating is not. Emergency stops are rare, so this costs
  almost no extra contact operations. Phase 2g.
- **Shrink the speed.** Battery pack voltage, if Phase 2 shows it is
  needed.
- **Grow the blind wedge margin.** Camera pitch trades near vision
  against far vision. Pitch down and the wedge shrinks but lookahead
  shortens; pitch up and the reverse. There is a correct angle and it
  is found by measurement, not by eye. Phase 2f.

Measure all three early. If the equation cannot be satisfied at any
pitch, that is a finding, and the honest responses are a lower-voltage
pack or a wider lens — not more model.

### Standing rules

1. Physical kill switch on the motor battery, always reachable.
2. The control loop re-sends every tick; no independent keepalive.
3. The model treats **unknown as obstacle**, never as floor.
4. Every autonomy session keeps the 30 seconds before any stall.
5. Nothing drives unattended until Phase 9.

---

## 4. The architecture we are heading for

```
   ┌───────────────────────── Raspberry Pi ──────────────────────────┐
   │                                                                 │
   │              ┌──────────────────────────────────┐               │
   │  Pi camera ─►│  one network, two heads          │               │
   │      │       │    ├─ cost per heading  ◄── used │               │
   │      │       │    └─ floor mask        ◄── shown│               │
   │      │       └──────────────┬───────────────────┘               │
   │      │                      │                                   │
   │      │  ┌───────────────────▼────────────────────┐              │
   │      │  │  arbiter  (~40 lines, readable)        │              │
   │      │  │    × momentum   × novelty  (Phase 8)   │              │
   │      │  │    sample a bearing, commit to it      │              │
   │      │  └───────────────────┬────────────────────┘              │
   │      │                      │                                   │
   │      └─► optical flow ──┬───┤ visual yaw + speed  (Phase 4)     │
   │                         └───┤ stall detection                   │
   │                             │                                   │
   │                             ▼   asymmetric smoother             │
   │                     (left, right) ∈ 9 discrete states           │
   └─────────────────────────────┼───────────────────────────────────┘
                                 │ re-sent every tick, 10 Hz
   ┌─────────────────────────────▼───────────────────────────────────┐
   │           ESP32 relay bridge — unchanged, except that           │
   │           the watchdog now tracks control-loop liveness         │
   └─────────────────────────────┼───────────────────────────────────┘
                                 ▼
                         relays ─► motors
```

Three ideas carry the weight.

**Cost per heading is multimodal-safe.** This is why it was the right
output to pick. A tank facing an open room should report that left is
good *and* right is good *and* straight is good — three separate
numbers, all correct. A network regressing directly to a steering
angle has to pick one, and gradient descent picks the average, which
in that situation is a lie. `roam.py`'s docstring already makes this
argument against cloning actions; cost-per-heading is what makes the
model decide the move without inheriting the flaw.

**Two heads, one network.** The cost head drives the tank. The floor
mask head is trained alongside and only ever displayed. It costs
almost nothing, it makes the multi-task training better, and it means
a model that decides moves is still a model you can *watch* being
wrong. Never ship a network whose only output is a decision you cannot
inspect.

**The camera is also the odometer.** Optical flow on the ground plane
gives yaw rate and forward speed. That is what closes the loop on
turns, detects stalls, shifts the costmap, and supplies outcome labels
— all the jobs the IMU and bumper were going to do. It is the single
highest-leverage thing to build in the camera-only design.

---

## 5. Phase 1 — Get the camera and recording working

**Why now.** The camera is the entire sensor suite, and recorded
frames are the only training data that will ever exist. This phase is
mostly about proving the camera stack on this specific Pi, module and
OS image — work that reliably takes longer than the code suggests.

**What recording is for now.** `record.py`'s docstring frames the
dataset as behaviour-cloning labels and argues for a gamepad on class
balance. `roam.py` then argues cloning fails on roaming data, and that
argument held — §4 explains how cost-per-heading sidesteps it. So
recording's purpose has shifted: it captures sessions for **replay
(Phase 3), for cost targets (Phase 6), and for outcome labels
(Phase 4)**. All three care about frames and telemetry, not about
action-class balance. Two consequences:

- **Input is the keyboard.** The gamepad existed to keep action
  classes balanced for cloning, and nothing here clones actions.
  `record.py` now defaults to keyboard, with the gamepad behind an
  opt-in `--gamepad` if it is ever wanted. Latched keys are fine when
  the frames are the point.
- Drive for **scene variety** — rooms, lighting, floor types,
  obstacles at varied ranges — rather than for correction frequency.
  Also deliberately drive *into* things at low speed: those frames
  become the negative examples that no other source will provide.

### Work

**1a — Camera stack.** Verify `rpicam-hello` / `libcamera-hello` sees
the module before touching Python. Install picamera2 from apt
(`sudo apt install -y python3-picamera2 python3-opencv`); a venv must
be created with `--system-site-packages` or the import at
[record.py:91](record.py#L91) fails regardless of what pip says.

**1b — Sanity-check the frames.** Capture something recognisably red
and confirm the `RGB888`-is-really-BGR assumption at
[record.py:104](record.py#L104). Confirm what `_full_fov_mode()` picks
and that the FOV really is full — a silently cropped mode costs
peripheral vision exactly where obstacles first appear, and there is
no rangefinder to make up for it.

**1c — Keyboard input needs a real terminal.** `KeyboardInput` puts the
tty into cbreak mode, so `record.py` works over SSH but not with piped
stdin and not from a systemd unit. Worth knowing before Phase 9 tries
to start things headless.

**1d — Record at the resolution roaming consumes.** `record.py` saves
320×240; `roam.py` runs perception on the full 640×480 frame.
Thresholds are fractions of height so they scale, but the morphology
kernels in `FloorModel.mask` — a 5×5 blur and a 7×7 ellipse — are
absolute pixels and behave differently at half resolution. **Replay
would not reproduce live behaviour**, which quietly undermines
Phase 3.

Record at 640×480 (`--width 640 --height 480`) and downscale later for
training. About 2 GB/hour at 12 fps, so pair it with log rotation.
Scaling the kernels instead is also fine — pick one now and write it
down.

**1e — Throughput.** Watch the `dropped` counter. If frames drop,
lower `--fps` or `--quality` before concluding anything about the Pi.

**1f — Capture the first real corpus.** Several rooms, floor types,
daylight and artificial light, plus deliberate low-speed contacts.
Phases 3, 4 and 6 all live on this, so breadth beats volume.

### Done when

- 10 minutes of continuous recording, zero dropped frames.
- Frames are the right colours, resolution and full-FOV.
- The terminal is usable after a session ends, including after a crash.
- `log.csv` and `meta.json` complete, frame count matching.
- At least three sessions across different rooms archived.

---

## 6. Phase 2 — Get roaming working, and find out if the geometry works

**Why now.** First time perception drives the motors, and it produces
the baseline everything later is measured against. It also settles the
question in §3 that decides whether camera-only is viable at all.

### Work

**2a — Verify the watchdog fix on real hardware.** The code change is
done (§1), but it has only been tested against a simulated bridge.
Before anything drives under perception control, confirm on the tank:
start `roam.py`, `SIGSTOP` the process mid-drive, and check the motors
release within about a second. Then confirm the opposite — that a
healthy loop never trips it — by watching for unexplained stutters
across a full run. Raise `--command-timeout` if they appear.

**2b — Re-check the `roam.py` behaviour fixes in the room.** Exposure
locking and the bounded reverse are in. What is not settled is whether
`--reverse-for 1.0` is the right length on this floor, and whether
locking exposure at startup leaves the frame usable when the tank
drives from a bright patch into a dark one. Both are tuning, and both
need the arena.

**2c — Dry run on the bench.** `--dry-run --debug-image /tmp/roam.jpg`,
then look. Green is what it thinks is floor, red dots are where each
column stops being free, cyan is the learned patch. Carry the tank
around by hand and watch the mask respond. Most tuning belongs here,
where mistakes cost nothing.

**2d — Tune the floor model on the real floor.** `--threshold` against
this carpet, then again in a second room. That spread is the honest
measure of how brittle colour-only perception is, and it is the number
that justifies Phase 6.

**2e — Wheels off the ground.** Verify decisions become the right
relay states, and that dead-time and watchdog behave under a real
control loop rather than under teleop.

**2f — Measure the three numbers in the design equation.** The most
important half-hour in this phase.

- **Speed** at full duty: tape, stopwatch, five minutes.
- **Blind wedge**: put a marker on the floor and walk it toward the
  tank until it leaves the frame. That distance is the hard floor on
  how late the camera can possibly react.
- **Total latency**: flash something in view and time it to the relay
  click.

Then check `blind wedge > speed × latency`. Adjust camera pitch and
re-measure; there is a correct angle, trading near vision against
lookahead. If no pitch satisfies it, the answer is a lower-voltage
pack or a wider lens — record the finding and decide before Phase 6,
because no model fixes a geometry problem.

**2g — Make the smoother asymmetric.** Safer decisions (stop, turn
away) bypass the majority vote and the minimum interval; forward keeps
full smoothing. This cuts the dominant latency term for exactly the
decisions where latency hurts, at almost no cost in contact
operations. See §3.

**2h — First driving runs, hand on the kill switch.** Then tune `--go`
and `--turn-margin` for how late it turns, `--min-interval` for
dithering in corners.

**2i — Motion effects on perception.** Tracks on a hard floor shake
the camera, and rolling shutter turns that into skew. If frames are
unusable while moving, options are a softer mount, shorter exposure
(more gain, more noise), or a lower duty cycle. This silently caps
everything downstream, including the optical flow that Phase 4 depends
on entirely.

**2j — Build the arena and record the baseline.** Roughly 3 × 3 m
taped, six obstacles chosen to break different things:

- a dark chair leg (thin, low contrast)
- a white wall (bright, textureless)
- cardboard the same colour as the floor (defeats the histogram)
- a mirror or glass panel (defeats everything — see §14)
- something under camera height, near-field (defeats the FOV)
- a cable lying flat (thin, on the ground plane)

Photograph the layout so it can be rebuilt. Run 10 minutes, record
MDBC, coverage and stuck fraction by hand. **This is the baseline.**

### Done when

- 10 minutes of driving in the arena without human rescue, however
  many contacts it takes.
- Debug images show a recognisably right mask in two rooms.
- The `--threshold` spread between rooms is written down.
- **The design equation is satisfied, or the plan to satisfy it is
  chosen.**
- Frames while moving are usable, or the mitigation is chosen.
- Baseline MDBC, coverage and stuck fraction recorded.

### Risks

- **It learns a wall as floor at startup.** The docstring already warns
  about this. Add a startup check: if the learned patch matches a large
  fraction of the *upper* frame, it is probably a wall — refuse to
  start rather than drive into it.
- **The floor model works nowhere reliably.** Possible on patterned
  rugs. That is a valid result — it just means Phase 6 carries the
  whole load, which camera-only already implies.

---

## 7. Phase 3 — Make it measurable

**Why now.** Everything downstream is a claim that something got
better. Without replay and a metric, those claims cannot be checked
and tuning degenerates into superstition. No new capability, on
purpose.

### Work

**3a — Restructure into a package.** Flat scripts importing each other
(`roam.py` pulling `Camera` and `label_of` out of `record.py`, which
imports `termios` at module scope) will not survive six more phases.

```
tank/
  config.py              # YAML-backed dataclasses, replacing the argparse sprawl
  drivers/  bridge.py  camera.py
  perception/  floor_hsv.py  geometry.py  flow.py  costmap.py  net.py
  control/  arbiter.py  explore.py  recovery.py  smoothing.py
  runtime/  loop.py  telemetry.py  webui.py
  tools/  replay.py  calibrate.py  label.py  bench.py
apps/     teleop.py  record.py  roam.py  autonomy.py
firmware/ relay_bridge/
```

Keep `teleop.py`, `record.py` and `roam.py` at the root as thin
wrappers. Arduino requires a sketch folder to match its `.ino` name, so
firmware moves wholesale.

The constraint that matters: **`runtime/loop.py` must run against
recorded frames with no hardware attached.** Perception and arbiter
take arrays and return decisions; only drivers touch the world.

**3b — Replay harness.** `tools/replay.py` runs perception and the
arbiter over a recorded session and reports the decisions it would
have made, plus an annotated video. Every later phase is evaluated by
replaying the same Phase 1 sessions through old and new code.

Verify fidelity once: replay a session and confirm decisions match
what was logged live. If Phase 1d was skipped, the resolution mismatch
surfaces here.

**3c — Telemetry and a live view.** Structured JSONL per run:
timestamp, cost curve, chosen bearing, smoothed decision, flow
estimates, stall flags. Plus a small Flask MJPEG page showing the
annotated frame over WiFi. When the model decides the moves, watching
what it sees is the only real debugging tool there is.

**3d — Instrument the metrics.** MDBC, coverage and stuck fraction
computed from telemetry, so Phase 2's hand-counted baseline becomes a
number the machine produces.

**3e — Power and wiring hardening.** Brownouts from motor current are
the most common cause of inexplicable failures in this exact build,
and a browned-out Pi is now an unmonitored one.

- Separate the Pi's 5 V from the motor pack, or feed it through a
  3 A+ buck converter with real bulk capacitance. Common ground with
  the ESP32.
- Flyback diodes or RC snubbers across the motor terminals.
- Short, twisted motor leads; a ferrite bead on the USB cable. Relay
  arcing and brush noise corrupt USB serial.
- Log `vcgencmd get_throttled` into telemetry so undervoltage appears
  as data rather than as a mystery.
- Physical kill switch on the motor battery.
- Foam or a sprung skid plate across the front (§3).

### Done when

- `pytest` runs perception and arbiter with no camera and no serial port.
- `tools/replay.py` reproduces a session's live decisions.
- Web view shows the annotated frame at ≥5 fps over WiFi.
- Metrics computed automatically, agreeing with Phase 2's baseline.
- A 20-minute run shows no undervoltage events.

**Effort:** the largest non-hardware phase and the most tempting to
skip. Skipping it makes Phases 5–7 unfalsifiable.

---

## 8. Phase 4 — Make the camera the odometer

**Why now.** This is the phase that replaces the entire sensor suite
from the earlier draft. Yaw rate, forward speed and stall detection
are what the IMU and bumper were going to provide, and all three come
out of optical flow. Nothing downstream works well without them:
Phase 5's costmap needs egomotion to shift, Phase 6's outcome labels
need stalls, and Phase 8's recovery needs to know a turn completed.

### Work

**4a — Yaw rate from horizontal flow.** The most robust camera-only
measurement available. A pure rotation produces near-uniform
horizontal image flow, so sparse Lucas-Kanade on good features, take
the median horizontal displacement, divide by focal length. Reliable
even on plain floors, because it can use the *whole* frame including
walls and furniture, not just the ground.

**4b — Forward speed from ground-plane flow.** Track features in the
lower frame region and project their motion through the Phase 5b
homography, so displacement comes out in metres. Noisier than yaw:
plain carpet has few features, and motion blur eats them. Expect this
to be the weaker of the two, and design downstream code to tolerate it
— the costmap should degrade gracefully when speed is uncertain, not
lie confidently.

**4c — Calibrate the action model.** With yaw working, measure what
the relays actually do:

- degrees per second while spinning, both directions
- degrees per second while arcing, and the arc radius
- metres per second forward
- coast distance and coast angle after the relays release

That table is what lets the Phase 6 arbiter convert "I want to face
20° left" into "arc left for 0.6 s." Re-measure on carpet and on hard
floor; skid steer differs enormously between them.

**4d — Closed-loop turns.** Rather than spinning open-loop for a
computed duration, spin until the integrated visual yaw says the turn
is done. Removes the dependence on a calibration that drifts with
battery voltage and floor surface — and battery voltage sags
continuously over a session, so open-loop turns get shorter as the run
goes on.

**4e — Stall and contact detection.** Commanded forward plus
ground-plane flow near zero means contact or wedged. Debounce over 2–3
frames, then trigger a stop. This is the camera-only bumper: it does
not prevent the contact but it ends it in ~250 ms, and it produces the
negative labels Phase 6 needs.

Distinguish two cases if possible — flow zero everywhere (wedged,
tracks slipping) versus flow zero only at the bottom (something caught
underneath). They need different escapes.

**4f — Handle the failure cases honestly.** Flow dies on textureless
floors, in low light, and under motion blur. When feature count drops
below a threshold, **report low confidence and let the arbiter slow
its commitment**, rather than emitting a confident wrong velocity.
Silent degradation here poisons the costmap and the training labels
simultaneously.

### Done when

- Visual yaw over a hand-turned 360° lands within ~10°.
- A commanded 90° spin executes within ~15° across three surfaces.
- Driving into a wall triggers a stall stop within 400 ms, in 10 of 10
  attempts.
- Speed estimate tracks the tape-and-stopwatch measurement within ~20%
  on textured floor, and reports low confidence on plain floor rather
  than guessing.

---

## 9. Phase 5 — Stop thinking in pixels

**Why now.** This makes every later number mean something, and it is
what turns the model's output from "a number per column" into "free
metres per bearing" — the quantity Phase 6 regresses to.

### Work

**5a — Camera calibration.** About 20 checkerboard shots →
`cv2.calibrateCamera` → intrinsics and distortion. The wide Pi lens has
enough barrel distortion to matter at the frame edges, which is exactly
where obstacles appear before you hit them.

**5b — Ground homography.** With the camera fixed at a known height
and pitch, every floor pixel maps to a ground-plane point. Lay a tape
grid, click four or more known points, `cv2.findHomography`. Store it
in config beside the mount geometry, because re-seating the camera
invalidates it and that will happen.

This is also where **metric scale comes from with no rangefinder**:
camera height and pitch make the true distance of any floor pixel
known analytically. It is the reason camera-only can still produce
metres.

`free_profile`'s per-column pixel heights then become **free distance
in metres per bearing** — a small polar scan. That scan is the
training target for Phase 6.

**5c — Local costmap.** Body-fixed, ~4 m forward × 3 m lateral at 5 cm
cells (80 × 60), log-odds occupancy. Writers: vision free-space, and
stall events (which stamp a hard obstacle at the contact point and are
not forgotten quickly). Shift the grid with Phase 4's visual odometry.

**Keep the memory short: 3–5 seconds, a few metres.** Visual odometry
on a shaking tracked vehicle drifts fast. This is a rolling local
costmap, not a map. See §15 on why SLAM is out.

**5d — Policy on the map.** Port the reactive policy to read the
costmap, same behaviour to begin with. The point is to change the
representation without changing the outcome, so replay can prove the
port was clean.

### Done when

- On replay, projected obstacle distances match tape-measure truth
  within ~15% out to 2 m.
- Free-distance-per-bearing is produced and logged for every frame.
- The ported policy reproduces the Phase 2 baseline within noise.
- Config carries clearances in metres.

---

## 10. Phase 6 — The vision model

**Why now.** With Phase 5 there is a metric target to regress to, with
Phase 4 an outcome signal, and with Phase 3 a way to prove it beat the
histogram.

### The design

**One network, two heads.**

```
   160×120 BGR ─► shared encoder ─┬─► cost head:  free metres per bearing
                                  │               (~15–31 bins, -75°..+75°)
                                  └─► mask head:  floor / obstacle / unknown
                                                  (never used — only displayed)
```

The cost head is what drives the tank. The mask head is trained
alongside, costs almost nothing, improves the shared features, and
gives a human something to look at when the tank does something
stupid. See §4 for why cost-per-heading is the right output shape.

**Where the targets come from — three sources, in order of when they
become available.**

1. **Geometric, from Phase 5.** Free distance per bearing computed
   from the floor mask and the homography. Unlimited labels, free, and
   available for every recorded frame. Training on these alone
   produces a *distillation* of the existing pipeline: not smarter,
   but far faster, and one network instead of a chain.

2. **Outcome-based, from Phase 4.** Where the tank actually drove and
   made smooth progress, that bearing was genuinely cheap — lower its
   cost. Where it drove and stalled, that bearing was expensive
   regardless of what it looked like — raise it sharply. **This is
   where the model gets smarter than the pipeline it was distilled
   from**, because it learns cues geometry cannot express: the shadow
   line at the base of floor-coloured cardboard, the texture of a rug
   edge that catches the tracks, the visual signature of a corner it
   has been wedged in before.

3. **Hand-labelled validation.** 300 frames, brushed by hand, covering
   every arena case and at least two rooms. Two or three hours of
   work, and non-negotiable — sources 1 and 2 share their biases with
   the system that produced them, so a model evaluated only against
   them can be confidently wrong. Never train on this set.

**The arbiter** — the thin readable part, ~40 lines:

```
   cost curve
     → preference = softmax(-cost / temperature)
     → × momentum prior      (favour the current heading)
     → × novelty prior       (Phase 8: penalise recently-taken bearings)
     → sample a target bearing
     → hold it for a minimum commit time
     → map bearing to a relay action via Phase 4c's table
          |θ| < 10°   → forward
          10°–35°     → arc, closed-loop on visual yaw
          > 35°       → spin, closed-loop on visual yaw
     → asymmetric smoother (Phase 2g)
     → if every bearing is above the panic threshold: stop, reverse
       briefly, spin, resume
```

Two details that matter more than they look:

- **Sampling, not argmax.** Argmax in a symmetric room produces a tank
  that goes straight until something stops it. Sampling is what makes
  routes unpredictable, which is the literal request. It is also why
  the multimodal-safe output shape was worth insisting on.
- **A minimum commit time.** Without it, sampling every tick produces
  a tank that dithers and burns relay contacts at a rate that ends the
  hardware in days. Commit to a bearing for at least ~0.5 s.

### Work

**6a — Generate geometric targets** over the whole Phase 1 corpus.
**6b — Train the two-head net.** MobileNetV3-Small encoder with a
small decoder for the mask head and a global-pooled MLP for the cost
head; or honestly six conv layers plus two heads. Input resolution
matters more than depth. Train on a laptop or Colab, not the Pi.
Augment hard — brightness, contrast, colour jitter, blur, JPEG,
mild perspective — because domain shift is the failure mode that
matters and there is no sensor to fall back on.
**6c — Deploy.** Export ONNX, quantise int8, run under ONNX Runtime or
TFLite with XNNPACK. Target **≥15 fps at 160 × 120 on Pi 5 CPU**, loop
still at 10 Hz. Bench before wiring it in.
**6d — Add outcome fine-tuning** once a few hours of autonomous
sessions with stall labels exist. This is a second training round, not
a second model.
**6e — Keep the histogram** as a permanently-running second opinion.
It is nearly free, it fails on different inputs, and **disagreement is
a usable uncertainty signal** — when the two diverge, raise the cost
of that bearing rather than trusting either. Cheaper and more honest
than extracting calibrated confidence from a small quantised network.
**6f — A/B against the baseline.** Replay both over Phase 1 sessions,
then run both live in the arena.

### Done when

- Cost-head error against held-out geometric targets is low, and the
  mask head beats the HSV mask on the hand-labelled set — especially
  on the floor-coloured obstacle.
- ≥15 fps at 160 × 120, loop still at 10 Hz.
- Live arena MDBC beats the Phase 2 baseline.
- The tank visibly makes *different* choices on repeated runs from the
  same start.
- Histogram remains as a config-switchable fallback if the net fails
  to load.

### Risks

- **Domain shift.** Trained in one room, blind in the next. Mitigated
  by multi-room collection and hard augmentation — and by nothing
  else, because there is no sensor underneath. This is the largest
  single risk in the camera-only design and it deserves proportionate
  data collection.
- **Distillation with no fine-tuning is just a faster histogram.** If
  step 6d never happens, the model inherits every colour-only weakness
  it was supposed to fix. Do not stop at 6c.
- **Outcome labels are sparse and noisy.** Stalls are rare by design.
  Weight them heavily, and keep the geometric loss as the regulariser.
- **Quantisation eating the gain.** Compare fp32 and int8 before
  concluding anything about the architecture.

---

## 11. Phase 7 — A slow teacher, run offline (optional)

**Why now, and why this shape.** Monocular depth would catch the one
category vision-from-the-ground-plane structurally cannot: obstacles
that never touch the floor — an overhanging table edge at chassis
height, a chair seat the tank drives under and wears as a hat.

But **Depth Anything V2 Small runs at roughly 1–2 fps on a Pi 5 CPU**,
far too slow to drive with. The insight that makes it useful anyway:
it does not have to run live. Run it **offline over recorded sessions**
to produce better cost targets, then let Phase 6's fast network learn
to predict what the slow pipeline would have said. Offline compute is
free, and the student pays no runtime cost for the teacher's size.

This is the highest-value use of a big model on hardware that cannot
run one.

### Work

- Run **Depth Anything V2 Small** or **MiDaS small** over the corpus.
- **Anchor the scale** on the ground plane — camera height and pitch
  make the true depth of floor pixels analytically known, so relative
  depth can be fitted to metres with no rangefinder.
- Fold overhang obstacles into the cost targets, then retrain Phase 6.
- Optionally also **YOLO11n** offline, to raise the cost of bearings
  containing pets, feet or cables — things that deserve extra
  clearance rather than minimum clearance.

### Done when

- Overhang obstacles in the arena appear in the offline cost targets.
- The retrained student detects the overhang case live, at no fps cost.
- Arena MDBC improves measurably. **If it does not, revert** — a
  larger training pipeline that does not help is just more to maintain.

---

## 12. Phase 8 — Actually roaming

**Why now.** Phases 2–7 make the tank not hit things. Not hitting
things is satisfied perfectly by standing still, and nearly as well by
orbiting one clear patch forever. This is the part of the goal nothing
else addresses.

Phase 6's arbiter already samples, so the *mechanism* exists. This
phase is about the priors that multiply into it.

### Work

**8a — Novelty prior.** Keep a rolling histogram of recently-taken
bearings and visited costmap cells. Multiply the arbiter's preference
by a term that penalises the recently-taken. Cheap, and it is most of
what separates roaming from circling.

**8b — Anti-trap.** If the tank has stayed inside a small area or
revisited the same headings for too long, commit to a long straight
run or a large committed turn and suppress the novelty term until it
breaks out. Corners and furniture legs are attractors; something has
to be repulsive about them.

**8c — Recovery state machine.** On a Phase 4e stall, escalate:

```
  reverse 0.5 s  →  turn 90°  →  resume
       ↓ still stalled
  reverse 1.0 s  →  turn 150° the other way  →  resume
       ↓ still stalled
  stop, hold the relays off, alert loudly
```

Every state has a timeout and an exit. Turns close the loop on visual
yaw (Phase 4d). This is where Phase 2b's bounded reverse grows up.

Note what reversing means with no rear sensor: it is genuinely blind,
so keep it **short and rare**, and prefer spinning in place — the tank
can see where it is about to face before committing to go there.

**8d — Frontier preference.** Bias toward bearings with the most
*unknown* space in the costmap, not merely the most free space.
Unknown is where information is. A mild bias, not a planner — the map
is a few seconds deep and supports nothing more ambitious.

**8e — Relay budget check.** Log direction changes per minute over a
full run. If it is above ~20, raise the commit time or the momentum
prior. This is the number that decides whether the hardware lasts
weeks or months.

### Done when

- 10 minutes in the arena covers ≥70% of accessible floor cells.
- No heading-histogram spike indicating a persistent orbit.
- Deliberately wedging the tank produces recovery without human help
  in at least 8 of 10 attempts.
- Two runs from the same start produce visibly different paths.
- Direction changes per minute are within the relay budget.

---

## 13. Phase 9 — Soak and operations

**Why now.** Everything above was validated in ten-minute bursts in one
taped square.

### Work

- **Headless startup.** systemd unit, autostart on boot, sane restart
  policy, and a hardware arming switch so it does not drive off the
  bench at power-up.
- **Log rotation.** JSONL plus 640×480 frames fill an SD card fast.
  Cap by size, keep the last N sessions, always keep the 30 seconds
  before any stall.
- **Battery voltage from the ESP32's ADC** — not a new sensor, a pin
  and a divider. Worth it because visual odometry calibration drifts
  as the pack sags, and because parking beats browning out mid-turn.
- **Soak runs.** Three rooms, 30 minutes each, unattended. Every
  contact classified into the taxonomy below.
- **Tune from the failure log**, not intuition. By now the data exists
  to say which cause dominates.
- **Write the operator's guide.** `roam.py`'s TUNING block is the model
  to follow — it is already the best documentation in the repo.

### Done when

- Three 30-minute unattended runs in unfamiliar rooms, no damaging
  contact.
- Recovery from every wedge without help.
- Survives a full battery cycle without corrupting a session log.

---

## 14. Failure taxonomy

Camera-only changes what is solvable. Being honest about the last two
rows is part of the plan, not an admission of defeat.

| Failure | Answer |
|---|---|
| Obstacle same colour as floor | **Phase 6.** This is the model's central job — texture, shadow lines, context. |
| Thin objects: chair legs, cables | Percentile clearance (already), resolution, Phase 7's offline teacher |
| Below camera FOV, near field | **Geometry, not sensing.** The design equation in §3. |
| Overhang at chassis height | Phase 7's depth teacher, folded into cost targets |
| Wedged, high-centred | Phase 4e stall detection, Phase 8c recovery |
| Circling one patch | Phase 8a/b |
| Direct sunlight, low light | Locked exposure (2b), hard augmentation, flow confidence gating (4f) |
| Drop-offs, stairs | **Partly.** The floor mask catches most drop-offs because the floor visibly ends — but a down-stair carpeted in the same carpet may not. Test it explicitly; if unreliable, block the stairs with a board. Not a sensor, and the honest answer. |
| Glass, mirrors | **Unsolved.** A mirror looks like more room, and monocular depth fails on it too. Put mirrors in the training set if any exist, otherwise accept it as a known failure and do not run the tank in those rooms. |

---

## 15. What this plan deliberately does not do

**SLAM.** Tracked vehicles slip badly under skid steer, there are no
encoders, and monocular VO on a low-texture indoor floor at 10 Hz will
not hold a map. A short-horizon rolling costmap gives most of the
benefit for a fraction of the difficulty. Aimless roaming does not
need to know where it is, only what is near it.

**End-to-end behaviour cloning.** Regressing image → steering averages
the multimodal choices roaming presents, and camera-only removes the
signals that would have partly rescued it. Cost-per-heading gives the
model the decision without the flaw. See §4.

**ROS 2.** It buys nav2 and tf, but nav2 assumes odometry far better
than visual flow on a shaking tracked chassis, and the existing code is
clean and small.

**A global planner.** "Without predetermined routes" is the
requirement.

**Pulse-and-coast to fake speed control.** Duty-cycling the relays at
1 Hz would halve the effective speed and roughly triple the contact
wear, ending the relay board in under three hours of driving. If speed
must come down, it comes down at the battery.

---

## 16. Open questions

Answering these sharpens Phases 4, 6 and 7. None block Phase 1.

1. **Which Pi?** Pi 5 makes Phase 6 comfortable on CPU. Pi 4 makes it
   tight but workable at 160 × 120. Pi Zero 2 W means a very small
   network and probably skipping Phase 7.
2. **Camera module and lens FOV?** The `_full_fov_mode` logic at
   [record.py:138](record.py#L138) suggests an IMX219. FOV drives the
   blind wedge in §3, and a wider lens is the one purchase that would
   genuinely help — and is not a sensor.
3. **Floor type where it will actually run?** Optical flow (Phase 4)
   needs texture. Plain lino is the hard case and would push more
   weight onto yaw-only odometry.
4. **Stairs or drop-offs anywhere in scope?** Decides how hard to push
   on the drop-off row in §14.
5. **Is a lower-voltage pack acceptable** if Phase 2f shows the design
   equation fails? It is the only speed lever left.

---

## 17. Suggested order of attack

```
  Phase 1  camera + recording      ── the only data source there will ever be
     │
  Phase 2  roaming + baseline      ── also settles whether the geometry works at all
     │
  Phase 3  measurement             ── blocks everything after it
     │
  Phase 4  camera as odometer      ── replaces the whole sensor suite
     │
  Phase 5  metric frame + costmap
     │
  Phase 6  the vision model        ← the stated means
     │
     ├── Phase 7  offline depth teacher (optional)
     │
  Phase 8  exploration             ← the stated goal
     │
  Phase 9  soak + operations
```

The chain is longer than the earlier draft because Phase 4 now sits on
the critical path instead of being a shopping list. Two shortcuts if
that is too patient:

- **The vision model sooner.** Phase 6 can train on geometric targets
  alone, which need only Phase 5b's homography — not Phase 4, not the
  costmap. That gets a working two-head net running the tank early. It
  will be a fast distillation of the histogram rather than an
  improvement on it, but it proves the whole cost-per-heading path end
  to end, and outcome fine-tuning can be added later.
- **Roaming sooner.** Sampling a bearing instead of taking the freest
  third can be bolted onto Phase 2's policy in an afternoon. Without
  Phase 8b it will trap itself in corners, but it makes the tank
  visibly stop following the same route, which is most of what the
  goal describes.
