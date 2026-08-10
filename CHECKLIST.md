# Phase checklist

The step-by-step companion to [PLAN.md](PLAN.md). The plan holds the
reasoning; this holds the doing.

**How to use it.** Work down a phase in order. `Do` items are the
work. **`Gate` items are the exit test — a phase is finished when
every Gate box is ticked, and not before.** Anything marked ⛔ is a
hard stop: if it fails, do not carry on to the next phase, because
later work will be built on a broken foundation and you will not be
able to tell.

Numbers you measure go in the [measurements log](#measurements-log) at
the bottom. Several later phases compare against them, so write them
down even when they look obvious at the time.

---

## Phase 0 — Re-verify teleop after the bug fixes ✅ mostly done

**Goal:** confirm the keepalive rewrite in `car.py` did not break the
one thing that already worked.

### Do
- [ ] `python3 teleop.py` — connects and prints an `info bridge=1 …` line
- [ ] `w` `s` `a` `d` `q` `e` each move the tank the direction they claim
- [ ] `space` stops, `x` quits cleanly
- [ ] Press `w`, then touch nothing for 30 seconds — **the tank must keep driving**
      (teleop passes no `command_timeout`, so the keepalive holds the state)
- [ ] `i` prints the firmware config

### Gate
- [ ] ⛔ Every direction key does what it says (a swapped motor now is a
      swapped motor in every later phase)
- [ ] Latched forward survives 30 s of no input
- [ ] Quitting with `x` releases the motors
- [ ] Unplug the USB cable mid-drive → motors release within ~1 s
- [ ] The terminal is normal after quitting (not stuck in cbreak)

---

## Phase 1 — Camera and recording

**Goal:** frames on disk, correct and complete. This is the only sensor
data that will ever exist.

### Do — camera stack
- [ ] `rpicam-hello --list-cameras` lists the module
- [ ] `rpicam-hello` shows a live preview
- [ ] `sudo apt install -y python3-picamera2 python3-opencv`
- [ ] If using a venv, it was created with `--system-site-packages`
- [ ] `python3 -c "from picamera2 import Picamera2; print('ok')"`
- [ ] `pip3 install pyserial`

### Do — sanity-check the frames
- [ ] Run `record.py` and note the printed `sensor mode WxH (full field of view)` line
- [ ] Point at something clearly **red**, record a few frames, open one:
      it must look red, not blue *(catches the RGB888/BGR assumption)*
- [ ] Compare the frame's field of view against `rpicam-hello`'s preview —
      they should match *(catches a silently cropped sensor mode)*

### Do — capture the corpus
- [ ] Record at the size roaming uses:
      `python3 record.py --width 640 --height 480 --lock-exposure`
- [ ] One 10-minute continuous session, watching the `dropped` counter
- [ ] Sessions in **at least 3 different rooms**
- [ ] At least one session on each floor type available
- [ ] At least one session in daylight and one under lamplight
- [ ] Deliberately drive slowly **into** obstacles during some sessions
- [ ] Archive the sessions somewhere they will not be deleted

### Gate
- [ ] ⛔ Saved frames are the right colours and the full field of view
- [ ] ⛔ Saved frames are 640×480 *(replay stops matching live otherwise)*
- [ ] A 10-minute session ends with `dropped 0`
      — if not, lower `--fps` or `--quality` and repeat
- [ ] Frame count matches log rows:
      `ls SESSION/frames | wc -l` equals `tail -n +2 SESSION/log.csv | wc -l`
- [ ] `meta.json` exists and is complete in every session
- [ ] ≥3 rooms archived
- [ ] The terminal is usable after a session ends, including after a crash

---

## Phase 2 — Roaming, geometry, and the baseline

**Goal:** the first perception→motor loop, and the measurements that
decide whether camera-only avoidance is possible at all.

### Do — bench first, nothing moving
- [ ] `python3 roam.py --dry-run --debug-image /tmp/roam.jpg`
- [ ] Open the image: green overlay is floor, red dots are where each
      column stops being free, cyan box is the learned patch
- [ ] Carry the tank around by hand and watch the mask follow
- [ ] Tune `--threshold` in room 1 → record the value
- [ ] Tune `--threshold` in room 2 → record the value
- [ ] Record the **spread** between them — this is how brittle colour-only is

### Do — measure the three numbers that matter ⛔
- [ ] **Speed:** mark 2 m, drive full forward, time it, 3 runs, average
- [ ] **Blind wedge:** slide a small object toward the tank until it drops
      out of the bottom of the frame; measure from the front of the tracks
- [ ] **Latency:** film with a phone in slow motion — from an obstacle
      appearing in view to the relay clicking
- [ ] Compute `stopping distance = speed × latency`
- [ ] Adjust camera pitch and re-measure both — pitching down shrinks the
      wedge but shortens lookahead; there is a best angle

> **Worked example.** 0.35 m/s × 0.5 s = **17.5 cm** stopping distance.
> A 25 cm blind wedge passes with margin. A 12 cm one fails, and no
> amount of model fixes it — see [PLAN.md §3](PLAN.md).

### Do — safety checks before it drives itself
- [ ] Wheels off the ground: decisions produce the right relay states
- [ ] Watchdog test — with the tank driving forward:
      ```
      pkill -STOP -f roam.py     # motors must release within ~1 s
      pkill -CONT -f roam.py
      ```
- [ ] Kill switch is fitted and reachable
- [ ] Foam or a skid plate on the front

### Do — first driving runs
- [ ] First runs with a hand on the kill switch
- [ ] Tune `--go` for how late it turns
- [ ] Tune `--turn-margin` / `--min-interval` for dithering in corners
- [ ] Tune `--reverse-for` if it backs into things
- [ ] Watch for unexplained stutters — if they appear, raise `--command-timeout`
- [ ] Check frames are still usable while moving (vibration, rolling-shutter skew)

### Do — the arena and the baseline
- [ ] Tape out ~3 × 3 m
- [ ] Place all six obstacles: dark chair leg · white wall ·
      floor-coloured cardboard · mirror or glass · something below camera
      height · a flat cable
- [ ] Photograph the layout so it can be rebuilt exactly
- [ ] Run 10 minutes and hand-count the baseline numbers

### Gate
- [ ] ⛔ **`blind wedge > speed × latency`, with margin.** If this fails,
      stop and fix it — lower-voltage pack, wider lens, or different
      camera pitch. Everything downstream assumes it holds.
- [ ] ⛔ `pkill -STOP` releases the motors within ~1 s
- [ ] ⛔ A healthy 10-minute run shows no unexplained motor stutters
- [ ] The reverse never runs longer than `--reverse-for`
- [ ] The debug mask is recognisably right in **two** rooms
- [ ] 10 minutes of driving with no human rescue, however many contacts
- [ ] Frames while moving are usable, or a mitigation is chosen
- [ ] Baseline MDBC, coverage and stuck fraction are written down below

---

## Phase 3 — Make it measurable

**Goal:** no new capability. Every later phase claims an improvement,
and this is what lets you check.

### Do
- [ ] Restructure into the `tank/` package ([PLAN.md §7](PLAN.md))
- [ ] Root `teleop.py` / `record.py` / `roam.py` still work as wrappers
- [ ] Perception and arbiter run on plain arrays — no camera, no serial
- [ ] `tools/replay.py` runs a recorded session and writes an annotated video
- [ ] Telemetry to JSONL: clearances, decisions, timings
- [ ] Flask MJPEG live view over WiFi
- [ ] MDBC / coverage / stuck fraction computed from telemetry
- [ ] Instrument each latency stage separately
- [ ] Config in YAML instead of argparse flags

### Do — power and wiring
- [ ] Pi supply separated from the motor pack (or a 3 A+ buck with bulk caps)
- [ ] Common ground between Pi and ESP32
- [ ] Flyback diodes or snubbers across the motor terminals
- [ ] Motor leads short and twisted; ferrite bead on the USB cable
- [ ] `vcgencmd get_throttled` logged into telemetry

### Gate
- [ ] ⛔ `pytest` passes with no camera and no serial port attached
- [ ] ⛔ Replay reproduces the decisions a live session actually logged
- [ ] Live view sustains ≥5 fps over WiFi
- [ ] Automatic metrics agree with Phase 2's hand-counted baseline
- [ ] A 20-minute drive ends with `vcgencmd get_throttled` = `0x0`
- [ ] Measured latency per stage recorded below

---

## Phase 4 — Make the camera the odometer

**Goal:** yaw, speed and stall detection from optical flow. This
replaces the IMU and the bumper, and Phases 5, 6 and 8 all need it.

### Do
- [ ] Yaw rate from horizontal flow (sparse Lucas-Kanade, median shift)
- [ ] Forward speed from ground-plane flow
- [ ] Calibration table: spin °/s both directions · arc °/s and radius ·
      forward m/s · coast distance and coast angle
- [ ] Re-measure that table on carpet **and** on hard floor
- [ ] Closed-loop turns — spin until integrated yaw says done, not for a duration
- [ ] Stall detection: commanded forward + flow ≈ 0, debounced 2–3 frames
- [ ] Distinguish flow-zero-everywhere (wedged) from flow-zero-at-bottom (caught underneath)
- [ ] Low-confidence reporting when the feature count collapses

### Gate
- [ ] ⛔ Hand-turn the tank a full 360° → integrated yaw within ~10°
- [ ] ⛔ Driving into a wall triggers a stall stop within 400 ms, **10 out of 10**
- [ ] A commanded 90° spin lands within ~15° on three different surfaces
- [ ] Speed estimate within ~20% of the tape-and-stopwatch figure on textured floor
- [ ] On plain floor it reports **low confidence** rather than a confident wrong number
- [ ] Calibration table recorded below

---

## Phase 5 — Stop thinking in pixels

**Goal:** distances in metres, so the model has something real to
predict and the policy has something real to threshold.

### Do
- [ ] ~20 checkerboard shots → `cv2.calibrateCamera` → intrinsics + distortion
- [ ] Tape grid on the floor, ≥4 known points → `cv2.findHomography`
- [ ] Store the homography with the mount geometry beside it in config
- [ ] Convert per-column pixel heights into free distance per bearing
- [ ] Local costmap: ~4 m × 3 m, 5 cm cells, log-odds
- [ ] Shift the costmap using Phase 4 odometry
- [ ] Decay cells after 3–5 seconds
- [ ] Stall events stamp a hard obstacle at the contact point
- [ ] Port the reactive policy to read the costmap

### Gate
- [ ] ⛔ Projected distances match a tape measure within ~15% at
      0.5 / 1.0 / 1.5 / 2.0 m
- [ ] Free-distance-per-bearing is logged for every frame
- [ ] The ported policy reproduces the Phase 2 baseline within noise
      *(same behaviour, new representation — this is the proof the port was clean)*
- [ ] All clearance settings are in metres
- [ ] Re-seating the camera and re-running calibration is documented

---

## Phase 6 — The vision model

**Goal:** one network, two heads. Cost per heading drives the tank;
the floor mask is there so you can see it being wrong.

### Do — labels
- [ ] Generate geometric cost targets over the whole Phase 1 corpus
- [ ] Hand-label **300 validation frames**, covering every arena case, ≥2 rooms
- [ ] ⛔ Confirm the validation set is excluded from training

### Do — train and deploy
- [ ] Two-head net, 160×120 input
- [ ] Augment: brightness · contrast · colour jitter · blur · JPEG · mild perspective
- [ ] Train on a laptop or Colab, not on the Pi
- [ ] Export ONNX, quantise int8
- [ ] Compare fp32 vs int8 accuracy *(before blaming the architecture)*
- [ ] Bench on the Pi with `tools/bench.py`
- [ ] Arbiter: softmax → momentum prior → sample → **minimum commit time** → bearing → relay action
- [ ] Keep the HSV histogram running as a second opinion
- [ ] Raise the cost of bearings where the two disagree
- [ ] Collect autonomous sessions with stall labels, then **fine-tune on outcomes**

### Gate
- [ ] ⛔ ≥15 fps at 160×120 on the Pi, with the loop still at 10 Hz
- [ ] ⛔ Mask head beats the HSV mask on the hand-labelled set —
      **especially on the floor-coloured cardboard**
- [ ] ⛔ Live arena MDBC beats the Phase 2 baseline
- [ ] Outcome fine-tuning has actually happened
      *(stop at distillation and you have a faster histogram, not a better one)*
- [ ] Two runs from the same start produce visibly different paths
- [ ] Histogram still switchable by config if the net fails to load
- [ ] Direction changes per minute still within the relay budget

---

## Phase 7 — Offline depth teacher *(optional)*

**Goal:** catch obstacles that never touch the floor. Runs offline, so
it costs no fps.

**Do this only if Phase 8's failure log actually contains overhang
collisions.** Otherwise skip it.

### Do
- [ ] Run Depth Anything V2 Small (or MiDaS small) over the corpus offline
- [ ] Anchor relative depth to metres using the ground plane
- [ ] Fold overhang obstacles into the cost targets
- [ ] Retrain the Phase 6 student on the enriched targets

### Gate
- [ ] Overhangs appear in the offline cost targets
- [ ] The retrained student detects the arena overhang live
- [ ] ⛔ No fps regression — the student must be the same size
- [ ] ⛔ Arena MDBC improves. **If it does not, revert.**

---

## Phase 8 — Actually roaming

**Goal:** the stated goal. Not hitting things is also satisfied by
standing still; this is the part that makes it roam.

### Do
- [ ] Novelty prior — penalise recently-taken bearings and visited cells
- [ ] Anti-trap — detect a small-area orbit, commit to a long run or big turn
- [ ] Recovery state machine, every state with a timeout and an exit:
      `reverse 0.5s → turn 90° → resume` →
      `reverse 1.0s → turn 150° other way → resume` → `stop and alert`
- [ ] Turns in recovery close the loop on visual yaw
- [ ] Keep reverses short and rare — reversing is blind
- [ ] Frontier preference — bias toward unknown space, mildly
- [ ] Log direction changes per minute

### Gate
- [ ] ⛔ 10 minutes in the arena covers **≥70%** of accessible floor cells
- [ ] ⛔ Deliberately wedging the tank produces self-recovery **8 times out of 10**
- [ ] No spike in the heading histogram indicating a persistent orbit
- [ ] Two runs from the same start produce visibly different paths
- [ ] Direction changes per minute **≤ ~20** *(this decides whether the
      relay board lasts weeks or months)*

---

## Phase 9 — Soak and operations

**Goal:** thirty unattended minutes in a room it has never seen.

### Do
- [ ] systemd unit, autostart on boot, restart policy
- [ ] Hardware arming switch so it does not drive off the bench at power-up
- [ ] Log rotation by size; always keep the 30 s before any stall
- [ ] Battery voltage on an ESP32 ADC pin; park at a threshold
- [ ] Classify every contact from the soak runs into the
      [failure taxonomy](PLAN.md)
- [ ] Tune from the failure log, not from intuition
- [ ] Write the operator's guide, in the style of `roam.py`'s TUNING block

### Gate
- [ ] ⛔ Three 30-minute unattended runs, in three unfamiliar rooms,
      with no damaging contact
- [ ] ⛔ Every wedge recovered without human help
- [ ] Survives a full battery cycle without corrupting a session log
- [ ] Someone else can start it from the operator's guide alone

---

## Measurements log

Fill these in as you go. Later phases compare against them.

| Measurement | Phase | Value | Notes |
|---|---|---|---|
| Sensor mode / FOV | 1 | | full-array? |
| Speed at full forward | 2 | ___ m/s | avg of 3 runs |
| Near-field blind wedge | 2 | ___ cm | from front of tracks |
| Total latency, glass to relay | 2 | ___ ms | phone slow-mo |
| Stopping distance (speed × latency) | 2 | ___ cm | must be < blind wedge ⛔ |
| Camera pitch chosen | 2 | ___ ° | |
| `--threshold`, room 1 | 2 | | |
| `--threshold`, room 2 | 2 | | spread = brittleness |
| **Baseline MDBC** | 2 | ___ m | the number to beat |
| **Baseline coverage** | 2 | ___ % | 10 min in the arena |
| **Baseline stuck fraction** | 2 | ___ % | |
| Latency per stage | 3 | | capture / infer / smoother / serial |
| Spin rate, carpet | 4 | ___ °/s | |
| Spin rate, hard floor | 4 | ___ °/s | |
| Arc rate and radius | 4 | | |
| Coast distance and angle | 4 | | |
| Model fps at 160×120 | 6 | ___ fps | ≥15 |
| Model MDBC | 6 | ___ m | vs baseline |
| Direction changes / min | 8 | | ≤ ~20 |