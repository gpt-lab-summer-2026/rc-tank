/* =============================================================
   ESP32 relay bridge — serial in, relays out
   -------------------------------------------------------------
   The Pi decides everything. This firmware holds no idea of what
   forward means, how a turn should look, or how long to hold it.
   It sets relay states and reports back.

   Flash it once. All driving logic lives in Python.

   -------------------------------------------------------------
   WIRING

     ESP32 GPIO 27 -> IN1     motor 1, relay A
     ESP32 GPIO 26 -> IN2     motor 1, relay B
     ESP32 GPIO 25 -> IN3     motor 2, relay A
     ESP32 GPIO 33 -> IN4     motor 2, relay B
     ESP32 GND     -> module GND
     module VCC    -> 5V

     COM -> its own motor terminal
     NO  -> battery +    (all four)
     NC  -> battery -    (all four)

   Pi connects over USB. 115200 baud.

   FIT A 10k PULL-UP FROM EACH OF IN1..IN4 TO 3V3.

   Between the moment this chip resets and the first line of
   setup() below, its pins are floating inputs. On an active-low
   board a floating input can read as "on", which holds the relays
   closed while no code is running to release them. Firmware cannot
   close that window — only the pull-ups can. Without them, a
   brownout can leave the motors driving with nothing in control.

   -------------------------------------------------------------
   PROTOCOL — one line in, one line out, always

     R a b c d     set relays IN1..IN4, each 0 or 1
     P             ping, changes nothing
     S             all relays off
     ?             report configuration
     C D <ms>      dead-time, 0 disables
     C W <ms>      watchdog, 0 disables
     C A <0|1>     active low (1) or active high (0)
     C S <ms>      stagger between the two motors, 0 disables

   Replies:

     ok a b c d up=<ms>      the states actually applied right now
     info ... up=<ms>        configuration
     err <reason>

   And two lines that arrive WITHOUT being asked for. They start
   with their own words so the Pi can tell them from a reply:

     boot reason=<why>       printed once, when this chip starts
     evt watchdog-released   printed when the watchdog drops the relays

   'boot' appearing mid-session means the bridge restarted under
   you. reason=BROWNOUT means the supply sagged, which on this
   hardware means the motors took the rail down. So does up=
   suddenly counting from near zero again.

   The reply to R reports what is APPLIED, which during a
   dead-time pause is not yet what you asked for. Send P a moment
   later to see it land.

   -------------------------------------------------------------
   THE THREE THINGS FIRMWARE STILL DOES

   Dead-time delays a reversal, it never changes one. Asking a
   spinning motor to reverse still reverses it, just deadtime= ms
   later with a stop inserted. The ESP32 does this because it knows
   when the contact settled and the Pi does not.

   It delays NOTHING ELSE. A reversal is the only order that waits:
   a stop, a start, or going back to the direction the motor was
   already turning is applied the moment the line lands, including
   part-way through a pause that some earlier order started. An
   order that arrives mid-pause and is still a reversal keeps
   waiting out the original delay rather than restarting it, so
   hammering the keys cannot stretch a pause indefinitely.

   The practical shape of that, per motor: pause only when the
   contacts are making one way and the new order makes them the
   other way. Once both coils are open, nothing is queued.

   The watchdog acts only when there is NO command. If the USB
   drops or the Pi dies, there is nothing to obey and the relays
   release. It cannot help if this chip is the thing that failed —
   see the pull-up note above.

   Stagger starts the two motors a few milliseconds apart. Both
   starting together draws double the inrush, and inrush is what
   sags a shared rail. Off by default; turn it on with C S 15 if
   boot reason ever comes back BROWNOUT.

   All three are off switches away:  C D 0, C W 0, C S 0.
   With the watchdog off, a crashed Pi leaves the motors running.
   Keep a physical kill switch on the motor battery.
   ============================================================= */

#include <Arduino.h>
#include <esp_system.h>

const int PIN_IN[4] = {27, 26, 25, 33};   // IN1, IN2, IN3, IN4

bool activeLow = true;                    // most 4-relay boards
unsigned long deadtimeMs = 300;
unsigned long watchdogMs = 400;
unsigned long staggerMs = 0;

unsigned long lastRx = 0;
bool watchdogTripped = false;
esp_reset_reason_t bootReason;

// Relays pair up per motor: (IN1,IN2) and (IN3,IN4). This is the
// only topology the firmware knows, and it exists so dead-time
// can tell a reversal from any other state change.
//
// The two delays are kept apart on purpose. They are unrelated —
// one protects the motor, the other protects the supply rail — and
// sharing a single timer between them is what used to make an
// unrelated command wait out a reversal it had nothing to do with.
struct Pair {
  int ia, ib;                 // indices into PIN_IN
  bool apA, apB;              // applied
  bool tgA, tgB;              // target
  unsigned long deadUntil;    // reversal pause expiry
  bool inDead;                // a reversal pause is running
  bool deadFrom;              // apA as it was when that pause began
  unsigned long staggerUntil; // inrush delay expiry
};

Pair pairs[2] = {
  {0, 1, false, false, false, false, 0, false, false, 0},
  {2, 3, false, false, false, false, 0, false, false, 0},
};

// ---------------------------- relays ----------------------------

inline void coil(int idx, bool on) {
  digitalWrite(PIN_IN[idx], on == activeLow ? LOW : HIGH);
}

void push(Pair &p) {
  coil(p.ia, p.apA);
  coil(p.ib, p.apB);
}

void pushAll() {
  push(pairs[0]);
  push(pairs[1]);
}

// Walks applied towards target. A direction flip goes through
// both-off first and waits out the dead-time. Everything else is
// applied the moment it arrives.
void tick(Pair &p) {
  if (p.tgA == p.apA && p.tgB == p.apB) {
    // Nothing outstanding. If a reversal pause was running, the move
    // it was holding back has since been withdrawn — a stop, usually
    // — so drop the pause rather than leave it timing out against an
    // order nobody is waiting for any more.
    p.inDead = false;
    return;
  }

  unsigned long now = millis();
  bool driving   = p.apA != p.apB;     // contacts are making, one way
  bool wantDrive = p.tgA != p.tgB;     // asked to keep making, either way

  // Stagger spreads inrush, so it only has anything to say about
  // orders that draw current. A stop is let straight through.
  if (wantDrive && now < p.staggerUntil) return;

  // The one change worth delaying: both coils open, then the other
  // way round once the armature has had a moment to stop pushing
  // back against it.
  if (deadtimeMs > 0 && driving && wantDrive && p.apA != p.tgA) {
    p.deadFrom = p.apA;
    p.inDead = true;
    p.deadUntil = now + deadtimeMs;
    p.apA = false;
    p.apB = false;
    push(p);
    return;
  }

  // Mid-pause. Keep waiting only while the standing order still
  // points the other way. Asking for the direction it was already
  // turning, or for a stop, is not a reversal and must not sit out
  // the rest of a delay that was protecting against a different move
  // entirely — waiting there is what made the controls feel
  // rate-limited, and it protected nothing, because the motor is
  // already stopped and the request is not a reversal.
  // deadtimeMs is re-read here rather than trusted from pause entry,
  // so C D 0 releases a pause already running instead of leaving one
  // last delay to time out after the feature was switched off.
  if (deadtimeMs > 0 && p.inDead && now < p.deadUntil &&
      wantDrive && p.tgA != p.deadFrom) return;

  p.inDead = false;
  p.apA = p.tgA;
  p.apB = p.tgB;
  push(p);
}

void targetsOff() {
  for (int i = 0; i < 2; i++) {
    pairs[i].tgA = false;
    pairs[i].tgB = false;
  }
}

// ---------------------------- replies ----------------------------

const char *resetReasonName(esp_reset_reason_t r) {
  switch (r) {
    case ESP_RST_POWERON:   return "power-on";
    case ESP_RST_EXT:       return "external-pin";
    case ESP_RST_SW:        return "software";
    case ESP_RST_PANIC:     return "PANIC";
    case ESP_RST_INT_WDT:   return "INT-WATCHDOG";
    case ESP_RST_TASK_WDT:  return "TASK-WATCHDOG";
    case ESP_RST_WDT:       return "WATCHDOG";
    case ESP_RST_DEEPSLEEP: return "deep-sleep-wake";
    case ESP_RST_BROWNOUT:  return "BROWNOUT";
    case ESP_RST_SDIO:      return "sdio";
    default:                return "unknown";
  }
}

void replyOk() {
  bool s[4];
  s[pairs[0].ia] = pairs[0].apA;
  s[pairs[0].ib] = pairs[0].apB;
  s[pairs[1].ia] = pairs[1].apA;
  s[pairs[1].ib] = pairs[1].apB;
  Serial.printf("ok %d %d %d %d up=%lu\n", s[0], s[1], s[2], s[3], millis());
}

void replyInfo() {
  Serial.printf(
    "info bridge=1 deadtime=%lu watchdog=%lu activelow=%d stagger=%lu "
    "boot=%s up=%lu\n",
    deadtimeMs, watchdogMs, activeLow ? 1 : 0, staggerMs,
    resetReasonName(bootReason), millis());
}

// ---------------------------- parsing ----------------------------

void handleLine(char *line) {
  lastRx = millis();
  watchdogTripped = false;

  switch (line[0]) {
    case 'R': case 'r': {
      int a, b, c, d;
      if (sscanf(line + 1, "%d %d %d %d", &a, &b, &c, &d) != 4) {
        Serial.println("err bad-args");
        return;
      }
      if ((a | b | c | d) & ~1) {         // anything other than 0 or 1
        Serial.println("err not-binary");
        return;
      }

      // Does each motor actually change? Asked before the targets
      // move, because that is what decides whether staggering them
      // buys anything.
      bool moves0 = (pairs[0].apA != (bool)a) || (pairs[0].apB != (bool)b);
      bool moves1 = (pairs[1].apA != (bool)c) || (pairs[1].apB != (bool)d);

      pairs[0].tgA = a; pairs[0].tgB = b;
      pairs[1].tgA = c; pairs[1].tgB = d;

      if (staggerMs > 0 && moves0 && moves1) {
        unsigned long due = millis() + staggerMs;
        if (pairs[1].staggerUntil < due) pairs[1].staggerUntil = due;
      }

      tick(pairs[0]);                     // apply now if nothing pending
      tick(pairs[1]);
      replyOk();
      return;
    }

    case 'P': case 'p':
      replyOk();
      return;

    case 'S': case 's':
      targetsOff();
      tick(pairs[0]);
      tick(pairs[1]);
      replyOk();
      return;

    case '?':
      replyInfo();
      return;

    case 'C': case 'c': {
      char which;
      int val;
      if (sscanf(line + 1, " %c %d", &which, &val) != 2) {
        Serial.println("err bad-args");
        return;
      }
      if (val < 0) val = 0;
      switch (toupper(which)) {
        case 'D': deadtimeMs = val; break;
        case 'W': watchdogMs = val; break;
        case 'S': staggerMs  = val; break;
        case 'A': activeLow = (val != 0); pushAll(); break;
        default:  Serial.println("err bad-key"); return;
      }
      replyInfo();
      return;
    }

    default:
      Serial.println("err bad-command");
  }
}

char buf[48];
int  blen = 0;

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (blen == 0) continue;            // ignore blank lines
      buf[blen] = '\0';
      handleLine(buf);
      blen = 0;
    } else if (blen < (int)sizeof(buf) - 1) {
      buf[blen++] = c;
    } else {
      blen = 0;                           // overlong, drop it
      Serial.println("err too-long");
    }
  }
}

// ----------------------------- main -----------------------------

void setup() {
  // Pins first, ahead of everything including Serial. Every
  // millisecond spent before this line is a millisecond the relays
  // are answering to a floating input rather than to us.
  //
  // Latch the level BEFORE switching the pin to an output: a fresh
  // output starts low, and low is exactly what an active-low board
  // reads as "on". Writing first means it comes up already released.
  for (int i = 0; i < 4; i++) {
    coil(i, false);
    pinMode(PIN_IN[i], OUTPUT);
    coil(i, false);
  }

  bootReason = esp_reset_reason();

  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.printf("boot reason=%s\n", resetReasonName(bootReason));

  lastRx = millis();
}

void loop() {
  readSerial();

  // Nothing heard for too long means there is no command to obey.
  // Say so once, rather than every pass — silence here has been
  // mistaken for the watchdog working before now.
  if (watchdogMs > 0 && millis() - lastRx > watchdogMs) {
    if (!watchdogTripped) {
      watchdogTripped = true;
      Serial.println("evt watchdog-released");
    }
    targetsOff();
  }

  tick(pairs[0]);
  tick(pairs[1]);
}