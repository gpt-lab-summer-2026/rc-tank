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

   -------------------------------------------------------------
   PROTOCOL — one line in, one line out, always

     R a b c d     set relays IN1..IN4, each 0 or 1
     P             ping, changes nothing
     S             all relays off
     ?             report configuration
     C D <ms>      dead-time, 0 disables
     C W <ms>      watchdog, 0 disables
     C A <0|1>     active low (1) or active high (0)

   Replies:

     ok a b c d    the states actually applied right now
     info ...      configuration
     err <reason>

   The reply to R reports what is APPLIED, which during a
   dead-time pause is not yet what you asked for. Send P a moment
   later to see it land.

   -------------------------------------------------------------
   THE TWO THINGS FIRMWARE STILL DOES

   Dead-time delays a reversal, it never changes one. Asking a
   spinning motor to reverse still reverses it, just up to
   DEADTIME_MS later with a stop inserted. The ESP32 does this
   because it knows when the contact settled and the Pi does not.

   The watchdog acts only when there is NO command. If the USB
   drops or the Pi dies, there is nothing to obey and the relays
   release.

   Both are off switches away:  C D 0  and  C W 0.
   With the watchdog off, a crashed Pi leaves the motors running.
   Keep a physical kill switch on the motor battery.
   ============================================================= */

#include <Arduino.h>

const int PIN_IN[4] = {27, 26, 25, 33};   // IN1, IN2, IN3, IN4

bool activeLow = true;                    // most 4-relay boards
unsigned long deadtimeMs = 300;
unsigned long watchdogMs = 400;

unsigned long lastRx = 0;

// Relays pair up per motor: (IN1,IN2) and (IN3,IN4). This is the
// only topology the firmware knows, and it exists so dead-time
// can tell a reversal from any other state change.
struct Pair {
  int ia, ib;                 // indices into PIN_IN
  bool apA, apB;              // applied
  bool tgA, tgB;              // target
  unsigned long holdUntil;
};

Pair pairs[2] = {
  {0, 1, false, false, false, false, 0},
  {2, 3, false, false, false, false, 0},
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
// both-off first and waits out the dead-time.
void tick(Pair &p) {
  if (p.tgA == p.apA && p.tgB == p.apB) return;

  unsigned long now = millis();
  if (now < p.holdUntil) return;

  bool flipping = deadtimeMs > 0 &&
                  (p.apA != p.apB) &&      // currently driving
                  (p.tgA != p.tgB) &&      // asked to keep driving
                  (p.apA != p.tgA);        // the other way

  if (flipping) {
    p.apA = false;
    p.apB = false;
    p.holdUntil = now + deadtimeMs;
  } else {
    p.apA = p.tgA;
    p.apB = p.tgB;
  }
  push(p);
}

void targetsOff() {
  for (int i = 0; i < 2; i++) {
    pairs[i].tgA = false;
    pairs[i].tgB = false;
  }
}

// ---------------------------- replies ----------------------------

void replyOk() {
  bool s[4];
  s[pairs[0].ia] = pairs[0].apA;
  s[pairs[0].ib] = pairs[0].apB;
  s[pairs[1].ia] = pairs[1].apA;
  s[pairs[1].ib] = pairs[1].apB;
  Serial.printf("ok %d %d %d %d\n", s[0], s[1], s[2], s[3]);
}

void replyInfo() {
  Serial.printf("info bridge=1 deadtime=%lu watchdog=%lu activelow=%d\n",
                deadtimeMs, watchdogMs, activeLow ? 1 : 0);
}

// ---------------------------- parsing ----------------------------

void handleLine(char *line) {
  lastRx = millis();

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
      pairs[0].tgA = a; pairs[0].tgB = b;
      pairs[1].tgA = c; pairs[1].tgB = d;
      tick(pairs[0]);                     // apply now if no flip pending
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
  Serial.begin(115200);

  for (int i = 0; i < 4; i++) {
    pinMode(PIN_IN[i], OUTPUT);
    coil(i, false);
  }

  delay(200);
  Serial.println();
  Serial.println("info relay bridge ready");
  replyInfo();

  lastRx = millis();
}

void loop() {
  readSerial();

  // Nothing heard for too long means there is no command to obey.
  if (watchdogMs > 0 && millis() - lastRx > watchdogMs) targetsOff();

  tick(pairs[0]);
  tick(pairs[1]);
}
