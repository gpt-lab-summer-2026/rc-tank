/* =============================================================
   ESP32 + 4-relay module -> two DC motors, forward and reverse
   -------------------------------------------------------------
   Two relays per motor make an H-bridge. Serial Monitor at
   115200 drives them, and a scripted sequence runs on boot.

   -------------------------------------------------------------
   WIRING

     ESP32 GPIO 27 -> IN1     motor 1, relay A
     ESP32 GPIO 26 -> IN2     motor 1, relay B
     ESP32 GPIO 25 -> IN3     motor 2, relay A
     ESP32 GPIO 33 -> IN4     motor 2, relay B
     ESP32 GND     -> module GND        <-- required
     module VCC    -> 5V

   Screw terminals, per channel:

     COM -> its own motor terminal
     NO  -> battery +      (all four channels)
     NC  -> battery -      (all four channels)

   NC MUST be wired. With only COM and NO you get on/off but no
   reverse, because releasing both relays leaves the motor
   floating instead of tying both its terminals to one rail.

   Per motor:

     A off, B off -> both terminals on -, stopped (braked)
     A on,  B off -> forward
     A off, B on  -> reverse
     A on,  B on  -> both terminals on +, stopped (braked)

   Each COM sits on exactly one rail at all times, so there is no
   shoot-through state the way a transistor bridge has.

   -------------------------------------------------------------
   COMMANDS

     motor 1:   q forward    a reverse    z stop
     motor 2:   e forward    d reverse    c stop
     both:      w forward    s reverse    x stop

     t  run the full test sequence
     i  invert active level
     ?  help

   -------------------------------------------------------------
   RELAYS ARE NOT PWM DEVICES

   No speed control here. Contacts take about 10ms to settle and
   are rated for roughly 10,000 operations under load. Pulsing
   them wears the module out fast.
   ============================================================= */

const bool ACTIVE_LOW_DEFAULT = true;   // most 4-relay boards

// Stop before reversing. A spinning motor thrown straight into
// reverse draws well above stall current and arcs the contacts.
const unsigned long DEADTIME_MS = 300;

// Cuts a motor left running this long outside the test sequence,
// so a stall is not held indefinitely. 0 disables.
const unsigned long MAX_ON_MS = 10000;

bool activeLow = ACTIVE_LOW_DEFAULT;

struct Motor {
  const char *name;
  int pinA, pinB;
  int applied;                 // what the relays are doing now
  int target;                  // what we want
  unsigned long holdUntil;     // dead-time expiry
  unsigned long onSince;
};

Motor m1 = {"M1", 27, 26, 0, 0, 0, 0};
Motor m2 = {"M2", 25, 33, 0, 0, 0, 0};

// --------------------------- relays ---------------------------

inline void coil(int pin, bool on) {
  digitalWrite(pin, on == activeLow ? LOW : HIGH);
}

const char *word(int d) {
  return d > 0 ? "forward" : (d < 0 ? "reverse" : "stop");
}

void apply(Motor &m) {
  coil(m.pinA, m.applied > 0);
  coil(m.pinB, m.applied < 0);

  Serial.print("   ");
  Serial.print(m.name);
  Serial.print("  ");
  Serial.println(word(m.applied));
}

// Walks applied towards target, forcing a stop in between when
// the direction flips.
void tick(Motor &m) {
  if (m.target == m.applied) return;

  unsigned long now = millis();
  if (now < m.holdUntil) return;              // still coasting down

  if (m.applied != 0 && m.target != 0) {      // reversal: stop first
    m.applied = 0;
    m.holdUntil = now + DEADTIME_MS;
  } else {
    m.applied = m.target;
    if (m.applied != 0) m.onSince = now;
  }
  apply(m);
}

// ------------------------ test sequence ------------------------

struct Step { int d1, d2; unsigned long ms; const char *label; };

const Step SEQ[] = {
  { 0,  0, 1000, "settle"       },
  { 1,  0, 1500, "M1 forward"   },
  { 0,  0, 1000, "stop"         },
  {-1,  0, 1500, "M1 reverse"   },
  { 0,  0, 1000, "stop"         },
  { 0,  1, 1500, "M2 forward"   },
  { 0,  0, 1000, "stop"         },
  { 0, -1, 1500, "M2 reverse"   },
  { 0,  0, 1000, "stop"         },
  { 1,  1, 1500, "both forward" },
  { 0,  0, 1000, "stop"         },
  {-1, -1, 1500, "both reverse" },
  { 0,  0, 1000, "stop"         },
};
const int SEQ_LEN = sizeof(SEQ) / sizeof(SEQ[0]);

bool seqActive = false;
int  seqIndex = 0;
unsigned long seqStart = 0;

void enterStep(int i) {
  seqIndex = i;
  seqStart = millis();
  m1.target = SEQ[i].d1;
  m2.target = SEQ[i].d2;
  Serial.print("\n[");
  Serial.print(SEQ[i].label);
  Serial.println("]");
}

void startSeq() {
  seqActive = true;
  enterStep(0);
}

void runSeq() {
  if (!seqActive) return;
  if (millis() - seqStart < SEQ[seqIndex].ms) return;

  if (seqIndex + 1 >= SEQ_LEN) {
    seqActive = false;
    m1.target = 0;
    m2.target = 0;
    Serial.println("\nsequence complete. '?' for commands.\n");
    return;
  }
  enterStep(seqIndex + 1);
}

// --------------------------- serial ---------------------------

void help() {
  Serial.println();
  Serial.println("  motor 1:  q forward   a reverse   z stop");
  Serial.println("  motor 2:  e forward   d reverse   c stop");
  Serial.println("  both:     w forward   s reverse   x stop");
  Serial.println("  t test sequence   i invert   ? help");
  Serial.println();
}

void command(Motor &m, int dir) {
  seqActive = false;
  m.target = dir;
}

void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r' || c == '\n' || c == ' ') continue;

    switch (c) {
      case 'q': command(m1,  1); break;
      case 'a': command(m1, -1); break;
      case 'z': command(m1,  0); break;

      case 'e': command(m2,  1); break;
      case 'd': command(m2, -1); break;
      case 'c': command(m2,  0); break;

      case 'w': command(m1,  1); command(m2,  1); break;
      case 's': command(m1, -1); command(m2, -1); break;
      case 'x': command(m1,  0); command(m2,  0); break;

      case 't': startSeq(); break;

      case 'i':
        activeLow = !activeLow;
        Serial.print("active level now ");
        Serial.println(activeLow ? "low" : "high");
        apply(m1);
        apply(m2);
        break;

      case '?': help(); break;

      default:
        Serial.print("unknown: ");
        Serial.println(c);
    }
  }
}

// -------------------------- safety ---------------------------

void safety(Motor &m) {
  if (seqActive || m.applied == 0 || MAX_ON_MS == 0) return;
  if (millis() - m.onSince < MAX_ON_MS) return;

  Serial.print(m.name);
  Serial.println("  max on-time reached, stopping");
  m.target = 0;
}

// ---------------------------- main ----------------------------

void setup() {
  Serial.begin(115200);
  delay(300);

  int pins[] = {m1.pinA, m1.pinB, m2.pinA, m2.pinB};
  for (int p : pins) {
    pinMode(p, OUTPUT);
    coil(p, false);
  }

  Serial.println("\n=== two-motor relay test ===");
  Serial.println("all relays released. 3 seconds of quiet before");
  Serial.println("anything moves — nothing should be turning.");
  Serial.println("if a motor is already running, press 'i'.\n");

  delay(3000);
  startSeq();
}

void loop() {
  handleSerial();
  runSeq();

  tick(m1);
  tick(m2);

  safety(m1);
  safety(m2);
}

/* =============================================================
   READING THE RESULT

   One motor only goes one way
     Its NC terminal is not wired, or is on the wrong rail. That
     relay can reach + but never -, so one direction is dead.

   A motor runs the wrong way round
     Swap its two COM wires at the motor, or swap that motor's
     two GPIO numbers in the struct above.

   Both motors twitch at every direction change
     Normal. That is the dead-time stopping before reversing.
     You should hear a click, roughly a third of a second of
     silence, then the next click.

   Relays click, nothing turns
     Meter across a motor while its relay is closed. Near zero
     means the contact path is wrong. Full voltage with no
     movement means the motor is stalled or open.

   Voltage sags, motors barely turn
     Supply current. Two motors together pull more than a
     breadboard power module can give — separate pack.

   ESP32 resets when relays switch
     Coil current through the ESP32's regulator, or motor noise.
     Pull the JD-VCC jumper and feed the coils from their own 5V,
     and put a 100nF ceramic across each motor at the motor.

   Relays chatter or the board buzzes
     Coil supply is too weak to hold four coils. Same fix.
   ============================================================= */
