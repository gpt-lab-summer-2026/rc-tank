/* =============================================================
   ESP32 single relay + DC motor test
   -------------------------------------------------------------
   Smallest useful test: click one relay, watch one motor.

     ESP32 GPIO 27  -> relay module IN1
     ESP32 GND      -> relay module GND      <-- required
     relay VCC      -> 5V

   Motor goes on the screw terminals of channel 1, never on the
   IN1 header pin:

     motor lead 1   -> COM
     motor lead 2   -> battery -
     battery +      -> NO

   So the relay interrupts the positive side. Closed = motor runs.
   One relay is on/off only; reversing needs a second relay.

   -------------------------------------------------------------
   WHAT IT DOES

   On boot: holds the relay released for 3 seconds, then runs 3
   slow on/off cycles by itself, then waits for serial commands.

   Serial Monitor at 115200:

     1   relay on
     0   relay off
     t   toggle
     c   start / stop continuous cycling
     i   invert active level (if the logic is backwards)
     ?   show this list

   -------------------------------------------------------------
   IF THE RELAY IS ON WHEN IT SHOULD BE OFF

   Most 4-relay boards are active low: the input is pulled to
   ground to energise. If yours behaves backwards, press 'i' and
   it flips at runtime — then set ACTIVE_LOW_DEFAULT to match so
   it comes up correctly next time.

   Getting this backwards means the motor runs the instant the
   ESP32 boots, which is why the sketch idles for 3 seconds
   before doing anything.
   ============================================================= */

const int RELAY_PIN = 27;

const bool ACTIVE_LOW_DEFAULT = false;   // most modules

const unsigned long ON_MS  = 2000;      // auto-cycle timings
const unsigned long OFF_MS = 2000;
const int  STARTUP_CYCLES  = 3;

// Cuts the relay if it has been closed this long outside cycling
// mode, so a stalled motor is not left energised. 0 disables.
const unsigned long MAX_ON_MS = 10000;

bool activeLow = ACTIVE_LOW_DEFAULT;
bool relayOn   = false;
int  pinLevel  = HIGH;

bool cycling = true;
int  cyclesLeft = STARTUP_CYCLES;       // negative means forever
unsigned long lastChange = 0;
unsigned long onSince = 0;

// ------------------------------------------------------------

void setRelay(bool on) {
  relayOn  = on;
  pinLevel = (on == activeLow) ? LOW : HIGH;
  digitalWrite(RELAY_PIN, pinLevel);
  if (on) onSince = millis();

  Serial.print(on ? "  ON   " : "  off  ");
  Serial.print("GPIO ");
  Serial.print(RELAY_PIN);
  Serial.print(" = ");
  Serial.print(pinLevel == HIGH ? "HIGH" : "LOW ");
  Serial.print("   (active ");
  Serial.print(activeLow ? "low)" : "high)");
  Serial.println();
}

void help() {
  Serial.println();
  Serial.println("  1  on      0  off     t  toggle");
  Serial.println("  c  cycle   i  invert  ?  help");
  Serial.println();
}

void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r' || c == '\n' || c == ' ') continue;

    switch (c) {
      case '1': cycling = false; setRelay(true);  break;
      case '0': cycling = false; setRelay(false); break;
      case 't': cycling = false; setRelay(!relayOn); break;

      case 'c':
        cycling = !cycling;
        cyclesLeft = -1;                  // forever
        lastChange = millis();
        Serial.println(cycling ? "cycling" : "stopped");
        if (!cycling) setRelay(false);
        break;

      case 'i':
        activeLow = !activeLow;
        Serial.print("active level now ");
        Serial.println(activeLow ? "low" : "high");
        setRelay(relayOn);                // re-apply with new polarity
        break;

      case '?': help(); break;

      default:
        Serial.print("unknown: ");
        Serial.println(c);
    }
  }
}

void runCycle() {
  if (!cycling) return;

  unsigned long now = millis();
  unsigned long due = relayOn ? ON_MS : OFF_MS;
  if (now - lastChange < due) return;

  lastChange = now;

  if (relayOn) {
    setRelay(false);
    if (cyclesLeft > 0 && --cyclesLeft == 0) {
      cycling = false;
      Serial.println("\nauto cycles done. serial control is live, '?' for help.");
    }
  } else {
    setRelay(true);
  }
}

void safetyCutoff() {
  if (cycling || !relayOn || MAX_ON_MS == 0) return;
  if (millis() - onSince < MAX_ON_MS) return;

  Serial.println("max on-time reached, cutting relay");
  setRelay(false);
}

// ------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(300);

  pinMode(RELAY_PIN, OUTPUT);
  setRelay(false);

  Serial.println("\n=== relay + motor test ===");
  Serial.println("relay released. listening for 3 seconds before");
  Serial.println("anything moves — check it is quiet and still.");
  Serial.println("if the motor is already running, press 'i'.\n");

  delay(3000);

  Serial.println("starting 3 auto cycles...");
  lastChange = millis();
}

void loop() {
  handleSerial();
  runCycle();
  safetyCutoff();
}

/* =============================================================
   READING THE RESULT

   Relay clicks, motor spins
     Working. Wire the second relay per the H-bridge pattern when
     you want reverse.

   Relay clicks, motor does nothing
     The logic side is fine and the problem is downstream. Meter
     across the motor terminals while the relay is closed. Near
     zero volts means the contact path is wrong — check the motor
     is on COM and NO, not COM and NC. Full voltage with no
     movement means the motor is stalled or open.

   Motor runs the moment the ESP32 boots
     Polarity is inverted. Press 'i', then change
     ACTIVE_LOW_DEFAULT.

   No click at all
     Coil supply. The module needs 5V on VCC, and its GND must be
     bonded to the ESP32's. Meter VCC to GND at the module.

   Voltage sags and the motor barely turns
     Supply current. A breadboard power module is built for logic
     and will fold under motor load — give the motor its own pack.

   ESP32 resets when the relay closes
     Coil current through the ESP32's regulator, or motor noise.
     Feed the coils separately and put a 100nF ceramic across the
     motor terminals at the motor.
   ============================================================= */
