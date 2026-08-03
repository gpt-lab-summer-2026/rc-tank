/* =============================================================
   ESP32 -> RC remote potentiometer replacement (DAC)
   -------------------------------------------------------------
   The remote's sticks are potentiometers. Each one is a voltage
   divider across the remote's 2.7 V rail, and the MCU reads the
   wiper on an ADC. That is the whole signal — a DC level, no
   clock, no protocol.

   So we replace the wiper with an ESP32 DAC and output discrete
   levels. Precise control is not needed, so three per axis:
   full one way, neutral, full the other way.

   -------------------------------------------------------------
   WIRING  (measured: red = 2.7 V rail, yellow = 0 V, black = wiper)

     yellow -> ESP32 GND                      (common ground)
     black  -> 1k resistor -> ESP32 GPIO 25   (throttle wiper)
     red    -> LEAVE DISCONNECTED, insulate it

   Red is the supply. Landing a GPIO on it shorts the remote's
   battery through an ESP32 output transistor the moment that pin
   drives low.

   Second stick, same pattern:
     its yellow -> ESP32 GND
     its black  -> 1k -> ESP32 GPIO 26        (steering wiper)
     its red    -> disconnected

   Trigger, if it really is a switch:
     one side -> ESP32 GND
     other    -> ESP32 GPIO 13   (open-drain, see below)

   GPIO 25 and 26 are the only true DACs on a classic ESP32,
   which is exactly two sticks.

   -------------------------------------------------------------
   BEFORE YOU CUT ANYTHING

   Reconnect the pot temporarily, power the remote, and meter the
   wiper (middle pin) against yellow while moving the stick. You
   should see a smooth sweep. Write down three numbers:

     full one way ... V     -> V_FULL_REV
     centred ........ V     -> V_NEUTRAL
     full other way . V     -> V_FULL_FWD

   Put them in the constants below. Defaults assume a clean
   0 / 1.35 / 2.7 V sweep with a little margin off each rail.

   Then lift the wiper (middle pin) out of the circuit. If the
   pot stays connected it fights the DAC across the wiper node,
   and the pot's low impedance wins — you will measure the right
   voltage at the ESP32 and see nothing happen at the car.

   -------------------------------------------------------------
   POWER-UP ORDER

   Power the ESP32 first, let it settle at neutral, then switch
   the remote on. Many toy remotes sample their sticks at boot to
   find centre. Before setup() runs, GPIO 25/26 are floating.
   ============================================================= */

#include <WiFi.h>
#include <WebServer.h>

const char* AP_SSID = "ESP32-RC-Sim";
const char* AP_PASS = "drive1234";      // min 8 chars, or "" for open

const int PIN_THR_DAC = 25;
const int PIN_STR_DAC = 26;
const int PIN_TRIGGER = 13;             // open-drain switch contact

// The remote's rail, measured. Nothing is ever driven above this.
const float REMOTE_VCC = 2.70;

// Measured wiper voltages. Small margins keep us off both rails.
const float V_FULL_REV = 0.15;
const float V_NEUTRAL  = 1.35;
const float V_FULL_FWD = 2.55;

const float V_FULL_LEFT  = 0.15;
const float V_NEUTRAL_S  = 1.35;
const float V_FULL_RIGHT = 2.55;

const int DEADZONE = 30;        // stick % before it snaps off neutral
const int FAILSAFE_MS = 400;

// The ESP32 DAC spans 0..255 over roughly 0..3.3 V.
const float ESP_VREF = 3.30;

WebServer server(80);
unsigned long lastCmd = 0;

int  thr = 0, str = 0;          // -100 .. +100
bool trig = false;
bool calMode = false;           // raw DAC sweep for finding levels
int  dacThr = 0, dacStr = 0;

// ------------------------ level helpers ------------------------

int countFor(float volts) {
  int c = (int)(volts / ESP_VREF * 255.0 + 0.5);
  return constrain(c, 0, 255);
}

// Hard ceiling. Above the remote's rail we forward-bias its input
// protection diode and push current into its battery.
int dacCeiling() { return countFor(REMOTE_VCC); }

int levelFor(int stick, float lo, float mid, float hi) {
  if (stick >= DEADZONE)  return countFor(hi);
  if (stick <= -DEADZONE) return countFor(lo);
  return countFor(mid);
}

inline void contact(int pin, bool pressed) {
  digitalWrite(pin, pressed ? LOW : HIGH);   // HIGH = high-Z = open
}

// --------------------------- outputs ---------------------------

void applyOutputs() {
  int ceil = dacCeiling();

  if (calMode) {
    // Sweep the full usable range so you can find the levels the
    // remote actually responds to.
    dacThr = map(thr, -100, 100, 0, ceil);
    dacStr = map(str, -100, 100, 0, ceil);
  } else {
    dacThr = levelFor(thr, V_FULL_REV,  V_NEUTRAL,   V_FULL_FWD);
    dacStr = levelFor(str, V_FULL_LEFT, V_NEUTRAL_S, V_FULL_RIGHT);
  }

  dacThr = constrain(dacThr, 0, ceil);
  dacStr = constrain(dacStr, 0, ceil);

  dacWrite(PIN_THR_DAC, dacThr);
  dacWrite(PIN_STR_DAC, dacStr);
  contact(PIN_TRIGGER, trig);
}

// Neutral, not zero. Zero would be full reverse.
void failsafe() {
  thr = 0; str = 0; trig = false;
  applyOutputs();
}

// ------------------------- control page -------------------------

const char PAGE[] PROGMEM = R"HTML(<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>RC Sim</title><style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-user-select:none;user-select:none;
  -webkit-tap-highlight-color:transparent}
html,body{height:100%;overflow:hidden;background:#0e1013;color:#e8eaed;
  font:500 14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
body{display:flex;flex-direction:column;align-items:center;gap:11px;
  padding:12px 12px 16px;touch-action:none}
header{display:flex;align-items:center;gap:9px;letter-spacing:.14em;
  font-size:11px;text-transform:uppercase;color:#7d848c}
#dot{width:8px;height:8px;border-radius:50%;background:#39d353;
  box-shadow:0 0 9px #39d353;transition:background .2s}
#dot.bad{background:#f0503a;box-shadow:0 0 9px #f0503a}
.lbl{font-size:10px;letter-spacing:.16em;text-transform:uppercase;
  color:#666d75;text-align:center;margin-bottom:4px}
.track{position:relative;background:#171b1f;border:1px solid #262b31;
  touch-action:none;overflow:hidden}
.track::before{content:"";position:absolute;background:#1e2429}
#thrTrack{width:96px;height:170px;border-radius:48px}
#thrTrack::before{left:0;right:0;top:35%;height:30%}
#strTrack{width:276px;height:88px;border-radius:44px}
#strTrack::before{top:0;bottom:0;left:35%;width:30%}
.knob{position:absolute;left:50%;top:50%;width:68px;height:68px;
  margin:-34px 0 0 -34px;border-radius:50%;background:#e8eaed;
  box-shadow:0 5px 16px rgba(0,0,0,.65);transition:transform .13s ease-out}
.knob.live{transition:none}
#trig{width:276px;height:50px;border-radius:13px;background:#171b1f;
  border:1px solid #262b31;color:#9aa2aa;font:600 12px/50px ui-monospace,monospace;
  letter-spacing:.2em;text-align:center;touch-action:none}
#trig.on{background:#e8eaed;color:#0e1013;border-color:#e8eaed}
.opts{display:flex;gap:18px}
.opt{display:flex;align-items:center;gap:6px;font-size:10px;
  letter-spacing:.12em;text-transform:uppercase;color:#666d75}
.opt input{width:15px;height:15px;accent-color:#e8eaed}
#readout{font-size:11px;letter-spacing:.06em;color:#8d949c;text-align:center}
</style></head><body>

<header><span id="dot"></span><span id="status">connected</span></header>

<div><div class="lbl">throttle</div>
  <div class="track" id="thrTrack"><div class="knob" id="thrKnob"></div></div></div>

<div><div class="lbl">steering</div>
  <div class="track" id="strTrack"><div class="knob" id="strKnob"></div></div></div>

<div id="trig">TRIGGER</div>

<div class="opts">
  <label class="opt"><input type="checkbox" id="latch"> latch</label>
  <label class="opt"><input type="checkbox" id="cal"> calibrate</label>
</div>

<div id="readout">--</div>

<script>
var busy=false, trig=false;

function makeAxis(trackId, knobId, vertical){
  var track=document.getElementById(trackId), knob=document.getElementById(knobId);
  var drag=false, val=0;
  function travel(){
    var r=track.getBoundingClientRect();
    var span = vertical ? r.height : r.width;
    var size = vertical ? knob.offsetHeight : knob.offsetWidth;
    return (span - size)/2 - 4;
  }
  function place(e){
    var r=track.getBoundingClientRect(), T=travel();
    var d = vertical ? e.clientY-(r.top+r.height/2) : e.clientX-(r.left+r.width/2);
    if(d> T) d= T;
    if(d<-T) d=-T;
    knob.style.transform = vertical ? 'translateY('+d+'px)' : 'translateX('+d+'px)';
    val = Math.round((vertical ? -d : d)/T*100);
  }
  function release(){
    drag=false; val=0;
    knob.classList.remove('live');
    knob.style.transform='translate(0,0)';
    send();
  }
  track.addEventListener('pointerdown',function(e){
    drag=true; knob.classList.add('live');
    track.setPointerCapture(e.pointerId); place(e);
  });
  track.addEventListener('pointermove',function(e){ if(drag) place(e); });
  track.addEventListener('pointerup',release);
  track.addEventListener('pointercancel',release);
  return { get:function(){ return val; } };
}

var thrAxis = makeAxis('thrTrack','thrKnob',true);
var strAxis = makeAxis('strTrack','strKnob',false);

var trigEl=document.getElementById('trig'), latchEl=document.getElementById('latch');
trigEl.addEventListener('pointerdown',function(e){
  trigEl.setPointerCapture(e.pointerId);
  trig = latchEl.checked ? !trig : true;
  trigEl.classList.toggle('on',trig); send();
});
function up(){ if(!latchEl.checked){ trig=false; trigEl.classList.remove('on'); send(); } }
trigEl.addEventListener('pointerup',up);
trigEl.addEventListener('pointercancel',up);

// In calibrate mode the sticks sweep the DAC continuously instead
// of snapping to three levels, so you can find what the remote
// actually responds to. The readout is the board's real output.
function send(){
  if(busy) return;
  busy=true;
  fetch('/c?t='+thrAxis.get()+'&s='+strAxis.get()
        +'&b='+(trig?1:0)+'&c='+(document.getElementById('cal').checked?1:0))
    .then(function(r){ return r.text(); })
    .then(function(txt){
      document.getElementById('dot').className='';
      document.getElementById('status').textContent='connected';
      document.getElementById('readout').textContent=txt;
    })
    .catch(function(){
      document.getElementById('dot').className='bad';
      document.getElementById('status').textContent='signal lost';
      document.getElementById('readout').textContent='--';
    })
    .then(function(){ busy=false; });
}
setInterval(send,70);
</script></body></html>)HTML";

// --------------------------- handlers ---------------------------

void handleRoot() { server.send_P(200, "text/html", PAGE); }

void handleCmd() {
  thr  = constrain(server.hasArg("t") ? server.arg("t").toInt() : 0, -100, 100);
  str  = constrain(server.hasArg("s") ? server.arg("s").toInt() : 0, -100, 100);
  trig = server.hasArg("b") && server.arg("b").toInt() != 0;
  calMode = server.hasArg("c") && server.arg("c").toInt() != 0;
  lastCmd = millis();

  applyOutputs();

  // Millivolts as integers — avoids float formatting on the ESP32.
  int mvT = (int)(dacThr * ESP_VREF * 1000.0 / 255.0);
  int mvS = (int)(dacStr * ESP_VREF * 1000.0 / 255.0);

  char body[80];
  snprintf(body, sizeof(body), "THR %3d = %d mV   STR %3d = %d mV   TRIG %d",
           dacThr, mvT, dacStr, mvS, trig ? 1 : 0);
  server.send(200, "text/plain", body);
}

// ----------------------------- setup -----------------------------

void setup() {
  Serial.begin(115200);

  // Open-drain: LOW presses, HIGH releases to high-impedance,
  // which is what an open switch actually is. Never drive HIGH.
  pinMode(PIN_TRIGGER, OUTPUT_OPEN_DRAIN);
  digitalWrite(PIN_TRIGGER, HIGH);

  // Sit at neutral immediately, before the remote is switched on.
  failsafe();

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("Join \"");
  Serial.print(AP_SSID);
  Serial.print("\" then open http://");
  Serial.println(WiFi.softAPIP());   // 192.168.4.1
  Serial.print("DAC ceiling for ");
  Serial.print(REMOTE_VCC);
  Serial.print(" V rail: ");
  Serial.println(dacCeiling());

  server.on("/", handleRoot);
  server.on("/c", handleCmd);
  server.onNotFound(handleRoot);
  server.begin();

  lastCmd = millis();
}

void loop() {
  server.handleClient();
  if (millis() - lastCmd > FAILSAFE_MS) failsafe();
}

/* =============================================================
   BRING-UP, NOTHING CONNECTED TO THE REMOTE

   Meter from GPIO 25 to ESP32 GND. Work the throttle stick:

     centre    -> about 1.35 V
     full up   -> about 2.55 V
     full down -> about 0.15 V

   The on-screen readout should agree with the meter. If it does,
   the whole chain works and you can wire it in.

   -------------------------------------------------------------
   FINDING THE RIGHT LEVELS

   Tick "calibrate". The sticks now sweep the DAC continuously
   from 0 to the rail ceiling instead of snapping to three levels,
   and the readout shows the exact count and millivolts.

   Car on a block, remote on. Ease the stick up until the car
   starts moving, note the millivolts. Keep going to full travel,
   note that too. Those are your real numbers — put them in
   V_FULL_FWD and friends, untick calibrate, and the three-level
   mode will hit them exactly.

   This is also how you find out whether the car is proportional.
   If speed rises smoothly with voltage, you can have proper
   throttle for free by adding more levels, or just leaving
   calibrate mode on permanently.

   -------------------------------------------------------------
   TROUBLESHOOTING

   Nothing happens, readout looks right
     The pot's wiper is probably still connected. It has to be
     lifted out of the circuit or it overpowers the DAC.

   Car runs off the instant the remote powers on
     The remote sampled a floating DAC at boot. Power the ESP32
     first and confirm it is sitting at neutral before switching
     the remote on.

   Works one direction, dead the other
     V_NEUTRAL is wrong. Meter the real centre with the original
     pot before you trust 1.35 V.

   Remote's LED flashes error / refuses to arm
     Stick calibration at boot failed. Some remotes want the
     sticks held at centre for a second or two after power-up —
     the sketch does that automatically, so check the ceiling
     value printed on the serial monitor is sane.
   ============================================================= */
