/* =============================================================
   ESP32 -> RC remote button simulator
   -------------------------------------------------------------
   The original remote keeps transmitting on its own. The ESP32
   only closes its switch contacts. Nothing about the A7105 link
   is touched.

   Simulates two single-axis joysticks plus one trigger button.
   Phone joins the ESP32's WiFi, opens http://192.168.4.1.

   -------------------------------------------------------------
   OUTPUTS

     GPIO 32 -> throttle forward
     GPIO 33 -> throttle reverse
     GPIO 27 -> steering left
     GPIO 14 -> steering right
     GPIO 13 -> trigger

   Left deliberately free:
     GPIO 5/18/19/23 - the VSPI set, if you later add an A7105
     GPIO 25/26      - the DACs, if a control turns out to be a pot

   -------------------------------------------------------------
   WIRING — one optocoupler per control, across the switch

     ESP32 GPIO --[ 220R ]-- PC817 pin 1 (LED anode)
     ESP32 GND ------------- PC817 pin 2 (LED cathode)
     PC817 pin 4 (collector) - one side of the remote's switch
     PC817 pin 3 (emitter) --- other side of the remote's switch

   Solder across the switch, not in place of it. The remote's own
   buttons keep working, which makes it obvious whether a failure
   is your side or the remote's.

   The opto isolates the two supplies, so it doesn't matter what
   voltage the remote runs at or which side of its switch is
   ground. If a control refuses to trigger, swap that opto's
   pins 3 and 4 — some switches sit on the high side.

   A 4- or 8-channel opto breakout board works fine and saves
   building five of these. Check whether its inputs are active
   high or active low and set ACTIVE_HIGH to match.
   ============================================================= */

#include <WiFi.h>
#include <WebServer.h>

const char* AP_SSID = "ESP32-RC-Sim";
const char* AP_PASS = "drive1234";      // min 8 chars, or "" for open

const int PIN_FWD  = 32;
const int PIN_REV  = 33;
const int PIN_LEFT = 27;
const int PIN_RGHT = 14;
const int PIN_TRIG = 13;

// True if driving an optocoupler LED directly. Set false for relay
// or opto boards with active-low inputs.
const bool ACTIVE_HIGH = true;

// Stick travel (%) before a direction asserts. The controls are
// on/off, so this is just how far you push before it fires.
const int DEADZONE = 25;

const int FAILSAFE_MS = 400;   // release everything if the phone goes quiet

// The remote only knows on and off. Pulsing a contact faster than
// the eye and slower than the RF frame (~20ms) fakes a throttle.
// Off by default so testing shows exactly what you asked for.
const bool SOFT_PWM = false;
const int  SOFT_PERIOD_MS = 80;

// Trigger held vs toggled. The web UI can override this per press.
const bool TRIGGER_LATCHES = false;

WebServer server(80);
unsigned long lastCmd = 0;

int  thr = 0;          // -100 (reverse) .. +100 (forward)
int  str = 0;          // -100 (left)    .. +100 (right)
bool trig = false;

bool outState[5] = {false, false, false, false, false};   // F R L R T

// --------------------------- outputs ---------------------------

inline void writePin(int pin, bool on) {
  digitalWrite(pin, on == ACTIVE_HIGH ? HIGH : LOW);
}

void applyOutputs() {
  int phase = SOFT_PWM ? (int)((millis() % SOFT_PERIOD_MS) * 100 / SOFT_PERIOD_MS)
                       : -1;

  int athr = abs(thr), astr = abs(str);
  bool thrLive = athr >= DEADZONE && (!SOFT_PWM || phase < athr);
  bool strLive = astr >= DEADZONE && (!SOFT_PWM || phase < astr);

  // Opposing directions are never asserted together — on some
  // remotes that's an undefined code, on others a dead short.
  outState[0] = thrLive && thr > 0;
  outState[1] = thrLive && thr < 0;
  outState[2] = strLive && str < 0;
  outState[3] = strLive && str > 0;
  outState[4] = trig;

  writePin(PIN_FWD,  outState[0]);
  writePin(PIN_REV,  outState[1]);
  writePin(PIN_LEFT, outState[2]);
  writePin(PIN_RGHT, outState[3]);
  writePin(PIN_TRIG, outState[4]);
}

void releaseAll() {
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
body{display:flex;flex-direction:column;align-items:center;gap:14px;
  padding:14px 12px 20px;touch-action:none}
header{display:flex;align-items:center;gap:9px;letter-spacing:.14em;
  font-size:11px;text-transform:uppercase;color:#7d848c}
#dot{width:8px;height:8px;border-radius:50%;background:#39d353;
  box-shadow:0 0 9px #39d353;transition:background .2s,box-shadow .2s}
#dot.bad{background:#f0503a;box-shadow:0 0 9px #f0503a}

.lbl{font-size:10px;letter-spacing:.16em;text-transform:uppercase;
  color:#666d75;text-align:center;margin-bottom:5px}

.track{position:relative;background:#171b1f;border:1px solid #262b31;
  touch-action:none;overflow:hidden}
/* the shaded band is the deadzone — inside it, nothing fires */
.track::before{content:"";position:absolute;background:#1e2429}
#thrTrack{width:104px;height:190px;border-radius:52px}
#thrTrack::before{left:0;right:0;top:37.5%;height:25%}
#strTrack{width:280px;height:96px;border-radius:48px}
#strTrack::before{top:0;bottom:0;left:37.5%;width:25%}

.knob{position:absolute;left:50%;top:50%;width:74px;height:74px;
  margin:-37px 0 0 -37px;border-radius:50%;background:#e8eaed;
  box-shadow:0 5px 16px rgba(0,0,0,.65);transition:transform .13s ease-out}
.knob.live{transition:none}

#trig{width:280px;height:58px;border-radius:14px;background:#171b1f;
  border:1px solid #262b31;color:#9aa2aa;font:600 12px/58px ui-monospace,monospace;
  letter-spacing:.2em;text-align:center;touch-action:none}
#trig.on{background:#e8eaed;color:#0e1013;border-color:#e8eaed}

.latch{display:flex;align-items:center;gap:7px;font-size:10px;
  letter-spacing:.12em;text-transform:uppercase;color:#666d75}
.latch input{width:15px;height:15px;accent-color:#e8eaed}

#leds{display:flex;gap:8px;margin-top:2px}
.led{display:flex;flex-direction:column;align-items:center;gap:5px;
  font-size:9px;letter-spacing:.1em;color:#5a6067}
.led i{width:26px;height:5px;border-radius:3px;background:#262b31;
  display:block;transition:background .06s}
.led.on i{background:#39d353;box-shadow:0 0 8px #39d353}
</style></head><body>

<header><span id="dot"></span><span id="status">connected</span></header>

<div>
  <div class="lbl">throttle</div>
  <div class="track" id="thrTrack"><div class="knob" id="thrKnob"></div></div>
</div>

<div>
  <div class="lbl">steering</div>
  <div class="track" id="strTrack"><div class="knob" id="strKnob"></div></div>
</div>

<div id="trig">TRIGGER</div>
<label class="latch"><input type="checkbox" id="latch"> latch</label>

<div id="leds">
  <div class="led" id="l0"><i></i>FWD</div>
  <div class="led" id="l1"><i></i>REV</div>
  <div class="led" id="l2"><i></i>LEFT</div>
  <div class="led" id="l3"><i></i>RGHT</div>
  <div class="led" id="l4"><i></i>TRIG</div>
</div>

<script>
var busy=false, trig=false;

// Generic single-axis stick. Springs back to centre on release.
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
    var d = vertical ? e.clientY-(r.top+r.height/2)
                     : e.clientX-(r.left+r.width/2);
    if(d> T) d= T;
    if(d<-T) d=-T;
    knob.style.transform = vertical ? 'translateY('+d+'px)' : 'translateX('+d+'px)';
    // up and right are positive
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
function paintTrig(){ trigEl.classList.toggle('on',trig); }

trigEl.addEventListener('pointerdown',function(e){
  trigEl.setPointerCapture(e.pointerId);
  trig = latchEl.checked ? !trig : true;
  paintTrig(); send();
});
trigEl.addEventListener('pointerup',function(){
  if(!latchEl.checked){ trig=false; paintTrig(); send(); }
});
trigEl.addEventListener('pointercancel',function(){
  if(!latchEl.checked){ trig=false; paintTrig(); send(); }
});

// Indicators show what the ESP32 actually asserted, echoed back
// from the board — not what the browser thinks it sent.
function paintLeds(s){
  for(var i=0;i<5;i++){
    document.getElementById('l'+i).classList.toggle('on', s.charAt(i)==='1');
  }
}

function send(){
  if(busy) return;
  busy=true;
  fetch('/c?t='+thrAxis.get()+'&s='+strAxis.get()+'&b='+(trig?1:0))
    .then(function(r){ return r.text(); })
    .then(function(txt){
      document.getElementById('dot').className='';
      document.getElementById('status').textContent='connected';
      paintLeds(txt);
    })
    .catch(function(){
      document.getElementById('dot').className='bad';
      document.getElementById('status').textContent='signal lost';
      paintLeds('00000');
    })
    .then(function(){ busy=false; });
}

// ~14 sends/sec, doubling as the heartbeat
setInterval(send,70);
</script></body></html>)HTML";

// --------------------------- handlers ---------------------------

void handleRoot() { server.send_P(200, "text/html", PAGE); }

void handleCmd() {
  thr  = constrain(server.hasArg("t") ? server.arg("t").toInt() : 0, -100, 100);
  str  = constrain(server.hasArg("s") ? server.arg("s").toInt() : 0, -100, 100);
  trig = server.hasArg("b") && server.arg("b").toInt() != 0;
  lastCmd = millis();

  applyOutputs();

  // Echo the real pin states back so the UI shows the board's
  // view, not the browser's guess.
  char s[6];
  for (int i = 0; i < 5; i++) s[i] = outState[i] ? '1' : '0';
  s[5] = '\0';
  server.send(200, "text/plain", s);
}

// ----------------------------- setup -----------------------------

void setup() {
  Serial.begin(115200);

  int pins[] = {PIN_FWD, PIN_REV, PIN_LEFT, PIN_RGHT, PIN_TRIG};
  for (int p : pins) pinMode(p, OUTPUT);
  releaseAll();

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("Join \"");
  Serial.print(AP_SSID);
  Serial.print("\" then open http://");
  Serial.println(WiFi.softAPIP());   // 192.168.4.1

  server.on("/", handleRoot);
  server.on("/c", handleCmd);
  server.onNotFound(handleRoot);
  server.begin();

  lastCmd = millis();
}

void loop() {
  server.handleClient();

  if (millis() - lastCmd > FAILSAFE_MS) releaseAll();

  applyOutputs();   // must run continuously for soft PWM
}

/* =============================================================
   BRING-UP

   Do this before wiring anything to the remote. Put an LED and a
   1k resistor from each output pin to GND, flash, and work the
   UI. The five indicators on the page should track the LEDs
   exactly. That proves the whole chain without risking the
   remote.

   Then wire the optos and test with the car up on a block:

   1. Push throttle up -> car goes forward.
      Goes backwards -> swap PIN_FWD and PIN_REV.
   2. Push steering right -> car turns right.
      Turns the wrong way -> swap PIN_LEFT and PIN_RGHT.
   3. Nothing at all, but the page indicator lights -> the ESP32
      side is fine. Press the remote's own button while watching;
      if that works, swap the opto's pins 3 and 4.
   4. Control fires too easily or too late -> adjust DEADZONE.
      The shaded band on each track shows it on screen.
   5. Want a throttle instead of full-speed-only -> set
      SOFT_PWM = true. If the car stutters or the remote's LED
      flickers, raise SOFT_PERIOD_MS to 120 or 160.

   -------------------------------------------------------------
   ADDING MORE CONTROLS

   If the remote turns out to have more buttons, each one needs:
     - a pin constant up top
     - a slot in outState[] and the arrays in applyOutputs/setup
     - a query arg in handleCmd
     - a button and an indicator in the page

   Safe spare output pins: 4, 16, 17, 21, 22. Avoid 0, 2, 12 and
   15 (strapping pins, they affect boot) and 34-39 (input only).
   ============================================================= */
