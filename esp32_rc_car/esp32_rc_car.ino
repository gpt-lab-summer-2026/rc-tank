/* =============================================================
   ESP32 RC Car  —  phone-controlled over WiFi, no app needed
   -------------------------------------------------------------
   The ESP32 creates its own WiFi network. Connect your phone to
   it, open http://192.168.4.1 and drive with the on-screen stick.

   HARDWARE (2 DC motors, tank/differential steering)
     ESP32 dev board  +  L298N (or TB6612 / DRV8833) H-bridge

     ESP32 GPIO 14 -> ENA   (left  speed / PWM)
     ESP32 GPIO 27 -> IN1   (left  direction)
     ESP32 GPIO 26 -> IN2   (left  direction)
     ESP32 GPIO 25 -> ENB   (right speed / PWM)
     ESP32 GPIO 33 -> IN3   (right direction)
     ESP32 GPIO 32 -> IN4   (right direction)
     ESP32 GND     -> L298N GND        <-- REQUIRED common ground

   POWER
     Motors run off the battery pack through the driver's +12V in.
     Never drive motors from the ESP32's 5V/3V3 pin.
     If the pack is 7-12V you can feed the L298N's 5V output into
     the ESP32 VIN. Above 12V, remove the L298N jumper and use a
     separate regulator. Add a 470uF+ cap across the motor supply;
     motor inrush is the usual cause of random ESP32 reboots.

   IDE SETUP
     Boards Manager -> "esp32" by Espressif. Works on core 2.x and
     3.x (the PWM API changed between them; handled below).
   ============================================================= */

#include <WiFi.h>
#include <WebServer.h>

// ---------------- Things you may want to change ----------------
const char* AP_SSID = "ESP32-RC-Car";
const char* AP_PASS = "drive1234";       // min 8 chars, or "" for open

const int PIN_L_PWM = 14, PIN_L_IN1 = 27, PIN_L_IN2 = 26;
const int PIN_R_PWM = 25, PIN_R_IN1 = 33, PIN_R_IN2 = 32;

const bool INVERT_LEFT  = false;   // flip to true if a wheel spins backwards
const bool INVERT_RIGHT = false;

const int MIN_DUTY   = 60;    // 0-255. Cheap motors stall below ~25% duty,
                              // so the low end of the stick is mapped up to
                              // here. Lower it if the car creeps too fast.
const int FAILSAFE_MS = 400;  // stop if no command arrives for this long
// ---------------------------------------------------------------

const int PWM_FREQ = 20000;   // 20 kHz — above hearing, no motor whine
const int PWM_RES  = 8;       // 8-bit: duty 0..255
const int CH_L = 0, CH_R = 1; // LEDC channels (only used on core 2.x)

// The LEDC API changed in ESP32 Arduino core 3.0. These macros make the
// sketch compile on both, so you don't get "ledcSetup was not declared".
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  #define PWM_SETUP(pin, ch)      ledcAttach(pin, PWM_FREQ, PWM_RES)
  #define PWM_WRITE(pin, ch, val) ledcWrite(pin, val)
#else
  #define PWM_SETUP(pin, ch)      do { ledcSetup(ch, PWM_FREQ, PWM_RES); \
                                       ledcAttachPin(pin, ch); } while (0)
  #define PWM_WRITE(pin, ch, val) ledcWrite(ch, val)
#endif

WebServer server(80);
unsigned long lastCmd = 0;
bool stopped = true;

// ------------------------- motor control -------------------------

// speed: -100 (full reverse) .. +100 (full forward)
void setMotor(int pwmPin, int ch, int in1, int in2, int speed, bool invert) {
  if (invert) speed = -speed;

  int mag = abs(speed);
  if (mag > 100) mag = 100;

  int duty = 0;
  if (mag > 0) duty = MIN_DUTY + (255 - MIN_DUTY) * mag / 100;

  if (duty == 0) {                 // coast
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
  } else if (speed > 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  }
  PWM_WRITE(pwmPin, ch, duty);
}

// x = steering (-100 left .. +100 right), y = throttle (-100 .. +100)
void drive(int x, int y) {
  int l = y + x;
  int r = y - x;

  // Scale both down instead of clipping, so turns keep their shape
  int peak = max(abs(l), abs(r));
  if (peak > 100) { l = l * 100 / peak; r = r * 100 / peak; }

  setMotor(PIN_L_PWM, CH_L, PIN_L_IN1, PIN_L_IN2, l, INVERT_LEFT);
  setMotor(PIN_R_PWM, CH_R, PIN_R_IN1, PIN_R_IN2, r, INVERT_RIGHT);
  stopped = (l == 0 && r == 0);
}

void stopAll() { drive(0, 0); }

// ------------------------- control page -------------------------

const char PAGE[] PROGMEM = R"HTML(<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>RC Car</title><style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-user-select:none;user-select:none}
html,body{height:100%;overflow:hidden;background:#101214;color:#e8eaed;
  font:500 15px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
body{display:flex;flex-direction:column;align-items:center;
  justify-content:space-between;padding:18px 16px 28px;touch-action:none}
header{display:flex;align-items:center;gap:10px;letter-spacing:.14em;
  font-size:12px;text-transform:uppercase;color:#7d848c}
#dot{width:9px;height:9px;border-radius:50%;background:#39d353;
  box-shadow:0 0 10px #39d353;transition:background .2s,box-shadow .2s}
#dot.bad{background:#f0503a;box-shadow:0 0 10px #f0503a}
#base{position:relative;width:270px;height:270px;border-radius:50%;
  background:radial-gradient(circle at 50% 45%,#1c2025,#141719);
  border:1px solid #262b31;touch-action:none;flex:none}
#base::before,#base::after{content:"";position:absolute;background:#22272d}
#base::before{left:50%;top:14%;bottom:14%;width:1px;transform:translateX(-50%)}
#base::after{top:50%;left:14%;right:14%;height:1px;transform:translateY(-50%)}
#knob{position:absolute;left:50%;top:50%;width:96px;height:96px;
  margin:-48px 0 0 -48px;border-radius:50%;background:#e8eaed;
  box-shadow:0 6px 20px rgba(0,0,0,.6);transition:transform .12s ease-out}
#knob.live{transition:none}
.row{width:270px;display:flex;justify-content:space-between;
  font-size:12px;color:#7d848c;letter-spacing:.08em}
input[type=range]{width:270px;height:34px;background:none;-webkit-appearance:none}
input[type=range]::-webkit-slider-runnable-track{height:3px;background:#2b3138}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:28px;
  height:28px;margin-top:-13px;border-radius:50%;background:#e8eaed}
</style></head><body>

<header><span id="dot"></span><span id="status">connected</span></header>

<div id="base"><div id="knob"></div></div>

<div style="width:270px">
  <div class="row"><span>power</span><span id="pct">70%</span></div>
  <input id="pow" type="range" min="20" max="100" value="70">
</div>

<script>
var x=0,y=0,R=87,busy=false,drag=false;
var base=document.getElementById('base'),knob=document.getElementById('knob'),
    dot=document.getElementById('dot'),status=document.getElementById('status'),
    pow=document.getElementById('pow'),pct=document.getElementById('pct');

pow.oninput=function(){pct.textContent=pow.value+'%'};

function place(e){
  var b=base.getBoundingClientRect();
  var dx=e.clientX-(b.left+b.width/2), dy=e.clientY-(b.top+b.height/2);
  var d=Math.hypot(dx,dy);
  if(d>R){dx=dx*R/d; dy=dy*R/d;}
  knob.style.transform='translate('+dx+'px,'+dy+'px)';
  x=dx/R; y=-dy/R;
}
function release(){
  drag=false; x=0; y=0;
  knob.classList.remove('live');
  knob.style.transform='translate(0,0)';
  send();
}
base.addEventListener('pointerdown',function(e){
  drag=true; knob.classList.add('live');
  base.setPointerCapture(e.pointerId); place(e);
});
base.addEventListener('pointermove',function(e){ if(drag) place(e); });
base.addEventListener('pointerup',release);
base.addEventListener('pointercancel',release);
window.addEventListener('blur',release);

// Sends ~14x/sec. This doubles as the heartbeat: if the car stops
// hearing from us it cuts the motors on its own.
function send(){
  if(busy) return;
  busy=true;
  var s=pow.value/100;
  fetch('/c?x='+Math.round(x*100*s)+'&y='+Math.round(y*100*s))
    .then(function(){ dot.className=''; status.textContent='connected'; })
    .catch(function(){ dot.className='bad'; status.textContent='signal lost'; })
    .then(function(){ busy=false; });
}
setInterval(send,70);
</script></body></html>)HTML";

// --------------------------- handlers ---------------------------

void handleRoot() { server.send_P(200, "text/html", PAGE); }

void handleCmd() {
  int x = server.hasArg("x") ? server.arg("x").toInt() : 0;
  int y = server.hasArg("y") ? server.arg("y").toInt() : 0;
  drive(constrain(x, -100, 100), constrain(y, -100, 100));
  lastCmd = millis();
  server.send(200, "text/plain", "ok");
}

// ----------------------------- setup -----------------------------

void setup() {
  Serial.begin(115200);

  int dirPins[] = {PIN_L_IN1, PIN_L_IN2, PIN_R_IN1, PIN_R_IN2};
  for (int p : dirPins) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }

  PWM_SETUP(PIN_L_PWM, CH_L);
  PWM_SETUP(PIN_R_PWM, CH_R);
  stopAll();

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

  // Failsafe — without this the car keeps its last command forever
  // when the phone locks, the tab closes, or you walk out of range.
  if (!stopped && millis() - lastCmd > FAILSAFE_MS) stopAll();
}

/* =============================================================
   VARIANT: one drive motor + steering servo (true RC car layout)

   Install the "ESP32Servo" library, then:

     #include <ESP32Servo.h>
     Servo steer;
     const int PIN_SERVO = 13;
     const int CENTER = 90, THROW = 40;   // trim CENTER if it pulls

     // in setup():
     steer.setPeriodHertz(50);
     steer.attach(PIN_SERVO, 500, 2400);

     // replace drive() with:
     void drive(int x, int y) {
       steer.write(CENTER + THROW * x / 100);
       setMotor(PIN_L_PWM, CH_L, PIN_L_IN1, PIN_L_IN2, y, INVERT_LEFT);
       stopped = (y == 0);
     }

   Everything else — the page, the failsafe, the mixing limits — is
   unchanged. Center the servo before bolting on the steering link.
   ============================================================= */
