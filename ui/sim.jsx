// sim.jsx — Simulated real-time data for WaveScape
// Generates threats, audio beam energy, IMU heading, system stats, and event stream.

const SIM_THREAT_TYPES = [
  { type:"vehicle",  icon:"🚗", label:"VEHICLE",   audio:"engine, tire-noise",       visual:"sedan, moving" },
  { type:"person",   icon:"👤", label:"PERSON",    audio:"footsteps, voice",         visual:"adult, walking" },
  { type:"cyclist",  icon:"🚲", label:"CYCLIST",   audio:"chain-noise, freewheel",   visual:"cyclist, lateral" },
  { type:"obstacle", icon:"⚠",  label:"OBSTACLE",  audio:"static",                   visual:"stationary object" },
  { type:"siren",    icon:"🚨", label:"SIREN",     audio:"emergency, doppler+",      visual:"—" },
  { type:"dog",      icon:"🐕", label:"DOG",       audio:"bark, panting",            visual:"canine, low" },
];

const SIM_SOURCES = ["YOLO","GEMMA","AUDIO","FUSION"];
const SIM_PATTERNS = {
  critical: "◦◦◦◦",
  high:     "● ● ●",
  medium:   "■ ■",
  low:      "▮"
};

function urgencyFromDist(d){
  if(d<2) return "critical";
  if(d<4) return "high";
  if(d<8) return "medium";
  return "low";
}

function rand(min,max){ return Math.random()*(max-min)+min; }
function rint(min,max){ return Math.floor(rand(min,max+1)); }
function pick(arr){ return arr[Math.floor(Math.random()*arr.length)]; }

let __tid = 0;
function newThreat(forcedUrgency){
  __tid += 1;
  const t = pick(SIM_THREAT_TYPES);
  let dist;
  if(forcedUrgency==="critical") dist = rand(1.0,1.9);
  else if(forcedUrgency==="high") dist = rand(2.1,3.9);
  else if(forcedUrgency==="medium") dist = rand(4.1,7.9);
  else dist = rand(2.0,14);
  const angle = rand(0,360);
  const u = urgencyFromDist(dist);
  return {
    id: "T-"+String(__tid).padStart(4,"0"),
    type: t.type, icon: t.icon, label: t.label,
    audioClass: t.audio, visualClass: t.visual,
    distance: dist,
    angle, // degrees, 0=front, clockwise
    confidence: rint(62,98),
    velocity: rand(-1.8, 3.4), // m/s (positive = closing)
    urgency: u,
    source: pick(SIM_SOURCES),
    bornAt: Date.now(),
    ttl: 4000 + rand(0,5000),
    eta: dist / Math.max(0.5, rand(0.6,2.2)),
  };
}

function summarize(threat){
  const d = threat.distance.toFixed(1);
  const dir = ["FRONT","NE","RIGHT","SE","BACK","SW","LEFT","NW"][Math.round(threat.angle/45)%8];
  const closing = threat.velocity>0 ? "closing" : "receding";
  return `${threat.label.toLowerCase()} detected ${d}m to the ${dir.toLowerCase()}, ${closing} at ${Math.abs(threat.velocity).toFixed(1)} m/s. confidence ${threat.confidence}%. fusion sources: ${threat.source}.`;
}

function hhmmss(t){
  const d = new Date(t);
  const p = n => String(n).padStart(2,"0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function compass(angle){
  // 0 = front, clockwise
  const idx = Math.round(((angle%360)+360)%360 / 45) % 8;
  return ["FRONT","NE","RIGHT","SE","BACK","SW","LEFT","NW"][idx];
}

function arrowFor(angle){
  const idx = Math.round(((angle%360)+360)%360 / 45) % 8;
  return ["↑","↗","→","↘","↓","↙","←","↖"][idx];
}

function patternFor(urg){ return SIM_PATTERNS[urg] || "▮"; }

// expose
Object.assign(window, {
  SIM_THREAT_TYPES, SIM_SOURCES, SIM_PATTERNS,
  newThreat, urgencyFromDist, summarize, hhmmss, compass, arrowFor, patternFor,
  rand, rint, pick
});
