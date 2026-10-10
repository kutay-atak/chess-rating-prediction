// Board sounds synthesised with WebAudio: a wooden tap for moves, a sharper double tap for
// captures, a bell-like blip for check and two tones for game start and end.
let ctx = null;
let enabled = true;
try { enabled = localStorage.getItem("sound") !== "off"; } catch {}

export function soundOn() { return enabled; }

export function toggleSound() {
  enabled = !enabled;
  try { localStorage.setItem("sound", enabled ? "on" : "off"); } catch {}
  return enabled;
}

function audio() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

function tap(t, freq, gain, length = 0.07) {
  const a = audio();
  const buffer = a.createBuffer(1, Math.floor(a.sampleRate * length), a.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (a.sampleRate * 0.008));
  const src = a.createBufferSource();
  src.buffer = buffer;
  const filter = a.createBiquadFilter();
  filter.type = "bandpass";
  filter.frequency.value = freq;
  filter.Q.value = 1.2;
  const g = a.createGain();
  g.gain.value = gain;
  src.connect(filter).connect(g).connect(a.destination);
  src.start(a.currentTime + t);
}

function tone(t, freq, length, gain = 0.12) {
  const a = audio();
  const osc = a.createOscillator();
  const g = a.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  g.gain.setValueAtTime(0, a.currentTime + t);
  g.gain.linearRampToValueAtTime(gain, a.currentTime + t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, a.currentTime + t + length);
  osc.connect(g).connect(a.destination);
  osc.start(a.currentTime + t);
  osc.stop(a.currentTime + t + length + 0.02);
}

export function play(kind) {
  if (!enabled) return;
  try {
    if (kind === "move") tap(0, 900, 2.2);
    else if (kind === "capture") { tap(0, 1300, 2.6); tap(0.035, 700, 1.6); }
    else if (kind === "check") { tap(0, 900, 2); tone(0.02, 1320, 0.18, 0.08); }
    else if (kind === "start") { tone(0, 660, 0.18); tone(0.12, 990, 0.25); }
    else if (kind === "end") { tone(0, 880, 0.2); tone(0.15, 660, 0.35); }
    else if (kind === "low") tone(0, 1500, 0.08, 0.06);
  } catch {}
}
