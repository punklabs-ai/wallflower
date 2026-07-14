'use strict';

/* ------------------------------------------------------------------ geometry */
const hm = document.getElementById('hm');
const hx = hm.getContext('2d');
const bloom = document.getElementById('bloom');
const bx = bloom.getContext('2d');
const trace = document.getElementById('trace');
const tx = trace.getContext('2d');
const HW = hm.width, HH = hm.height;
const BW = bloom.width, BH = bloom.height;
const TW = trace.width, TH = trace.height;
const AX = 56, BOT = 22, TOP = 12, RIGHT = 12;

const COLS = 430;
const COL_DT = 0.05;

let history = [];      // {col, p, l, c}
let traceRows = [];    // raw payloads
let freq = [];
let maxTrace = 0.05;
let userTouched = false;

/* ---------------------------------------------------- viridis colormap (data) */
const VIRIDIS = [
  [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141], [42, 120, 142],
  [33, 145, 140], [34, 168, 132], [68, 190, 112], [122, 209, 81],
  [189, 223, 38], [253, 231, 37],
];
function lerpRamp(ramp, t) {
  t = Math.max(0, Math.min(1, t));
  const p = t * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(p));
  const f = p - i, a = ramp[i], b = ramp[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function heatColor(t) {
  const c = lerpRamp(VIRIDIS, t);
  return `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
}
// build the colorbar gradient to match the map exactly
(function () {
  const stops = [];
  for (let i = 0; i <= 10; i++) {
    const c = lerpRamp(VIRIDIS, i / 10);
    stops.push(`rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0}) ${i * 10}%`);
  }
  document.getElementById('cbGrad').style.background = `linear-gradient(90deg, ${stops.join(',')})`;
})();

/* ------------------------------------------------------------- shared cursor */
const cursor = { active: false, x: 0 };       // x in canvas-space (hm/trace share AX..W-RIGHT)
function bindCursor(canvas) {
  canvas.addEventListener('mousemove', (e) => {
    const r = canvas.getBoundingClientRect();
    cursor.active = true;
    cursor.x = (e.clientX - r.left) * (canvas.width / r.width);
  });
  canvas.addEventListener('mouseleave', () => { cursor.active = false; });
}
bindCursor(hm); bindCursor(trace);

function cursorIndex(plotW) {
  // map canvas-space x -> index into the right-aligned rolling buffer
  const colsFromRight = (AX + plotW - cursor.x) / (plotW / COLS);
  return { colsFromRight, idx: traceRows.length - 1 - Math.round(colsFromRight) };
}

function drawCursorLine(ctx, h) {
  if (!cursor.active) return;
  if (cursor.x < AX || cursor.x > (ctx === hx ? HW : TW) - RIGHT) return;
  ctx.strokeStyle = 'rgba(246,247,249,.35)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(cursor.x, TOP);
  ctx.lineTo(cursor.x, h - BOT);
  ctx.stroke();
  ctx.setLineDash([]);
}

function updateReadout() {
  const ro = document.getElementById('cursorReadout');
  const plotW = TW - AX - RIGHT;
  if (!cursor.active || cursor.x < AX || cursor.x > TW - RIGHT || !traceRows.length) {
    ro.textContent = 'Hover the charts to inspect any instant.';
    return;
  }
  const { colsFromRight, idx } = cursorIndex(plotW);
  if (idx < 0 || idx >= traceRows.length) { ro.textContent = ''; return; }
  const r = traceRows[idx];
  const tAgo = Math.max(0, colsFromRight * COL_DT);
  let peakHz = '—';
  const h = history[idx + (history.length - traceRows.length)];
  if (h && h.col && freq.length) {
    let bi = 0; for (let i = 1; i < h.col.length; i++) if (h.col[i] > h.col[bi]) bi = i;
    peakHz = (freq[bi] != null ? freq[bi].toFixed(1) : '—') + ' Hz';
  }
  ro.innerHTML =
    `t <b>-${tAgo.toFixed(2)} s</b>` +
    ` · motion <b>${num(r.motion)}</b>` +
    ` · threshold <b>${num(r.thr)}</b>` +
    ` · floor <b>${num(r.floor)}</b>` +
    ` · peak band <b>${peakHz}</b>` +
    (r.presence ? ' · <b style="color:#ff9b9e">MOTION</b>' : '');
}

/* --------------------------------------------------------------------- axes */
function drawAxes(ctx, w, h, title, ylabels) {
  ctx.strokeStyle = '#404854';
  ctx.lineWidth = 1;
  ctx.font = '10px ' + MONO;
  ctx.beginPath();
  ctx.moveTo(AX, TOP); ctx.lineTo(AX, h - BOT); ctx.lineTo(w - RIGHT, h - BOT);
  ctx.stroke();
  ctx.fillStyle = '#8f99a8';
  ctx.fillText(title.toUpperCase(), AX, 9);

  for (const y of ylabels) {
    ctx.fillStyle = '#8f99a8';
    ctx.fillText(y.label, 6, y.py + 3);
    ctx.strokeStyle = 'rgba(64,72,84,.45)';
    ctx.beginPath(); ctx.moveTo(AX, y.py); ctx.lineTo(w - RIGHT, y.py); ctx.stroke();
  }

  const span = COLS * COL_DT;
  for (let i = 0; i <= 4; i++) {
    const x = AX + (w - AX - RIGHT) * i / 4;
    ctx.strokeStyle = 'rgba(64,72,84,.30)';
    ctx.beginPath(); ctx.moveTo(x, TOP); ctx.lineTo(x, h - BOT); ctx.stroke();
    ctx.fillStyle = '#8f99a8';
    ctx.fillText(`-${Math.round((4 - i) * span / 4)}s`, x - 10, h - 6);
  }
}
const MONO = '"SFMono-Regular",Consolas,Menlo,monospace';

/* -------------------------------------------------------------- spectrogram */
function drawHeat(col, d) {
  if (col) history.push({ col, p: d.presence, l: d.label, c: d.calibrating });
  while (history.length > COLS) history.shift();

  hx.fillStyle = '#14181d';
  hx.fillRect(0, 0, HW, HH);
  const plotW = HW - AX - RIGHT, plotH = HH - TOP - BOT;
  const colW = plotW / COLS;
  const n = freq.length || 33;

  for (let x = 0; x < history.length; x++) {
    const item = history[x];
    const px = AX + plotW - (history.length - x) * colW;
    for (let i = 0; i < item.col.length; i++) {
      hx.fillStyle = heatColor(item.col[i]);
      const y = TOP + plotH - (i + 1) * plotH / n;
      hx.fillRect(px, y, Math.ceil(colW) + 1, Math.ceil(plotH / n) + 1);
    }
    if (item.p) {
      hx.fillStyle = 'rgba(205,66,70,.30)';
      hx.fillRect(px, TOP, colW + 1, plotH);
    } else if (item.c) {
      hx.fillStyle = 'rgba(200,118,25,.16)';
      hx.fillRect(px, TOP, colW + 1, plotH);
    }
  }

  const maxF = freq.length ? freq[freq.length - 1] : 10;
  drawAxes(hx, HW, HH, 'frequency (Hz)', [0, .3, 1, 3, maxF].map(v => ({
    label: v.toFixed(v < 1 ? 1 : 0),
    py: TOP + plotH - (v / maxF) * plotH,
  })));
  drawCursorLine(hx, HH);
}

/* --------------------------------------------------------------- motion trace */
function drawTrace(d) {
  // stash each router's real motion at this instant (updateFeeds ran in drawBloom)
  d.mR2 = FEEDS[1].present ? FEEDS[1].motion : 0;
  d.mR3 = FEEDS[2].present ? FEEDS[2].motion : 0;
  traceRows.push(d);
  while (traceRows.length > COLS) traceRows.shift();

  tx.fillStyle = '#14181d';
  tx.fillRect(0, 0, TW, TH);
  const plotW = TW - AX - RIGHT, plotH = TH - TOP - BOT;
  // Scale the y-axis to the tallest point currently VISIBLE (any series) plus 15%
  // headroom, so peaks never touch the top edge / get flat-topped. Slow release
  // keeps it from snapping down jarringly after a spike scrolls off.
  let peak = 0.05;
  for (const r of traceRows) {
    peak = Math.max(peak, r.motion || 0, r.thr || 0,
      (FEEDS[1].enabled && FEEDS[1].present) ? r.mR2 || 0 : 0,
      (FEEDS[2].enabled && FEEDS[2].present) ? r.mR3 || 0 : 0);
  }
  maxTrace = Math.max(maxTrace * .95, peak * 1.15);
  drawAxes(tx, TW, TH, 'motion energy', [0, maxTrace / 2, maxTrace].map(v => ({
    label: v.toFixed(2),
    py: TOP + plotH - (v / maxTrace) * plotH,
  })));

  function line(key, color, dash, width) {
    tx.strokeStyle = color; tx.lineWidth = width || 1.6; tx.setLineDash(dash || []);
    tx.beginPath();
    traceRows.forEach((r, i) => {
      const x = AX + plotW - (traceRows.length - i) * plotW / COLS;
      const y = TOP + plotH - (Math.min(r[key], maxTrace) / maxTrace) * plotH;
      i === 0 ? tx.moveTo(x, y) : tx.lineTo(x, y);
    });
    tx.stroke(); tx.setLineDash([]);
  }
  line('floor', '#8f99a8', [2, 4], 1);
  line('thr', '#cd4246', [5, 4], 1.4);
  if (FEEDS[1].enabled && FEEDS[1].present) line('mR2', ROUTER_COLOR[1], [4, 3], 1.2);
  if (FEEDS[2].enabled && FEEDS[2].present) line('mR3', ROUTER_COLOR[2], [4, 3], 1.2);
  line('motion', ROUTER_COLOR[0], [], 1.8);
  drawCursorLine(tx, TH);
}

/* ----------------------------------------------------------------- bloom view
   Static frame: motion energy blooms from the centre and FADES over time, mapping
   the scalar motion level (the faithful signal for BFI-live, where "motion" is a
   sustained energy level, not a frequency pattern) to a central glow — instant
   attack, slow release — plus expanding ripples on each detection onset.

   FEEDS: each feed is a real router reported by the backend (its own capture,
   its own AP). With >1 router enabled the view becomes a mini-map: the receiver
   sits at centre, each router is placed at its estimated distance, and the world
   auto-zooms to fit 1 → 2 → 3 routers.

   DISTANCE is a rough, UNCALIBRATED estimate from RSSI via a log-distance
   path-loss model (lower RSSI ⇒ farther). Bearing is NOT measurable here, so the
   angular slots are arbitrary. When a feed has no dBm RSSI (fused/BFI modes), the
   distance is genuinely unknowable and is shown as "?" with a dashed link. */
const RSSI_REF_DBM = -40;   // assumed RSSI at 1 m (uncalibrated)
const PATHLOSS_N = 2.5;     // indoor-ish path-loss exponent (uncalibrated)
function estDistanceM(rssi) {
  if (rssi == null || Number.isNaN(rssi)) return null;
  return Math.pow(10, (RSSI_REF_DBM - rssi) / (10 * PATHLOSS_N));
}

// Each feed mirrors a REAL router the backend reports in d.routers — its own
// capture, its own AP. No feed is derived from another; a router slot with no
// backing source is marked absent (present=false), never fabricated.
const ROUTER_COLOR = ['#4c90f0', '#32a467', '#f0b56a'];
const FEEDS = [
  { id: 0, name: 'Router 1', enabled: true, locked: true },
  { id: 1, name: 'Router 2', enabled: true, locked: false },
  { id: 2, name: 'Router 3', enabled: true, locked: false },
];
FEEDS.forEach(f => { f.level = 0; f.max = 0.05; f.ripples = []; f.wasPresent = false; f.motion = 0; f.presence = false; f.rssi = null; f.dist = null; f.present = false; });

function updateFeeds(d) {
  const routers = Array.isArray(d.routers) ? d.routers : [];
  for (const f of FEEDS) {
    const r = routers.find(x => x.id === f.id);
    f.present = !!(r && r.enabled);
    f.motion = f.present ? (r.motion || 0) : 0;                 // REAL per-router motion
    f.rssi = (f.present && typeof r.rssi === 'number') ? r.rssi : null;
    f.dist = estDistanceM(f.rssi);
    f.presence = f.present ? !!r.presence : false;             // REAL per-router detection

    f.max = Math.max(f.max * 0.999, f.motion, d.thr || 0, 0.05);
    const target = Math.max(0, Math.min(1, f.motion / f.max));
    f.level = Math.max(target, f.level * 0.94);                // instant attack, slow release

    if (f.presence && !f.wasPresent && !d.calibrating) f.ripples.push({ t: 0, a: 0.6 });
    f.wasPresent = f.presence;
    f.ripples = f.ripples.filter(rp => rp.a > 0.02 && rp.t < 1.1);
    f.ripples.forEach(rp => { rp.t += 0.012; rp.a *= 0.96; });
  }
}

// Paint one feed's glow + ripples + core at (cx,cy). Returns nothing.
function paintFeedGlow(cx, cy, f, calib, glowMax, ringMax) {
  const base = lerpRamp(VIRIDIS, calib ? 0.22 : f.level);
  let cr = base[0], cg = base[1], cb = base[2];
  if (f.presence) { cr += (205 - cr) * 0.6; cg += (66 - cg) * 0.6; cb += (70 - cb) * 0.6; }
  const rgb = `${cr | 0},${cg | 0},${cb | 0}`;

  const glowR = Math.max(2, glowMax * (0.18 + 0.82 * f.level));
  const a = calib ? 0.22 : (0.15 + 0.85 * f.level);
  const g = bx.createRadialGradient(cx, cy, 0, cx, cy, glowR);
  g.addColorStop(0, `rgba(${rgb},${a})`);
  g.addColorStop(0.5, `rgba(${rgb},${a * 0.5})`);
  g.addColorStop(1, `rgba(${rgb},0)`);
  bx.fillStyle = g; bx.beginPath(); bx.arc(cx, cy, glowR, 0, Math.PI * 2); bx.fill();

  for (const rp of f.ripples) {
    bx.strokeStyle = `rgba(205,66,70,${rp.a})`; bx.lineWidth = 1.5;
    bx.beginPath(); bx.arc(cx, cy, rp.t * ringMax, 0, Math.PI * 2); bx.stroke();
  }

  bx.fillStyle = `rgba(${rgb},${calib ? 0.4 : 0.85})`;
  bx.beginPath(); bx.arc(cx, cy, 3 + 5 * f.level, 0, Math.PI * 2); bx.fill();
}

// Single centred bloom (1 feed enabled) — the calm "movement in the middle".
function drawSingleBloom(f, d) {
  const cx = BW / 2, cy = BH / 2, maxR = Math.min(BW, BH) * 0.46;
  bx.strokeStyle = 'rgba(64,72,84,.35)'; bx.lineWidth = 1;
  for (const r of [0.5, 1.0]) { bx.beginPath(); bx.arc(cx, cy, maxR * r, 0, Math.PI * 2); bx.stroke(); }
  paintFeedGlow(cx, cy, f, !!d.calibrating, maxR, maxR * 1.05);

  bx.font = '11px ' + MONO; bx.fillStyle = '#8f99a8'; bx.textAlign = 'left';
  bx.fillText(d.calibrating ? 'CALIBRATING' : f.presence ? 'MOTION' : 'idle', 10, BH - 10);
  bx.textAlign = 'right';
  bx.fillText('energy ' + num(f.motion), BW - 10, BH - 10);
  bx.textAlign = 'left';
}

// Fixed equilateral triangle slots by router id: 0 top, 1 bottom-left, 2 bottom-right.
function routerVertex(id, cx, cy, R) {
  const ang = [-Math.PI / 2, Math.PI * 5 / 6, Math.PI / 6][id] || 0;
  return [cx + R * Math.cos(ang), cy + R * Math.sin(ang)];
}

// Single bloom for the *estimated movement location* (centroid), not per-router.
const moveBlob = { level: 0, max: 0.05, ripples: [], wasPresent: false };

// Multi-router localisation (≥2 routers): pin the routers in a fixed triangle and
// show where the movement is, as the ENERGY-WEIGHTED CENTROID of the routers'
// real motion (the "diff/combine the routers" step). The angular slots are fixed,
// not measured, so a true position fix needs ≥3 spatially separated radios.
function drawLocalization(on, d) {
  const cx = BW / 2, cy = BH / 2, R = Math.min(BW, BH) * 0.34;
  on.forEach(f => { const [x, y] = routerVertex(f.id, cx, cy, R); f._x = x; f._y = y; });

  // triangle / segment edges between the pinned routers
  if (on.length >= 2) {
    bx.strokeStyle = 'rgba(64,72,84,.55)'; bx.lineWidth = 1;
    bx.beginPath();
    on.forEach((f, i) => (i ? bx.lineTo(f._x, f._y) : bx.moveTo(f._x, f._y)));
    if (on.length >= 3) bx.closePath();
    bx.stroke();
  }

  // energy-weighted centroid = estimated movement location
  let wsum = 0, px = 0, py = 0;
  for (const f of on) { const w = Math.max(0, f.motion); wsum += w; px += w * f._x; py += w * f._y; }
  if (wsum < 1e-6) {                                   // no motion → geometric centre
    px = on.reduce((s, f) => s + f._x, 0) / on.length;
    py = on.reduce((s, f) => s + f._y, 0) / on.length;
  } else { px /= wsum; py /= wsum; }

  // movement-bloom intensity from total motion energy (instant attack, slow release)
  const totalE = on.reduce((s, f) => s + f.motion, 0);
  moveBlob.max = Math.max(moveBlob.max * 0.999, totalE, (d.thr || 0) * on.length, 0.05);
  const target = Math.max(0, Math.min(1, totalE / moveBlob.max));
  moveBlob.level = Math.max(target, moveBlob.level * 0.94);
  const present = !d.calibrating && on.some(f => f.presence);
  if (present && !moveBlob.wasPresent) moveBlob.ripples.push({ t: 0, a: 0.6 });
  moveBlob.wasPresent = present;
  moveBlob.ripples = moveBlob.ripples.filter(rp => rp.a > 0.02 && rp.t < 1.1);
  moveBlob.ripples.forEach(rp => { rp.t += 0.012; rp.a *= 0.96; });

  // the movement bloom (reuse the glow painter with the centroid blob)
  paintFeedGlow(px, py, { level: moveBlob.level, presence: present, ripples: moveBlob.ripples },
                !!d.calibrating, R * 0.7, R * 0.85);

  // static router markers (NOT blooming) + labels
  for (const f of on) {
    bx.fillStyle = ROUTER_COLOR[f.id] || '#8f99a8';
    bx.beginPath(); bx.arc(f._x, f._y, 5, 0, Math.PI * 2); bx.fill();
    bx.strokeStyle = 'rgba(246,247,249,.30)'; bx.lineWidth = 1;
    bx.beginPath(); bx.arc(f._x, f._y, 8, 0, Math.PI * 2); bx.stroke();
  }
  bx.font = '11px ' + MONO; bx.textAlign = 'center';
  for (const f of on) {
    bx.fillStyle = '#c8ccd2'; bx.fillText(f.name, f._x, f._y - 14);
    bx.fillStyle = '#8f99a8'; bx.fillText('E ' + num(f.motion), f._x, f._y + 22);
  }

  // caption
  bx.textAlign = 'left'; bx.fillStyle = '#5b636f'; bx.font = '10px ' + MONO;
  bx.fillText('movement = energy-weighted centroid of routers · a true fix needs '
    + '≥3 spatially separated radios', 10, BH - 9);
}

function drawBloom(d) {
  updateFeeds(d);
  bx.fillStyle = '#14181d'; bx.fillRect(0, 0, BW, BH);
  const on = FEEDS.filter(f => f.enabled && f.present);
  if (on.length <= 1) drawSingleBloom(on[0] || FEEDS[0], d);
  else drawLocalization(on, d);
  updateFeedBoxes(d);
}

/* router selector boxes — toggle which routers feed the localisation view */
const feedbar = document.getElementById('feedbar');
(function buildFeeds() {
  for (const f of FEEDS) {
    const el = document.createElement('div');
    el.className = 'feedbox' + (f.locked ? ' locked' : '') + (f.enabled ? ' on' : ' off');
    el.id = `feed_${f.id}`;
    el.innerHTML =
      `<div class="feedbox-head">` +
      `<span class="feed-swatch" style="background:${ROUTER_COLOR[f.id]}"></span>` +
      `<span class="feed-name">${f.name}</span>` +
      `<span class="feed-dot" id="feedDot_${f.id}"></span></div>` +
      `<div class="feedbox-meta"><span class="feed-energy" id="feedE_${f.id}">energy –</span>` +
      `<span class="feed-dist" id="feedD_${f.id}">dist –</span></div>`;
    feedbar.appendChild(el);
    if (!f.locked) el.onclick = () => {
      f.enabled = !f.enabled;
      el.classList.toggle('on', f.enabled);
      el.classList.toggle('off', !f.enabled);
    };
  }
  const hint = document.createElement('div');
  hint.className = 'feedhint';
  hint.textContent = 'Each router is an independent real capture (its own AP). Toggle which routers feed the localisation view.';
  feedbar.appendChild(hint);
  feedbar.hidden = false;                            // feeds affect the trace too — always visible
})();
function updateFeedBoxes(d) {
  for (const f of FEEDS) {
    document.getElementById(`feed_${f.id}`).classList.toggle('absent', !f.present);
    const eEl = document.getElementById(`feedE_${f.id}`);
    const dEl = document.getElementById(`feedD_${f.id}`);
    const dot = document.getElementById(`feedDot_${f.id}`);
    if (!f.present) {                                // no backing capture source
      eEl.textContent = 'no source';
      dEl.textContent = ''; dEl.classList.add('unknown');
      dot.className = 'feed-dot';
      continue;
    }
    eEl.textContent = 'energy ' + num(f.motion);
    if (f.dist != null) { dEl.textContent = '≈' + f.dist.toFixed(1) + ' m'; dEl.classList.remove('unknown'); }
    else { dEl.textContent = 'dist ?'; dEl.classList.add('unknown'); }
    dot.className = 'feed-dot' + (d.calibrating ? ' calib' : f.presence ? ' motion' : '');
  }
}

/* view toggle: swap the main view between spectrogram and bloom */
let viewMode = 'spectrogram';
const colorbarEl = document.getElementById('colorbar');
const mainViewTitle = document.getElementById('mainViewTitle');
const viewBtns = Array.from(document.querySelectorAll('#viewToggle .seg-btn'));
function setView(mode) {
  viewMode = mode;
  const spec = mode === 'spectrogram';
  hm.hidden = !spec;
  bloom.hidden = spec;
  colorbarEl.classList.toggle('hidden', !spec);
  mainViewTitle.innerHTML = spec
    ? 'Spectrogram <span class="muted" style="font-weight:400">· frequency × time, newest at right</span>'
    : 'Motion Bloom <span class="muted" style="font-weight:400">· energy radiates from centre, fades over time</span>';
  viewBtns.forEach(b => b.classList.toggle('active', b.dataset.view === mode));
}
viewBtns.forEach(b => { b.onclick = () => setView(b.dataset.view); });
setView('spectrogram');

/* ------------------------------------------------------------------ metrics */
const METRICS = [
  { id: 'sig', label: 'Signal' },
  { id: 'mot', label: 'Motion' },
  { id: 'floor', label: 'Noise floor' },
  { id: 'thr', label: 'Threshold' },
  { id: 'mad', label: 'Resting MAD' },
  { id: 'fps', label: 'Frame rate', unit: '/s' },
  { id: 'csi', label: 'CSI comp' },
  { id: 'bfi', label: 'BFI comp' },
];
const sparks = {};   // id -> {buf, ctx}
(function buildMetrics() {
  const wrap = document.getElementById('metrics');
  for (const m of METRICS) {
    const t = document.createElement('div');
    t.className = 'tile';
    t.innerHTML =
      `<div class="tile-val"><b id="m_${m.id}">–</b><span class="tile-unit" id="u_${m.id}">${m.unit || ''}</span></div>` +
      `<div class="tile-label">${m.label}</div>` +
      `<canvas class="spark" id="s_${m.id}" width="160" height="22"></canvas>`;
    wrap.appendChild(t);
    sparks[m.id] = { buf: [], ctx: document.getElementById(`s_${m.id}`).getContext('2d') };
  }
})();
function num(v) {
  if (v == null || v === '' || Number.isNaN(Number(v))) return '–';
  const n = Number(v);
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(1);
  if (Math.abs(n) >= 1) return n.toFixed(2);
  return n.toFixed(3);
}
function setMetric(id, v, unit) {
  document.getElementById(`m_${id}`).textContent = num(v);
  if (unit != null) document.getElementById(`u_${id}`).textContent = unit;
  const s = sparks[id];
  if (v == null || Number.isNaN(Number(v))) return;
  s.buf.push(Number(v));
  while (s.buf.length > 80) s.buf.shift();
  drawSpark(s);
}
function drawSpark(s) {
  const c = s.ctx, w = c.canvas.width, h = c.canvas.height;
  c.clearRect(0, 0, w, h);
  if (s.buf.length < 2) return;
  let lo = Math.min(...s.buf), hi = Math.max(...s.buf);
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  c.strokeStyle = '#4c90f0'; c.lineWidth = 1.25; c.beginPath();
  s.buf.forEach((v, i) => {
    const x = i / (s.buf.length - 1) * (w - 2) + 1;
    const y = h - 2 - (v - lo) / (hi - lo) * (h - 4);
    i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
  });
  c.stroke();
}

/* ---------------------------------------------------------- parameter form */
const PARAMS = [
  { id: 'k', key: 'k', label: 'Sensitivity multiplier', unit: '×', min: 0.5, max: 12, step: 0.1, def: 3.5, dec: 1 },
  { id: 'floorOffset', key: 'floor_offset', label: 'Noise floor offset', unit: '', min: -0.08, max: 0.30, step: 0.005, def: 0, dec: 3 },
  { id: 'smooth', key: 'smooth_n', label: 'Jitter smoothing', unit: 'tap', min: 1, max: 12, step: 1, def: 3, dec: 0 },
  { id: 'motionWin', key: 'motion_win_s', label: 'Motion window', unit: 's', min: 0.25, max: 2.5, step: 0.05, def: 0.75, dec: 2 },
  { id: 'debOn', key: 'deb_on', label: 'Trigger debounce', unit: 'smp', min: 1, max: 20, step: 1, def: 4, dec: 0 },
  { id: 'debOff', key: 'deb_off', label: 'Clear debounce', unit: 'smp', min: 1, max: 40, step: 1, def: 10, dec: 0 },
  { id: 'pMargin', key: 'p_margin', label: 'Resting p99 guard', unit: '×', min: 1, max: 2, step: 0.01, def: 1.12, dec: 2 },
];
const P = {};
(function buildParams() {
  const form = document.getElementById('paramForm');
  for (const p of PARAMS) {
    const el = document.createElement('div');
    el.className = 'param';
    el.innerHTML =
      `<div class="param-top"><label for="${p.id}">${p.label}</label>` +
      `<span class="param-range">${p.min}–${p.max}</span></div>` +
      `<div class="param-ctl">` +
      `<input type="range" id="${p.id}" min="${p.min}" max="${p.max}" step="${p.step}">` +
      `<input type="number" id="${p.id}Num" min="${p.min}" max="${p.max}" step="${p.step}">` +
      `<span class="unit">${p.unit}</span></div>`;
    form.appendChild(el);
    const range = document.getElementById(p.id);
    const numIn = document.getElementById(p.id + 'Num');
    P[p.key] = { p, range, numIn };
    const push = (val) => {
      userTouched = true;
      const v = Number(val);
      range.value = v; numIn.value = v.toFixed(p.dec);
      fetch(`/set?${p.key}=${encodeURIComponent(v)}`);
      flashApplied();
    };
    range.oninput = () => push(range.value);
    numIn.onchange = () => push(numIn.value);
  }
})();
let appliedTimer = null;
function flashApplied() {
  const f = document.getElementById('appliedFlag');
  f.classList.add('show');
  clearTimeout(appliedTimer);
  appliedTimer = setTimeout(() => f.classList.remove('show'), 900);
}
function applySettings(s) {
  if (userTouched) return;     // don't fight the operator while they're tuning
  for (const key in P) {
    if (s[key] == null) continue;
    const { p, range, numIn } = P[key];
    range.value = s[key];
    numIn.value = Number(s[key]).toFixed(p.dec);
  }
}
document.getElementById('resetBtn').onclick = () => {
  for (const p of PARAMS) {
    P[p.key].range.value = p.def;
    P[p.key].numIn.value = p.def.toFixed(p.dec);
    fetch(`/set?${p.key}=${encodeURIComponent(p.def)}`);
  }
  userTouched = true;
  flashApplied();
};

/* --------------------------------------------------------------- calibration */
let lastCalib = null, wasCalibrating = false;
function utc(d) { return d.toISOString().slice(11, 19) + ' UTC'; }
document.getElementById('recalibrateBtn').onclick = () => {
  fetch('/calibrate');
  document.getElementById('calibState').textContent = 'Recalibrating…';
};

/* --------------------------------------------------------- ground truth */
function updateLabel(s) {
  const tag = document.getElementById('lblTag');
  tag.textContent = s;
  tag.style.color = s === 'still' ? 'var(--green-bright)' : s === 'moving' ? '#f0b56a' : 'var(--dim)';
  document.getElementById('stillBtn').classList.toggle('active', s === 'still');
  document.getElementById('movingBtn').classList.toggle('active', s === 'moving');
  curLabel = s;
}
let curLabel = 'unlabeled';
function setlbl(s) { fetch(`/label?state=${encodeURIComponent(s)}`); updateLabel(s); }
document.getElementById('stillBtn').onclick = () => setlbl('still');
document.getElementById('movingBtn').onclick = () => setlbl('moving');
document.getElementById('clearLabelBtn').onclick = () => setlbl('unlabeled');
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (k === 's') setlbl('still');
  else if (k === 'm') setlbl('moving');
  else if (k === 'c') setlbl('unlabeled');
});

/* ----------------------------------------------------------------- logging */
let logging = false;
function renderLog() {
  const b = document.getElementById('logBtn');
  b.textContent = logging ? 'Stop logging' : 'Enable logging';
  b.classList.toggle('active', logging);
}
document.getElementById('logBtn').onclick = () => {
  logging = !logging; fetch(`/log?on=${logging ? 1 : 0}`); renderLog();
};

/* ------------------------------------------------------------ detection log */
const events = [];      // newest first
let cur = null;         // open detection
function confidenceOf(ratio) {
  if (ratio >= 2) return 'high';
  if (ratio >= 1.3) return 'med';
  return 'low';
}
function onPresence(d, now) {
  if (d.calibrating) return;
  if (d.presence && !cur) {
    cur = { start: now, thr: d.thr, peak: d.motion, label: d.label };
  } else if (cur) {
    cur.peak = Math.max(cur.peak, d.motion);
    cur.thr = Math.max(cur.thr, d.thr);   // record the bar it had to clear
    if (d.label !== 'unlabeled') cur.label = d.label;
    if (!d.presence) { finalizeEvent(now); }
  }
}
function finalizeEvent(now) {
  cur.end = now;
  cur.dur = (now - cur.start) / 1000;
  cur.ratio = cur.thr > 0 ? cur.peak / cur.thr : 0;
  cur.conf = confidenceOf(cur.ratio);
  events.unshift(cur);
  while (events.length > 50) events.pop();
  cur = null;
  renderEvents();
}
function renderEvents() {
  const body = document.getElementById('evtBody');
  document.getElementById('detCount').textContent =
    `${events.length} event${events.length === 1 ? '' : 's'}`;
  if (!events.length) {
    body.innerHTML = '<tr class="empty"><td colspan="6">No detections recorded this session.</td></tr>';
    return;
  }
  body.innerHTML = events.map((e, i) =>
    `<tr data-i="${i}">` +
    `<td>${utc(e.start)}</td>` +
    `<td>${e.dur.toFixed(2)} s</td>` +
    `<td>${num(e.peak)}</td>` +
    `<td>${num(e.thr)}</td>` +
    `<td><span class="conf ${e.conf}">${e.conf} ·${e.ratio.toFixed(1)}×</span></td>` +
    `<td class="lblcell ${e.label}">${e.label}</td>` +
    `</tr>`).join('');
  body.querySelectorAll('tr').forEach(tr => {
    tr.onclick = () => selectEvent(Number(tr.dataset.i), tr);
  });
}
function selectEvent(i, tr) {
  document.querySelectorAll('#evtBody tr').forEach(r => r.classList.remove('selected'));
  tr.classList.add('selected');
  const e = events[i];
  document.getElementById('detailBody').innerHTML =
    row('Start', utc(e.start)) +
    row('End', utc(e.end)) +
    row('Duration', e.dur.toFixed(2) + ' s') +
    row('Peak motion', num(e.peak)) +
    row('Threshold', num(e.thr)) +
    row('Peak / threshold', e.ratio.toFixed(2) + '×') +
    row('Confidence', e.conf) +
    row('Ground-truth label', e.label);
  document.getElementById('detailCard').hidden = false;
}
function row(k, v) { return `<dt>${k}</dt><dd>${v}</dd>`; }
document.getElementById('detailClose').onclick = () => {
  document.getElementById('detailCard').hidden = true;
  document.querySelectorAll('#evtBody tr').forEach(r => r.classList.remove('selected'));
};

/* ------------------------------------------------------------- provenance */
function setProvenance(d) {
  document.getElementById('pvSource').textContent = d.source || '—';
  const m = d.meta || {};
  if (m.ap_bssid) document.getElementById('pvBssid').textContent = m.ap_bssid;
  if (m.mon_iface) document.getElementById('pvIface').textContent = m.mon_iface;
  if (m.fs) document.getElementById('pvFs').textContent = Number(m.fs).toFixed(0) + ' Hz';
  if (m.log_path) document.getElementById('logPath').textContent = m.log_path;
}

/* ------------------------------------------------------------------- clock */
setInterval(() => {
  document.getElementById('clock').textContent = utc(new Date());
}, 1000);

/* --------------------------------------------------------------- SSE stream */
const es = new EventSource('/stream');
es.onopen = () => document.getElementById('conn').classList.remove('off');
es.onerror = () => document.getElementById('conn').classList.add('off');
es.onmessage = function onMessage(e) {
  const d = JSON.parse(e.data);
  const now = new Date();
  freq = d.freq_hz || freq;

  drawHeat(d.col, d);   // keep spectrogram history current even while hidden
  drawBloom(d);         // keep the bloom fading even while hidden
  drawTrace(d);
  updateReadout();
  applySettings(d.settings || {});
  setProvenance(d);

  const isReal = (d.source || '').startsWith('real');
  setMetric('sig', d.rssi == null ? d.signal : d.rssi, isReal ? 'dBm' : '');
  setMetric('mot', d.motion);
  setMetric('floor', d.floor);
  setMetric('thr', d.thr);
  setMetric('mad', d.mad);
  setMetric('fps', d.fps);
  setMetric('csi', d.components && d.components.csi);
  setMetric('bfi', d.components && d.components.bfi);

  updateLabel(d.label);
  if (typeof d.logging === 'boolean' && d.logging !== logging) { logging = d.logging; renderLog(); }

  // status tag
  const tag = document.getElementById('statusTag');
  if (d.calibrating) { tag.dataset.state = 'calibrating'; tag.textContent = 'CALIBRATING'; }
  else if (d.presence) { tag.dataset.state = 'motion'; tag.textContent = '● MOTION'; }
  else { tag.dataset.state = 'idle'; tag.textContent = 'IDLE'; }

  // calibration state transition (true -> false means a fresh baseline locked)
  if (wasCalibrating && !d.calibrating) {
    lastCalib = now;
    document.getElementById('calibState').textContent = 'Last calibrated ' + utc(now);
  }
  wasCalibrating = d.calibrating;

  onPresence(d, now);
};

renderLog();
