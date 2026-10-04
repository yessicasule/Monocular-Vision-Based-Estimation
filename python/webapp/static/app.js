/* MonoArm dashboard front-end.
 *
 *   1. captures real webcam frames and pushes them to the server
 *   2. renders the returned per-frame result: skeleton overlay, joint angles,
 *      time-series traces with a hover readout, and the derivation behind
 *      every number
 *   3. drives the control plane (filter, calibration, logging, Unity link)
 *   4. surfaces the plots this project's analysis scripts generate
 *
 * No external libraries: charts and the overlay are drawn with the 2D canvas
 * API, so the page works with no network access beyond this server.
 *
 * Chart colours are read from CSS custom properties rather than hard-coded, so
 * the light and dark palettes — each separately validated for contrast and
 * colour-vision separation — follow the theme automatically.
 */
"use strict";

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

const CHANNELS = [
  { key: "shoulder_flexion",   label: "Shoulder flexion",   short: "flex",  range: [-60, 180] },
  { key: "shoulder_abduction", label: "Shoulder abduction", short: "abd",   range: [-45, 180] },
  { key: "shoulder_rotation",  label: "Shoulder rotation",  short: "rot",   range: [-90,  90] },
  { key: "elbow_flexion",      label: "Elbow flexion",      short: "elbow", range: [  0, 150] },
];

const FILTER_NAMES = { kalman: "Kalman", ma: "Moving average", sg: "Savitzky–Golay" };
const FILTER_SLOT  = { kalman: 1, ma: 2, sg: 3 };   // fixed categorical order

const EDGES = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [12, 14], [14, 16], [11, 13], [13, 15],
];
const ARM_POINTS = new Set([11, 12, 13, 14, 15, 16]);

const TRACE_SECONDS = 10;
const HISTORY_MAX   = 900;

const POSE_LABELS = {
  arm_down:    "Arm hanging straight down at your side",
  arm_forward: "Arm straight out in front, horizontal",
  arm_side:    "Arm straight out to the side, horizontal",
  elbow_bent:  "Upper arm down, elbow bent to about 90°",
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) =>
  (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const signed = (v, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;

async function api(path, method = "GET", body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) throw new Error((data && data.detail) || res.statusText);
  return data;
}

/** Current palette, read from CSS so it tracks the active theme. */
function palette() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return {
    series: [v("--series-1"), v("--series-2"), v("--series-3")],
    raw:    v("--series-raw"),
    accent: v("--accent"),
    grid:   v("--line"),
    axis:   v("--line-strong"),
    text:   v("--text-2"),
    muted:  v("--text-muted"),
    surface: v("--surface"),
  };
}

const filterColor = (ft, pal) => pal.series[(FILTER_SLOT[ft] || 1) - 1];

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

const S = {
  ws: null, wsReady: false,
  stream: null, capturing: false,
  sending: false, sendInterval: 1000 / 30,
  lastSentAt: 0, sendClaimedAt: 0, pendingSentAt: 0, lastRtt: undefined,
  frame: null, metrics: null, explain: null,
  history: { right: [], left: [] },
  chartSide: "right", traceSide: "right", allFilters: false,
  sourceMode: "browser",
  charts: [], hover: null,
  calibration: null,
  figures: null,
};

// ---------------------------------------------------------------------------
// theme
// ---------------------------------------------------------------------------

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") || "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("monoarm-theme", theme); } catch (e) { /* private mode */ }
  const dark = theme === "dark";
  $("iconSun").style.display  = dark ? "none" : "";
  $("iconMoon").style.display = dark ? "" : "none";
  $("btnTheme").setAttribute("aria-label",
    dark ? "Switch to light mode" : "Switch to dark mode");
  // Charts and any server-rendered plot must be repainted in the new palette.
  renderLegend(true);
  renderCharts();
  refreshSessionPlots();
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => { S.wsReady = true; setPill($("pConn"), "connected", "ok", true); };
  ws.onclose = () => {
    S.wsReady = false; S.sending = false;
    setPill($("pConn"), "reconnecting…", "warn", true);
    setTimeout(connect, 1200);
  };
  ws.onerror = () => setPill($("pConn"), "socket error", "bad", true);
  ws.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "frame")       onFrame(msg);
    else if (msg.type === "hello")  onStatus(msg.status);
    else if (msg.type === "status") onStatus(msg.status);
    else if (msg.type === "error")  { S.sending = false; console.warn("pipeline:", msg.message); }
  };
  S.ws = ws;
}

function onFrame(msg) {
  // A browser-sourced frame completes a round trip measurable on one clock:
  // the page stamped the send time, so this is a true end-to-end figure.
  if (msg.source === "browser" && S.pendingSentAt) {
    S.lastRtt = performance.now() - S.pendingSentAt;
    S.pendingSentAt = 0;
    S.sending = false;
    if (S.ws && S.wsReady && Math.random() < 0.25) {
      S.ws.send(JSON.stringify({ type: "rtt", ms: S.lastRtt }));
    }
  }
  S.frame = msg;
  if (msg.metrics) S.metrics = msg.metrics;
  pushHistory(msg);
  render(msg);
}

function pushHistory(msg) {
  for (const side of ["right", "left"]) {
    const banks = {};
    for (const ft of Object.keys(msg.filters || {})) {
      banks[ft] = msg.filters[ft] ? msg.filters[ft][side] : null;
    }
    const h = S.history[side];
    h.push({
      t: msg.t,
      raw:  msg.raw ? msg.raw[side] : null,
      filt: msg.filtered ? msg.filtered[side] : null,
      banks,
    });
    if (h.length > HISTORY_MAX) h.shift();
  }
}

// ---------------------------------------------------------------------------
// camera capture
// ---------------------------------------------------------------------------

async function startCamera() {
  const [w, h] = $("selRes").value.split("x").map(Number);
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: 30 } },
      audio: false,
    });
  } catch (err) {
    showPlaceholder("Camera unavailable",
      `${err.message}. Browsers grant camera access only over https:// or on localhost.`);
    return;
  }
  const video = $("video");
  video.srcObject = S.stream;
  await video.play();

  $("placeholder").style.display = "none";
  $("btnStart").disabled = true;
  $("btnStop").disabled = false;
  S.capturing = true;
  S.sourceMode = "browser";
  requestAnimationFrame(captureLoop);
}

function stopCamera() {
  S.capturing = false;
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  S.stream = null;
  $("btnStart").disabled = false;
  $("btnStop").disabled = true;
  showPlaceholder("Camera stopped", "Press Start camera to resume.");
}

const grabCanvas = document.createElement("canvas");
const grabCtx = grabCanvas.getContext("2d", { willReadFrequently: true });

function captureLoop() {
  if (!S.capturing) return;
  const video = $("video");
  const now = performance.now();

  // Recover the in-flight slot if a frame never came back.
  if (S.sending && now - S.sendClaimedAt > 3000) S.sending = false;

  // One frame in flight at a time: the server processes serially, so sending
  // faster would queue latency rather than raise the achieved rate. The slot is
  // claimed synchronously, before the asynchronous JPEG encode begins.
  if (S.wsReady && !S.sending && now - S.lastSentAt >= S.sendInterval &&
      video.videoWidth > 0) {
    S.lastSentAt = now;
    S.sending = true;
    S.sendClaimedAt = now;
    grabCanvas.width = video.videoWidth;
    grabCanvas.height = video.videoHeight;
    grabCtx.drawImage(video, 0, 0);
    grabCanvas.toBlob((blob) => {
      if (!blob || !S.wsReady) { S.sending = false; return; }
      blob.arrayBuffer().then((buf) => {
        if (!S.wsReady) { S.sending = false; return; }
        S.pendingSentAt = performance.now();
        S.ws.send(buf);
      }).catch(() => { S.sending = false; });
    }, "image/jpeg", 0.72);
  }

  drawOverlay();
  requestAnimationFrame(captureLoop);
}

function showPlaceholder(title, hint) {
  const p = $("placeholder");
  p.style.display = "flex";
  p.innerHTML = `<strong>${esc(title)}</strong><div class="hint">${esc(hint || "")}</div>`;
}

// ---------------------------------------------------------------------------
// video overlay
// ---------------------------------------------------------------------------

function drawOverlay() {
  const canvas = $("overlay");
  const video = $("video");
  const src = S.sourceMode === "browser" ? video : null;
  const fr = S.frame;

  const W = src && src.videoWidth ? src.videoWidth : (fr ? fr.width : 640);
  const H = src && src.videoHeight ? src.videoHeight : (fr ? fr.height : 480);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);

  if (src && src.videoWidth) {
    ctx.save();
    if ($("chkMirror").checked) { ctx.translate(W, 0); ctx.scale(-1, 1); }
    ctx.drawImage(src, 0, 0, W, H);
    ctx.restore();
  }

  if (!fr || !fr.landmarks || !fr.landmarks.length) return;

  const pal = palette();
  const pts = {};
  for (const lm of fr.landmarks) pts[lm.i] = lm;

  const scale = W / 640;
  ctx.lineWidth = Math.max(2, 2.4 * scale);
  ctx.lineCap = "round";

  for (const [a, b] of EDGES) {
    const p = pts[a], q = pts[b];
    if (!p || !q) continue;
    const isArm = ARM_POINTS.has(a) && ARM_POINTS.has(b) && !(a === 11 && b === 12);
    const conf = Math.min(p.v, q.v);
    ctx.globalAlpha = isArm ? clamp(conf, 0.3, 1) : clamp(conf * 0.6, 0.15, 0.6);
    ctx.strokeStyle = isArm ? pal.accent : "#8fa3b5";
    ctx.beginPath();
    ctx.moveTo(p.x * W, p.y * H);
    ctx.lineTo(q.x * W, q.y * H);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  for (const lm of fr.landmarks) {
    if (lm.i === 0) continue;
    const r = (ARM_POINTS.has(lm.i) ? 4.6 : 3.4) * scale;
    ctx.beginPath();
    ctx.arc(lm.x * W, lm.y * H, r, 0, Math.PI * 2);
    if (lm.v >= 0.5) {
      // A 2px surface ring keeps overlapping joints readable.
      ctx.fillStyle = ARM_POINTS.has(lm.i) ? pal.accent : "#8fa3b5";
      ctx.fill();
      ctx.lineWidth = 2 * scale;
      ctx.strokeStyle = "#011627";
      ctx.stroke();
    } else {
      // Low-confidence keypoints are hollow: the overlay shows tracking
      // certainty rather than implying every point is equally sure.
      ctx.lineWidth = 2 * scale;
      ctx.strokeStyle = "#e0b341";
      ctx.stroke();
    }
  }

  // Elbow angle arc, drawn at the joint with a direct label
  for (const [side, idx] of [["right", [12, 14, 16]], ["left", [11, 13, 15]]]) {
    const a = pts[idx[0]], b = pts[idx[1]], c = pts[idx[2]];
    const ang = fr.calibrated && fr.calibrated[side];
    if (!a || !b || !c || !ang) continue;
    const bx = b.x * W, by = b.y * H;
    const a1 = Math.atan2(a.y * H - by, a.x * W - bx);
    const a2 = Math.atan2(c.y * H - by, c.x * W - bx);
    const rad = 20 * scale;
    ctx.beginPath();
    ctx.strokeStyle = pal.series[1];
    ctx.lineWidth = 2.2 * scale;
    ctx.arc(bx, by, rad, a1, a2, ((a2 - a1 + Math.PI * 2) % (Math.PI * 2)) > Math.PI);
    ctx.stroke();

    const label = `${ang.elbow_flexion.toFixed(0)}°`;
    ctx.font = `600 ${Math.round(13 * scale)}px ui-monospace, monospace`;
    const tw = ctx.measureText(label).width;
    const lx = bx + rad + 6 * scale, ly = by + 4 * scale;
    ctx.fillStyle = "rgba(1,22,39,.72)";
    ctx.beginPath();
    ctx.roundRect(lx - 4, ly - 12 * scale, tw + 8, 17 * scale, 4);
    ctx.fill();
    ctx.fillStyle = "#fdfffc";
    ctx.fillText(label, lx, ly);
  }
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

function setPill(node, text, kind, withDot) {
  node.className = "pill" + (kind ? " " + kind : "");
  node.innerHTML = (withDot ? '<i class="dot"></i>' : "") + text;
}

function render(fr) {
  renderPills(fr);
  renderKpis(fr);
  renderArms(fr);
  renderLegend();
  renderCharts();
  renderTrace(fr);
  renderLatency(fr);
  renderFilterTable();
  renderWire(fr);
  syncControls(fr);
}

function renderPills(fr) {
  const tracked = ["right", "left"].filter((s) => fr.calibrated && fr.calibrated[s]);
  setPill($("pTrack"),
    fr.detected ? (tracked.length ? `tracking ${tracked.join(" + ")}` : "pose, no arms")
                : "no pose",
    fr.detected && tracked.length ? "ok" : "warn");

  const udp = fr.status && fr.status.udp;
  if (udp) {
    setPill($("pUdp"),
      udp.enabled ? `Unity <b>${esc(udp.host)}:${udp.port}</b>` : "Unity off",
      udp.enabled ? (udp.send_errors ? "bad" : "ok") : "");
    $("sPkts").textContent = udp.packets_sent.toLocaleString();
    $("sPkts").className = udp.send_errors ? "bad" : (udp.enabled ? "ok" : "");
  }
}

function renderKpis(fr) {
  const m = S.metrics;
  if (!m) return;

  $("kFps").innerHTML = `${fmt(m.fps, 1)}<small> fps</small>`;
  $("kFpsFoot").textContent = m.fps >= 20 ? "meets the ≥ 20 fps target"
                                          : "below the 20 fps target";
  $("sFps").textContent = `${fmt(m.fps, 1)} fps`;
  $("sFps").className = m.fps >= 20 ? "ok" : "";

  // Two different measurements, never conflated: the browser round trip is
  // glass-to-glass on the page's own clock, while within_budget describes the
  // server pipeline alone. Showing one and captioning it with the other would
  // read as a contradiction whenever browser-side encoding is the bottleneck.
  const serverP95 = m.latency.total_ms.p95;
  const lat = S.lastRtt !== undefined ? S.lastRtt : serverP95;
  $("kLat").innerHTML = `${fmt(lat, 0)}<small> ms</small>`;
  $("kLatFoot").textContent = S.lastRtt !== undefined
    ? `browser round trip · server pipeline ${fmt(serverP95, 0)} ms p95`
    : `server pipeline p95 · ${(m.within_budget * 100).toFixed(0)}% within 100 ms`;
  $("sLat").textContent = `${fmt(lat, 0)} ms`;
  $("sLat").className = lat <= 100 ? "ok" : "bad";

  const active = fr.status ? fr.status.filter_type : "kalman";
  const s = m.angle_std[`${active}:${S.chartSide}`];
  const elbow = s ? s.elbow_flexion : null;
  $("kStab").innerHTML = elbow === null ? "—<small> °σ</small>"
                                        : `${fmt(elbow, 2)}<small> °σ</small>`;
  $("kStabFoot").textContent = elbow === null ? "waiting for tracked frames"
    : (elbow <= 3 ? "inside the ±3° band" : elbow <= 5 ? "inside the ±5° band"
                                                       : "above the ±5° band");

  $("kDet").innerHTML = `${(m.detection_rate * 100).toFixed(0)}<small> %</small>`;
  $("kDetFoot").textContent = `${m.frames.toLocaleString()} frames processed`;
  $("sUp").textContent = m.uptime_s >= 60
    ? `${Math.floor(m.uptime_s / 60)}m ${Math.round(m.uptime_s % 60)}s`
    : `${fmt(m.uptime_s, 0)}s`;
  $("winN").textContent = m.latency.total_ms.n || 0;
}

function armCard(side) {
  const card = document.createElement("div");
  card.className = "arm";
  card.innerHTML =
    `<h4>${side} arm <span class="tag" id="tag-${side}">—</span></h4>` +
    CHANNELS.map((ch) => `
      <div class="angle" id="a-${side}-${ch.key}">
        <div class="angle-top">
          <span class="name">${ch.label}</span>
          <span class="raw" id="raw-${side}-${ch.key}">raw —</span>
        </div>
        <div class="angle-val none" id="val-${side}-${ch.key}">—<span class="u">°</span></div>
        <div class="meter"><div class="zero" id="z-${side}-${ch.key}"></div>
          <div class="fill" id="f-${side}-${ch.key}"></div></div>
      </div>`).join("");
  return card;
}

function renderArms(fr) {
  const host = $("arms");
  if (!host.childElementCount) {
    host.appendChild(armCard("right"));
    host.appendChild(armCard("left"));
  }
  for (const side of ["right", "left"]) {
    const out = fr.calibrated ? fr.calibrated[side] : null;
    const raw = fr.raw ? fr.raw[side] : null;
    const tag = $(`tag-${side}`);
    tag.className = "tag " + (out ? "live" : "lost");
    tag.textContent = out ? "tracked" : "not tracked";

    for (const ch of CHANNELS) {
      const val = $(`val-${side}-${ch.key}`);
      const rawEl = $(`raw-${side}-${ch.key}`);
      const fill = $(`f-${side}-${ch.key}`);
      const zero = $(`z-${side}-${ch.key}`);
      const row = $(`a-${side}-${ch.key}`);

      if (!out) {
        val.className = "angle-val none";
        val.innerHTML = `—<span class="u">°</span>`;
        rawEl.textContent = "raw —";
        fill.style.width = "0%";
        row.classList.remove("unreliable");
        continue;
      }
      const v = out[ch.key];
      val.className = "angle-val";
      val.innerHTML = `${signed(v)}<span class="u">°</span>`;
      rawEl.textContent = raw ? `raw ${raw[ch.key].toFixed(1)}°` : "raw —";
      row.classList.toggle("unreliable",
        ch.key === "shoulder_rotation" && !out.rotation_reliable);

      const [lo, hi] = ch.range;
      const zp = clamp((0 - lo) / (hi - lo), 0, 1) * 100;
      const vp = clamp((v - lo) / (hi - lo), 0, 1) * 100;
      zero.style.left = `${zp}%`;
      fill.style.left = `${Math.min(zp, vp)}%`;
      fill.style.width = `${Math.abs(vp - zp)}%`;
    }
  }
}

// ---------------------------------------------------------------------------
// charts
// ---------------------------------------------------------------------------

function buildCharts() {
  const host = $("charts");
  host.innerHTML = "";
  S.charts = CHANNELS.map((ch) => {
    const wrap = document.createElement("div");
    wrap.className = "chart";
    const canvas = document.createElement("canvas");
    wrap.appendChild(canvas);
    wrap.insertAdjacentHTML("beforeend",
      `<div class="c-title">${ch.label}</div>
       <div class="c-range"></div><div class="c-now"></div>`);
    host.appendChild(wrap);

    const entry = { ch, wrap, canvas,
                    range: wrap.querySelector(".c-range"),
                    now: wrap.querySelector(".c-now") };

    canvas.addEventListener("pointermove", (e) => {
      const r = canvas.getBoundingClientRect();
      S.hover = { x: e.clientX - r.left, chart: entry, clientX: e.clientX, clientY: e.clientY };
      renderCharts();
    });
    canvas.addEventListener("pointerleave", () => { S.hover = null; hideTooltip(); renderCharts(); });
    return entry;
  });

  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.id = "chartTip";
  host.appendChild(tip);
}

function hideTooltip() { const t = $("chartTip"); if (t) t.classList.remove("on"); }

let _legendKey = "";
function renderLegend(force) {
  const active = S.frame && S.frame.status ? S.frame.status.filter_type : "kalman";
  const key = `${S.allFilters}|${active}|${currentTheme()}`;
  if (!force && key === _legendKey) return;
  _legendKey = key;

  const pal = palette();
  const parts = [`<span><i style="background:${pal.raw}"></i>raw estimate</span>`];
  if (S.allFilters) {
    for (const ft of Object.keys(FILTER_NAMES)) {
      parts.push(`<span><i style="background:${filterColor(ft, pal)}"></i>${FILTER_NAMES[ft]}${
        ft === active ? " · to Unity" : ""}</span>`);
    }
  } else {
    parts.push(`<span><i style="background:${filterColor(active, pal)}"></i>${
      FILTER_NAMES[active] || active} → Unity</span>`);
  }
  $("legend").innerHTML = parts.join("");
}

function renderCharts() {
  const hist = S.history[S.chartSide];
  if (!hist.length || !S.charts.length) return;
  const pal = palette();
  const now = hist[hist.length - 1].t;
  // Before 10 s of history exists, start the axis at the first sample so the
  // trace fills the plot instead of hugging the right edge. Once the window is
  // full this is exactly now - TRACE_SECONDS and the trace scrolls.
  const t0 = Math.max(now - TRACE_SECONDS, hist[0].t);
  const spanS = Math.max(now - t0, 1e-3);
  const win = hist.filter((h) => h.t >= t0);
  const active = S.frame && S.frame.status ? S.frame.status.filter_type : "kalman";
  const shown = S.allFilters ? Object.keys(FILTER_NAMES) : [active];

  for (const entry of S.charts) {
    const { ch, canvas, range, now: nowEl } = entry;
    const dpr = globalThis.devicePixelRatio || 1;
    const W = canvas.clientWidth || 420, H = 104;
    if (canvas.width !== Math.round(W * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // Auto-scale to the data present, with a floor so a still arm is not
    // magnified into apparent violent motion.
    let lo = Infinity, hi = -Infinity;
    for (const h of win) {
      for (const s of [h.raw, h.filt]) {
        if (s) { lo = Math.min(lo, s[ch.key]); hi = Math.max(hi, s[ch.key]); }
      }
    }
    if (!isFinite(lo)) { lo = -10; hi = 10; }
    const mid = (lo + hi) / 2, span = Math.max(hi - lo, 12);
    lo = mid - span * 0.66; hi = mid + span * 0.66;
    range.textContent = `${lo.toFixed(0)}…${hi.toFixed(0)}°`;

    // The right gutter keeps the newest sample clear of the current-value
    // label; it is a plot margin, not a distortion of the time axis.
    const pad = { t: 20, b: 8, r: 62 };
    const plotW = Math.max(W - pad.r, 40);
    const X = (t) => ((t - t0) / spanS) * plotW;
    const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);

    // recessive zero line
    if (lo < 0 && hi > 0) {
      ctx.strokeStyle = pal.grid;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(plotW, Y(0)); ctx.stroke();
    }

    const drawSeries = (get, color, width) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.beginPath();
      let pen = false;
      for (const h of win) {
        const s = get(h);
        if (!s) { pen = false; continue; }   // gap: limb untracked, not zero
        const x = X(h.t), y = Y(s[ch.key]);
        if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
      }
      ctx.stroke();
    };

    drawSeries((h) => h.raw, pal.raw, 1);
    for (const ft of shown) drawSeries((h) => h.banks[ft], filterColor(ft, pal), 2);

    // current value, direct-labelled instead of a number on every point
    const last = win[win.length - 1];
    const lastFilt = last && last.banks[active];
    nowEl.textContent = lastFilt ? `${signed(lastFilt[ch.key])}°` : "";
    nowEl.style.color = filterColor(active, pal);

    // hover crosshair
    if (S.hover && win.length) {
      const hx = clamp(S.hover.x, 0, plotW);
      const ht = t0 + (hx / plotW) * spanS;
      let best = win[0], bd = Infinity;
      for (const h of win) { const d = Math.abs(h.t - ht); if (d < bd) { bd = d; best = h; } }
      const bx = X(best.t);
      ctx.strokeStyle = pal.axis;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(bx, pad.t - 6); ctx.lineTo(bx, H - pad.b); ctx.stroke();
      ctx.setLineDash([]);
      for (const ft of shown) {
        const s = best.banks[ft];
        if (!s) continue;
        ctx.beginPath();
        ctx.arc(bx, Y(s[ch.key]), 4, 0, Math.PI * 2);
        ctx.fillStyle = filterColor(ft, pal);
        ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = pal.surface; ctx.stroke();
      }
      if (S.hover.chart === entry) showTooltip(best, ch, shown, pal, now);
    }
  }
}

function showTooltip(sample, ch, shown, pal, now) {
  const tip = $("chartTip");
  if (!tip || !S.hover) return;
  const rows = [];
  if (sample.raw) {
    rows.push(`<div class="t-row"><i style="background:${pal.raw}"></i>
      <span>raw</span><b>${signed(sample.raw[ch.key], 2)}°</b></div>`);
  }
  for (const ft of shown) {
    const s = sample.banks[ft];
    if (!s) continue;
    rows.push(`<div class="t-row"><i style="background:${filterColor(ft, pal)}"></i>
      <span>${FILTER_NAMES[ft]}</span><b>${signed(s[ch.key], 2)}°</b></div>`);
  }
  if (!rows.length) { hideTooltip(); return; }

  tip.innerHTML = `<div class="t-time">${ch.label} · ${(sample.t - now).toFixed(1)} s</div>${rows.join("")}`;
  tip.classList.add("on");

  const host = $("charts").getBoundingClientRect();
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = S.hover.clientX - host.left + 14;
  if (left + tw > host.width) left = S.hover.clientX - host.left - tw - 14;
  let top = S.hover.clientY - host.top - th - 10;
  if (top < 0) top = S.hover.clientY - host.top + 16;
  tip.style.left = `${clamp(left, 4, Math.max(4, host.width - tw - 4))}px`;
  tip.style.top = `${top}px`;
}

// ---------------------------------------------------------------------------
// derivation trace
// ---------------------------------------------------------------------------

const vecStr = (v) => v ? `[${v.map((c) => c.toFixed(3)).join(", ")}]` : "—";
const confCls = (v) => v >= 0.7 ? "good" : v >= 0.4 ? "warn" : "bad";
const angleDoc = (key) => S.explain ? S.explain.angles.find((a) => a.key === key) : null;

function renderTrace(fr) {
  const host = $("tracePanel");
  const side = S.traceSide;
  const tr = fr.trace && fr.trace.sides ? fr.trace.sides[side] : null;
  if (!tr || !fr.trace.torso_frame) {
    host.innerHTML = `<p class="muted">The ${side} arm is not currently tracked, so no
      derivation is shown. Nothing is estimated for an untracked limb.</p>`;
    return;
  }
  const tf = fr.trace.torso_frame;
  const a = tr.angles;
  const cal = fr.calibrated ? fr.calibrated[side] : null;
  const filt = fr.filtered ? fr.filtered[side] : null;
  const idx = tr.landmark_indices;
  const unit = fr.trace.coordinate_space ? fr.trace.coordinate_space.unit : "";

  host.innerHTML = `
    <div class="step">
      <h4>1 · Keypoints read from this frame</h4>
      <p>Keypoints from the pose network, each with its own confidence. Angles are
         computed in ${esc(fr.trace.coordinate_space
           ? fr.trace.coordinate_space.label : "pixel space")}.</p>
      <table>
        <thead><tr><th>joint</th><th>index</th><th>visibility</th></tr></thead>
        <tbody>
          ${["shoulder", "elbow", "wrist"].map((j) => `
            <tr><td>${j}</td><td>${idx[j]}</td>
              <td class="${confCls(tr.visibility[j])}">${tr.visibility[j].toFixed(2)}</td></tr>`).join("")}
        </tbody>
      </table>
    </div>

    <div class="step">
      <h4>2 · Torso reference frame</h4>
      <p>Built from the shoulder and hip keypoints and orthonormalised, so the angles
         do not change when the subject turns relative to the camera.</p>
      <div class="kv">lateral X <b>${vecStr(tf.x_axis_lateral)}</b></div>
      <div class="kv">superior Y <b>${vecStr(tf.y_axis_superior)}</b></div>
      <div class="kv">anterior Z <b>${vecStr(tf.z_axis_anterior)}</b></div>
    </div>

    <div class="step">
      <h4>3 · Segment vectors in that frame</h4>
      <p>Two-link model: upper arm (shoulder→elbow) and forearm (elbow→wrist).</p>
      <div class="kv">upper arm <b>${vecStr(tr.upper_arm_torso)}</b> ·
        length ${tr.segment_lengths.upper_arm.toFixed(3)} ${unit}</div>
      <div class="kv">forearm&nbsp;&nbsp; <b>${vecStr(tr.forearm_torso)}</b> ·
        length ${tr.segment_lengths.forearm.toFixed(3)} ${unit}</div>
    </div>

    <div class="step">
      <h4>4 · Angles from geometry</h4>
      ${CHANNELS.map((ch) => {
        const doc = angleDoc(ch.key);
        const grey = ch.key === "shoulder_rotation" && a && !a.rotation_reliable;
        return `<div class="kv${grey ? " unreliable" : ""}" style="margin-bottom:4px">
            <code>${esc(doc ? doc.formula : "")}</code> →
            <b>${a ? signed(a[ch.key]) : "—"}°</b>
            <span class="muted">${ch.short}${grey ? " · not observable at this elbow angle" : ""}</span>
          </div>`;
      }).join("")}
    </div>

    <div class="step">
      <h4>5 · Filtering, calibration, transmission</h4>
      <table>
        <thead><tr><th>channel</th><th>raw</th><th>filtered</th><th>to Unity</th></tr></thead>
        <tbody>
          ${CHANNELS.map((ch) => `
            <tr><td>${ch.label}</td>
              <td>${a ? a[ch.key].toFixed(2) : "—"}</td>
              <td>${filt ? filt[ch.key].toFixed(2) : "—"}</td>
              <td><b>${cal ? cal[ch.key].toFixed(2) : "—"}</b></td></tr>`).join("")}
        </tbody>
      </table>
      <p class="muted" style="margin-top:7px">The right-hand column is exactly what
         appears in the UDP packet.</p>
    </div>`;
}

// ---------------------------------------------------------------------------
// latency + filters
// ---------------------------------------------------------------------------

function renderLatency(fr) {
  const t = fr.timings || {};
  const stages = S.explain ? S.explain.stages : [];
  const host = $("stageBars");
  if (!host.childElementCount && stages.length) {
    for (const st of stages) {
      const b = document.createElement("div");
      b.className = "stagebar";
      b.title = st.description;
      b.innerHTML = `<div class="top"><span>${st.label}</span><b id="ms-${st.key}">—</b></div>
                     <div class="track"><div id="bar-${st.key}" style="width:0%"></div></div>`;
      host.appendChild(b);
    }
  }
  const total = Math.max(t.total_ms || 0, 1);
  for (const st of stages) {
    const v = t[st.key] || 0;
    const ms = $(`ms-${st.key}`), bar = $(`bar-${st.key}`);
    if (ms) ms.textContent = `${v.toFixed(1)} ms`;
    if (bar) bar.style.width = `${clamp((v / total) * 100, 0, 100)}%`;
  }

  const m = S.metrics;
  if (!m) return;
  const rows = [
    ["Server pipeline", m.latency.total_ms],
    ["Pose inference", m.latency.pose_ms],
    ["Frame decode", m.latency.decode_ms],
  ];
  $("latTable").innerHTML = rows.map(([n, s]) =>
    `<tr><td>${n}</td><td>${fmt(s.mean)}</td><td>${fmt(s.p95)}</td><td>${fmt(s.max)}</td></tr>`
  ).join("") + (S.lastRtt !== undefined
    ? `<tr><td>Browser round trip</td><td colspan="3">${fmt(S.lastRtt)} ms latest</td></tr>` : "");

  $("budgetNote").innerHTML =
    `<b>${(m.within_budget * 100).toFixed(1)}%</b> of the last ${m.latency.total_ms.n}
     frames completed within the ${m.latency_budget_ms} ms budget. Uptime
     ${fmt(m.uptime_s, 0)} s over ${m.frames} frames.`;
}

function renderFilterTable() {
  const m = S.metrics;
  if (!m) return;
  const side = S.chartSide;
  $("cmpSide").textContent = side;
  const active = S.frame && S.frame.status ? S.frame.status.filter_type : "kalman";
  const rows = [["raw", "Raw (unfiltered)"], ["kalman", "Kalman"],
                ["ma", "Moving average"], ["sg", "Savitzky–Golay"]];
  $("filterTable").innerHTML = rows.map(([key, label]) => {
    const s = m.angle_std[`${key}:${side}`];
    const cells = CHANNELS.map((ch) => {
      if (!s) return "<td>—</td>";
      const v = s[ch.key];
      const cls = key === "raw" ? "" : (v <= 3 ? "good" : v <= 5 ? "warn" : "bad");
      return `<td class="${cls}">${v.toFixed(2)}</td>`;
    }).join("");
    return `<tr class="${key === active ? "is-active" : ""}"><td>${label}</td>${cells}</tr>`;
  }).join("");
  $("kpJitter").textContent = m.keypoint_jitter ? m.keypoint_jitter.mean_rms_px.toFixed(2) : "—";
}

// ---------------------------------------------------------------------------
// Unity wire
// ---------------------------------------------------------------------------

let lastPktCount = 0, lastPktTime = 0;

function renderWire(fr) {
  const udp = fr.status && fr.status.udp;
  if (!udp) return;
  const wire = $("wire");
  if (udp.last_packet) {
    wire.className = "wire";
    wire.textContent = udp.last_packet;
  } else {
    wire.className = "wire idle";
    wire.textContent = udp.enabled ? "waiting for the first transmitted packet…"
                                   : "Unity stream disabled";
  }
  setPill($("pPackets"), `<b>${udp.packets_sent.toLocaleString()}</b> packets`);
  const now = performance.now();
  if (lastPktTime && now > lastPktTime) {
    const rate = (udp.packets_sent - lastPktCount) / ((now - lastPktTime) / 1000);
    if (rate >= 0) setPill($("pRate"), `<b>${rate.toFixed(1)}</b> pkt/s`,
                           rate >= 20 ? "ok" : "warn");
  }
  lastPktCount = udp.packets_sent; lastPktTime = now;
  setPill($("pErrs"), `<b>${udp.send_errors}</b> errors`, udp.send_errors ? "bad" : "");

  if (udp.history && udp.history.length) {
    $("wireLog").innerHTML = udp.history.slice().reverse()
      .map((h) => `<div>${new Date(h.t * 1000).toLocaleTimeString()}  ${esc(h.packet)}</div>`)
      .join("");
  }
}

function syncControls(fr) {
  const st = fr.status || {};
  if (st.filter_type && $("selFilter").value !== st.filter_type) {
    $("selFilter").value = st.filter_type;
  }
  $("btnLogStart").disabled = !!st.logging;
  $("btnLogStop").disabled = !st.logging;
  if (st.logging) {
    $("logNotice").className = "notice ok";
    $("logNotice").textContent = "Recording joint angles to CSV…";
  }
}

function onStatus(st) {
  if (!st) return;
  S.metrics = st.metrics;
  if (st.udp) {
    $("inUdpHost").value = st.udp.host;
    $("inUdpPort").value = st.udp.port;
    $("inUdpHz").value = st.udp.target_hz || (st.config && st.config.stream_hz) || 30;
    $("chkUdp").checked = !!st.udp.enabled;
  }
  if (st.filter_type) $("selFilter").value = st.filter_type;
  if (st.calibration) renderCalibration(st.calibration);
  if (st.logging) {
    $("btnLogStart").disabled = !!st.logging.active;
    $("btnLogStop").disabled = !st.logging.active;
    $("logNotice").textContent = st.logging.active
      ? `Recording → ${st.logging.filename}`
      : (st.logging.filename ? `Last session: ${st.logging.filename}` : "Not recording.");
  }
}

// ---------------------------------------------------------------------------
// calibration
// ---------------------------------------------------------------------------

function renderCalibration(cal) {
  S.calibration = cal;
  const side = $("selCalSide").value;

  $("calChips").innerHTML = (cal.required_poses || []).map((p) => {
    const done = (cal.captured || []).includes(p);
    const now = cal.active && cal.pose === p;
    return `<span class="chip ${done ? "done" : now ? "now" : ""}">${
      done ? "✓ " : ""}${p.replace("_", " ")}</span>`;
  }).join("");

  const notice = $("calNotice");
  if (cal.active) {
    notice.className = "notice act";
    notice.innerHTML = `<b>${esc(POSE_LABELS[cal.pose] || cal.pose)}</b><br>
      <span class="muted">Hold it steady, then press Capture. The last 15 filtered
      frames are averaged, so one noisy frame cannot skew the mapping.</span>`;
  } else if (cal.calibrated && cal.calibrated[side]) {
    const warn = (cal.warnings && cal.warnings[side]) || [];
    if (warn.length) {
      notice.className = "notice err";
      notice.innerHTML = `<b>${side} arm calibrated with problems</b><br>` +
        warn.map((w) => `• ${esc(w)}`).join("<br>");
    } else {
      notice.className = "notice ok";
      notice.textContent = `${side} arm calibrated — offsets and scales applied to every frame.`;
    }
  } else {
    notice.className = "notice";
    notice.textContent = "Not calibrated — raw geometric estimates are being sent to Unity.";
  }

  $("btnCalCapture").disabled = !cal.active;
  $("btnCalCancel").disabled = !cal.active;
  $("btnCalBegin").disabled = !!cal.active;

  const params = (cal.parameters || {})[side] || {};
  $("calTable").innerHTML = ["flexion", "abduction", "rotation", "elbow"].map((dof) => {
    const p = params[dof] || { offset: 0, scale: 1 };
    return `<tr><td>${dof}</td><td>${p.offset.toFixed(2)}</td><td>${p.scale.toFixed(3)}</td></tr>`;
  }).join("");
}

// ---------------------------------------------------------------------------
// sessions
// ---------------------------------------------------------------------------

let openSession = null;

async function refreshSessions() {
  const data = await api("/api/sessions");
  const body = $("sessionTable");
  if (!data.sessions.length) {
    body.innerHTML = `<tr><td colspan="3" class="muted">No sessions recorded yet.</td></tr>`;
    return;
  }
  body.innerHTML = data.sessions.map((s) => `
    <tr><td>${esc(s.name)}</td><td>${(s.size_bytes / 1024).toFixed(1)} kB</td>
      <td style="white-space:nowrap">
        <button class="sm ghost" data-session="${esc(s.name)}">Inspect</button>
        <a href="/api/sessions/${encodeURIComponent(s.name)}" download>CSV</a>
      </td></tr>`).join("");
  body.querySelectorAll("[data-session]").forEach((b) => {
    b.onclick = () => showSession(b.dataset.session);
  });
}

async function showSession(name) {
  openSession = name;
  const host = $("sessionDetail");
  host.innerHTML = `<p class="muted" style="margin-top:10px">Computing statistics from ${esc(name)}…</p>`;
  try {
    const s = await api(`/api/sessions/${encodeURIComponent(name)}/summary`);
    const rows = CHANNELS.map((ch) => {
      const r = s.channels[`right_${ch.key}`] || { n: 0 };
      const l = s.channels[`left_${ch.key}`] || { n: 0 };
      return `<tr><td>${ch.label}</td>
        <td>${r.n ? r.mean.toFixed(1) : "—"}</td><td>${r.n ? r.std.toFixed(2) : "—"}</td>
        <td>${l.n ? l.mean.toFixed(1) : "—"}</td><td>${l.n ? l.std.toFixed(2) : "—"}</td></tr>`;
    }).join("");
    host.innerHTML = `
      <p class="muted" style="margin:12px 0 6px">
        <b>${esc(s.name)}</b> — ${s.rows} frames over ${s.duration_s} s
        (${s.mean_rate_hz} Hz mean). Right arm tracked in
        ${(s.tracked_fraction.right * 100).toFixed(1)}% of frames, left in
        ${(s.tracked_fraction.left * 100).toFixed(1)}%.</p>
      <table>
        <thead><tr><th>channel</th><th>R mean</th><th>R σ</th><th>L mean</th><th>L σ</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <img class="plot-img" id="sPlot" alt="joint angle time series">
      <img class="plot-img" id="sDist" alt="joint angle distribution">`;
    refreshSessionPlots();
  } catch (err) {
    host.innerHTML = `<div class="notice err">${esc(err.message)}</div>`;
  }
}

/** Re-request the server-rendered plots so they match the active theme. */
function refreshSessionPlots() {
  if (!openSession) return;
  const q = `side=right&theme=${currentTheme()}&v=${Date.now()}`;
  const plot = $("sPlot"), dist = $("sDist");
  if (plot) plot.src = `/api/sessions/${encodeURIComponent(openSession)}/plot.png?${q}`;
  if (dist) dist.src = `/api/sessions/${encodeURIComponent(openSession)}/distribution.png?${q}`;
}

// ---------------------------------------------------------------------------
// figure gallery
// ---------------------------------------------------------------------------

async function refreshFigures() {
  const host = $("figurePanel");
  try {
    S.figures = await api("/api/figures");
  } catch (err) {
    host.innerHTML = `<div class="notice err">${esc(err.message)}</div>`;
    return;
  }
  const all = S.figures.groups.flatMap((g) => g.figures);
  if (!all.length) {
    host.innerHTML = `<div class="empty">
      No figures on disk yet. They are written by the project's analysis scripts —
      run <code>python src/evaluation/eval_plots.py</code>,
      <code>python scripts/compare_filters.py</code> or
      <code>python benchmarks/visualize_benchmarks.py</code>, then refresh.
      Nothing is generated here, so an empty gallery means those runs have not happened.
    </div>`;
    return;
  }
  host.innerHTML = `
    <p class="muted" style="margin:0 0 12px">${all.length} figure${all.length === 1 ? "" : "s"}
      found on disk. Click one to enlarge.</p>
    <div class="figures">${all.map((f) => `
      <figure class="figure" data-fig="${esc(f.id)}" style="margin:0">
        <div class="thumb"><img loading="lazy" src="/api/figures/${f.id.split("/").map(encodeURIComponent).join("/")}" alt="${esc(f.name)}"></div>
        <figcaption class="meta">
          <h5>${esc(f.name)}</h5>
          <p>${esc(f.description || "Figure produced by the project's tooling.")}</p>
          ${f.generated_by ? `<span class="src">${esc(f.generated_by)}</span>` : ""}
        </figcaption>
      </figure>`).join("")}</div>`;

  host.querySelectorAll("[data-fig]").forEach((el) => {
    el.onclick = () => {
      const f = all.find((x) => x.id === el.dataset.fig);
      openLightbox(`/api/figures/${f.id.split("/").map(encodeURIComponent).join("/")}`,
                   `${f.name}${f.generated_by ? " — " + f.generated_by : ""}`);
    };
  });
}

function openLightbox(src, caption) {
  $("lbImg").src = src;
  $("lbCap").textContent = caption;
  $("lightbox").classList.add("on");
}

// ---------------------------------------------------------------------------
// explanations
// ---------------------------------------------------------------------------

function explainHtml(key) {
  const e = S.explain;
  if (!e) return "<p class='muted'>Explanation unavailable.</p>";

  switch (key) {
    case "live": return `
      <p>Frames come from one of three real sources — never a synthetic generator.</p>
      <h4>Browser webcam</h4><p>The page captures your camera with
        <code>getUserMedia</code> and pushes JPEG frames over the WebSocket. One frame
        is in flight at a time, because the server processes serially.</p>
      <h4>Server camera</h4><p>OpenCV opens a camera attached to the machine running
        Python. Use it when you reach the server remotely.</p>
      <h4>Recorded video</h4><p>A file replayed at its own frame rate. Identical frames
        every run, so a filter or framework change is the only variable — the
        reproducible path for comparisons.</p>
      <p>The overlay draws confident keypoints filled and low-confidence ones hollow,
        so tracking certainty is visible rather than implied.</p>`;

    case "angles": return e.angles.map((a) => `
      <h4>${esc(a.label)}</h4>
      <p>${esc(a.description)}</p>
      <code class="formula">${esc(a.formula)}</code>
      <p class="muted">${esc(a.sign)} · neutral: ${esc(a.neutral)} ·
        typical ${a.typical_range[0]}…${a.typical_range[1]}° ·
        from ${esc(a.reads_from)}</p>`).join("");

    case "traces": return `
      <p>Ten seconds of history per channel. The thin grey line is the raw geometric
        estimate; the thick line is the filter currently being transmitted to Unity.
        Hover anywhere to read exact values at that instant.</p>
      <p><b>Compare filters</b> overlays all three filter families at once — the live
        counterpart of the offline filter comparison.</p>
      <p>A break in a line is an untracked limb, not a zero. The vertical scale
        auto-fits the data with a floor, so a still arm is not magnified into
        apparent violent motion.</p>`;

    case "derivation": return `
      <p>The five steps that turn this frame's pixels into the numbers Unity receives.
        Every intermediate is the real value used, read back from the pipeline.</p>
      <p>Step 2's reference frame is what makes the angles independent of how the
        subject is turned relative to the camera; step 4 shows the formula and its
        result side by side; step 5's last column is byte-for-byte the UDP packet.</p>`;

    case "protocol": {
      const p = e.protocol;
      return `
        <p>${esc(p.transport)}, ${esc(p.encoding)}, ${esc(p.rate)}.</p>
        ${p.packets.map((k) => `
          <h4>${esc(k.prefix)} — ${esc(k.description)}</h4>
          <code class="formula">${esc(k.format)}</code>`).join("")}
        <p>${esc(p.hold_behaviour)}</p>
        <p class="muted">Unity receiver: <code>${esc(p.unity_receiver)}</code></p>`;
    }

    case "stages": return `
      <p>Each stage is timed with <code>perf_counter()</code> around the actual call —
        these are measurements, not estimates.</p>
      ${e.stages.map((s) => `<h4>${esc(s.label)}</h4><p>${esc(s.description)}</p>`).join("")}
      <p>The browser round trip is measured separately on the page's own clock, because
        the browser and server clocks cannot be compared directly.</p>`;

    case "filters": return `
      <p>All three run on every frame so they can be compared live; only the selected
        one reaches Unity and the session log.</p>
      ${e.filters.map((f) => `
        <h4>${esc(f.label)}</h4><p>${esc(f.description)}</p>
        <p class="muted">${esc(f.parameters)} — ${esc(f.tradeoff)}</p>`).join("")}
      <p>The table shows the rolling standard deviation per channel. Hold a pose still
        and a lower number means a steadier signal; the specification asks for ≤ 3–5°
        after filtering.</p>`;

    case "calibration": return `
      <p>Four reference poses per arm map your measured range onto the avatar's joints.
        Each capture averages the last 15 filtered frames — about half a second — so a
        single noisy frame cannot define the mapping.</p>
      <p><code>arm_down</code> fixes the per-channel offset; the other three fix the gain.</p>
      <h4>Why a pose can be rejected</h4>
      <p>A reference pose less than 20° from neutral cannot support a gain estimate —
        the ratio would be dominated by noise — so that axis is left uncalibrated and
        the panel names the pose to repeat. A fitted gain outside 0.25–4.0 is clamped
        and reported for the same reason.</p>
      <p>Parameters are saved to <code>outputs/web/calibration_&lt;side&gt;.json</code>
        and reloaded on the next start.</p>`;

    case "logging": return `
      <p>One CSV row per frame: both arms' four angles, the per-side tracked flag, the
        rotation-reliability flag, the active filter and whether calibration was applied.</p>
      <p>An untracked side is written as empty cells with <code>tracked = 0</code>, so
        occlusion stays distinguishable from a genuine zero reading.</p>
      <p><b>Inspect</b> computes statistics from the CSV itself and renders a time
        series and a distribution — the distribution is the view that matters when the
        log is used as a reference signal for calibrating another sensor.</p>`;

    case "figures": return `
      <p>Plots written to disk by this project's own analysis scripts — the evaluation
        figures, filter and framework comparisons, occlusion and latency benchmarks.</p>
      <p>Nothing is generated by this page. The gallery reports what exists, so an empty
        section honestly means those scripts have not been run yet rather than showing
        a placeholder.</p>`;

    default: return "<p class='muted'>No explanation for this section.</p>";
  }
}

function wireExplainButtons() {
  document.querySelectorAll("[data-explain]").forEach((btn) => {
    const key = btn.dataset.explain;
    btn.onclick = () => {
      const panel = $(`ex-${key}`);
      const open = panel.classList.toggle("open");
      btn.setAttribute("aria-expanded", String(open));
      if (open && !panel.dataset.filled) {
        $(`exc-${key}`).innerHTML = explainHtml(key);
        panel.dataset.filled = "1";
      }
    };
  });
}

async function loadExplain() {
  S.explain = await api("/api/explain");

  $("dataFlow").innerHTML = S.explain.data_flow
    .map((s, i) => `${i ? "<i>→</i>" : ""}<span>${esc(s)}</span>`).join("");

  $("refAngles").innerHTML = S.explain.angles.map((a) => `
    <div style="padding:9px 0;border-bottom:1px solid var(--line)">
      <h5 style="margin:0 0 3px;font-size:12.5px">${esc(a.label)}</h5>
      <p style="margin:0 0 4px;font-size:12px;color:var(--text-muted)">${esc(a.description)}</p>
      <code class="formula">${esc(a.formula)}</code>
    </div>`).join("");

  $("refReq").innerHTML = S.explain.requirements.map((r) =>
    `<tr><td>${esc(r.key.replace("_", " "))}</td><td>${esc(r.target)}</td></tr>`).join("");

  $("refFlags").innerHTML = S.explain.quality_flags.map((q) => `
    <div style="padding:7px 0;border-bottom:1px solid var(--line)">
      <code style="font-family:var(--mono);font-size:11.5px">${esc(q.key)}</code>
      <p style="margin:2px 0 0;font-size:12px;color:var(--text-muted)">${esc(q.description)}</p>
    </div>`).join("");
}

// ---------------------------------------------------------------------------
// sources
// ---------------------------------------------------------------------------

async function refreshSources() {
  const data = await api("/api/sources");
  const sel = $("selSourceItem");
  const mode = $("selSource").value;
  if (mode === "camera") {
    sel.innerHTML = data.cameras.length
      ? data.cameras.map((c) => `<option value="${c.index}">camera ${c.index} — ${c.width}×${c.height}</option>`).join("")
      : `<option value="">no server camera detected</option>`;
  } else if (mode === "file") {
    sel.innerHTML = data.videos.length
      ? data.videos.map((v) => `<option value="${esc(v.path)}">${esc(v.name)}</option>`).join("")
      : `<option value="">no videos in ${esc(data.video_dir)}</option>`;
  }
  sel.style.display = mode === "browser" ? "none" : "";
}

async function applySource() {
  const mode = $("selSource").value;
  S.sourceMode = mode;
  const item = $("selSourceItem").value;

  if (mode === "browser") {
    $("serverPreview").style.display = "none";
    $("overlay").style.display = "";
    $("srcHint").textContent = "browser webcam";
    await api("/api/source", "POST", { mode: "browser" });
    return;
  }
  stopCamera();
  try {
    const body = mode === "camera"
      ? { mode, camera_index: Number(item || 0) }
      : { mode, path: item };
    const st = await api("/api/source", "POST", body);
    $("placeholder").style.display = "none";
    $("serverPreview").src = "/api/preview.mjpg?" + Date.now();
    $("serverPreview").style.display = "";
    $("overlay").style.display = "none";
    $("srcHint").textContent = mode === "camera"
      ? `server camera ${st.camera_index}`
      : `replay: ${st.path.split(/[\\/]/).pop()}`;
  } catch (err) {
    showPlaceholder("Source unavailable", err.message);
    $("selSource").value = "browser";
    S.sourceMode = "browser";
  }
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

function wire() {
  $("btnStart").onclick = startCamera;
  $("btnStop").onclick = stopCamera;
  $("btnTheme").onclick = () => applyTheme(currentTheme() === "dark" ? "light" : "dark");
  $("btnMenu").onclick = () => $("sidebar").classList.toggle("open");

  $("selRate").onchange = (e) => { S.sendInterval = 1000 / Number(e.target.value); };
  $("chkMirror").onchange = (e) => api("/api/mirror", "POST", { enabled: e.target.checked });

  $("selFilter").onchange = async (e) => {
    await api("/api/filter", "POST", { type: e.target.value });
    renderLegend(true);
  };
  $("btnResetFilters").onclick = () => api("/api/reset", "POST");

  $("selChartSide").onchange = (e) => { S.chartSide = e.target.value; renderCharts(); renderFilterTable(); };
  $("selTraceSide").onchange = (e) => {
    S.traceSide = e.target.value;
    if (S.frame) renderTrace(S.frame);
  };
  $("chkAllFilters").onchange = (e) => {
    S.allFilters = e.target.checked;
    renderLegend(true); renderCharts();
  };

  $("btnUdpApply").onclick = async () => {
    const st = await api("/api/udp", "POST", {
      enabled: $("chkUdp").checked,
      host: $("inUdpHost").value.trim(),
      port: Number($("inUdpPort").value),
      hz: Number($("inUdpHz").value),
    });
    onStatus({ udp: st });
  };
  $("chkUdp").onchange = () => $("btnUdpApply").click();

  $("btnCalBegin").onclick = async () =>
    renderCalibration(await api("/api/calibration/begin", "POST", { side: $("selCalSide").value }));
  $("btnCalCapture").onclick = async () => {
    try { renderCalibration(await api("/api/calibration/capture", "POST")); }
    catch (err) {
      const n = $("calNotice"); n.className = "notice err"; n.textContent = err.message;
    }
  };
  $("btnCalCancel").onclick = async () =>
    renderCalibration(await api("/api/calibration/cancel", "POST"));
  $("btnCalClear").onclick = async () =>
    renderCalibration(await api("/api/calibration/clear", "POST", { side: $("selCalSide").value }));
  $("selCalSide").onchange = () => { if (S.calibration) renderCalibration(S.calibration); };

  $("btnLogStart").onclick = async () => {
    onStatus({ logging: await api("/api/logging/start", "POST",
                                  { label: $("inLogLabel").value.trim() }) });
  };
  $("btnLogStop").onclick = async () => {
    onStatus({ logging: await api("/api/logging/stop", "POST") });
    refreshSessions();
  };

  $("selSource").onchange = async () => { await refreshSources(); applySource(); };
  $("selSourceItem").onchange = applySource;
  $("btnRefreshFigures").onclick = refreshFigures;

  $("lbClose").onclick = () => $("lightbox").classList.remove("on");
  $("lightbox").onclick = (e) => { if (e.target.id === "lightbox") e.currentTarget.classList.remove("on"); };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("lightbox").classList.remove("on");
  });

  // sidebar section highlighting
  const links = [...document.querySelectorAll("#nav a")];
  const sections = links.map((a) => $(a.getAttribute("href").slice(1))).filter(Boolean);
  const syncNav = () => {
    const anchor = window.innerHeight * 0.25;
    let best = null, bestD = Infinity;
    for (const sec of sections) {
      const d = Math.abs(sec.getBoundingClientRect().top - anchor);
      if (d < bestD) { bestD = d; best = sec; }
    }
    if (best) links.forEach((a) =>
      a.classList.toggle("active", a.getAttribute("href") === `#${best.id}`));
  };
  window.addEventListener("scroll", syncNav, { passive: true });
  window.addEventListener("resize", syncNav);
  syncNav();
  links.forEach((a) => a.addEventListener("click", () => $("sidebar").classList.remove("open")));

  window.addEventListener("resize", renderCharts);
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

(async function boot() {
  wire();
  buildCharts();
  applyTheme(currentTheme());
  try { await loadExplain(); wireExplainButtons(); } catch (e) { console.warn("explain:", e); }
  try { onStatus(await api("/api/status")); } catch (e) { console.warn("status:", e); }
  try { await refreshSessions(); } catch (e) { console.warn("sessions:", e); }
  try { await refreshSources(); } catch (e) { console.warn("sources:", e); }
  try { await refreshFigures(); } catch (e) { console.warn("figures:", e); }
  connect();
  setInterval(() => { if (S.ws && S.wsReady) S.ws.send(JSON.stringify({ type: "status" })); }, 5000);
})();
