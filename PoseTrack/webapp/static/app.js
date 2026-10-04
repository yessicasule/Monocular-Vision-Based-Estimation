/* MonoArm dashboard front-end.
 *
 * Responsibilities
 *   1. capture real webcam frames in the browser and push them to the server
 *   2. render the returned per-frame result: skeleton overlay, joint angles,
 *      time-series traces, and the derivation behind each number
 *   3. drive the control plane (filter, calibration, logging, Unity link)
 *
 * No external libraries: charts and the overlay are drawn with the 2D canvas
 * API so the page works with no network access beyond this server.
 */
"use strict";

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

const CHANNELS = [
  { key: "shoulder_flexion",   label: "Shoulder flexion",    short: "flexion",   cls: "flex",  color: "#4ade80", range: [-60, 180] },
  { key: "shoulder_abduction", label: "Shoulder abduction",  short: "abduction", cls: "abd",   color: "#fbbf24", range: [-45, 180] },
  { key: "shoulder_rotation",  label: "Shoulder rotation",   short: "rotation",  cls: "rot",   color: "#c084fc", range: [-90,  90] },
  { key: "elbow_flexion",      label: "Elbow flexion",       short: "elbow",     cls: "elbow", color: "#f472b6", range: [  0, 150] },
];

const FILTER_COLORS = { kalman: "#4ea8de", ma: "#e8a33d", sg: "#7ee787" };
const FILTER_NAMES  = { kalman: "Kalman", ma: "Moving avg", sg: "Savitzky–Golay" };
const RAW_COLOR     = "#55657a";

const EDGES = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [12, 14], [14, 16],
  [11, 13], [13, 15],
];
const ARM_POINTS = new Set([11, 12, 13, 14, 15, 16]);

const TRACE_SECONDS = 10;
const HISTORY_MAX   = 600;

// ---------------------------------------------------------------------------
// tiny helpers
// ---------------------------------------------------------------------------

const $  = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt  = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : v.toFixed(d);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

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

// ---------------------------------------------------------------------------
// application state
// ---------------------------------------------------------------------------

const S = {
  ws: null,
  wsReady: false,
  stream: null,
  capturing: false,
  sending: false,
  sendInterval: 1000 / 30,
  lastSentAt: 0,
  sendClaimedAt: 0,
  pendingSentAt: 0,
  frame: null,          // most recent FrameResult
  metrics: null,
  explain: null,
  history: { right: [], left: [] },
  chartSide: "right",
  traceSide: "right",
  allFilters: false,
  sourceMode: "browser",
  charts: [],
  calibration: null,
};

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    S.wsReady = true;
    setPill($("pillConn"), "connected", "ok");
  };
  ws.onclose = () => {
    S.wsReady = false;
    S.sending = false;
    setPill($("pillConn"), "reconnecting…", "warn");
    setTimeout(connect, 1200);
  };
  ws.onerror = () => setPill($("pillConn"), "socket error", "bad");
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "frame")       onFrame(msg);
    else if (msg.type === "hello")  onStatus(msg.status);
    else if (msg.type === "status") onStatus(msg.status);
    else if (msg.type === "error")  { S.sending = false; console.warn("pipeline:", msg.message); }
  };
  S.ws = ws;
}

function onFrame(msg) {
  // A browser-sourced frame completes a round trip we can measure honestly:
  // the browser stamped the send time, so this is true glass-to-glass minus
  // the display step, on one clock.
  if (msg.source === "browser" && S.pendingSentAt) {
    const rtt = performance.now() - S.pendingSentAt;
    S.pendingSentAt = 0;
    S.sending = false;
    if (S.ws && S.wsReady && Math.random() < 0.25) {
      S.ws.send(JSON.stringify({ type: "rtt", ms: rtt }));
    }
    S.lastRtt = rtt;
  }
  S.frame = msg;
  if (msg.metrics) S.metrics = msg.metrics;
  pushHistory(msg);
  render(msg);
}

function pushHistory(msg) {
  const t = msg.t;
  for (const side of ["right", "left"]) {
    const raw = msg.raw ? msg.raw[side] : null;
    const filt = msg.filtered ? msg.filtered[side] : null;
    const banks = {};
    for (const ft of Object.keys(msg.filters || {})) {
      banks[ft] = msg.filters[ft] ? msg.filters[ft][side] : null;
    }
    const h = S.history[side];
    h.push({ t, raw, filt, banks });
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
    showPlaceholder(
      `Camera unavailable: ${err.message}. ` +
      `A browser will only grant camera access over https:// or on localhost.`
    );
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
  showPlaceholder("Camera stopped.");
}

const grabCanvas = document.createElement("canvas");
const grabCtx = grabCanvas.getContext("2d", { willReadFrequently: true });

function captureLoop() {
  if (!S.capturing) return;
  const video = $("video");
  const now = performance.now();

  // Recover the in-flight slot if a frame never came back (a dropped socket,
  // or a frame the server could not decode).
  if (S.sending && now - S.sendClaimedAt > 3000) S.sending = false;

  // One frame in flight at a time: the server processes serially, so sending
  // faster would only queue latency, not raise the achieved rate. The slot is
  // claimed synchronously, before the asynchronous JPEG encode starts —
  // claiming it inside the callback would let several frames leave together
  // and make the round-trip measurement meaningless.
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

function showPlaceholder(text) {
  const p = $("placeholder");
  p.style.display = "flex";
  p.innerHTML = `<div><strong>${esc(text)}</strong></div>`;
}

// ---------------------------------------------------------------------------
// overlay rendering
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
    const mirror = $("chkMirror").checked;
    ctx.save();
    if (mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
    ctx.drawImage(src, 0, 0, W, H);
    ctx.restore();
  }

  if (!fr || !fr.landmarks || fr.landmarks.length === 0) return;

  const pts = {};
  for (const lm of fr.landmarks) pts[lm.i] = lm;

  ctx.lineWidth = Math.max(2, W / 320);
  for (const [a, b] of EDGES) {
    const p = pts[a], q = pts[b];
    if (!p || !q) continue;
    const arm = ARM_POINTS.has(a) && ARM_POINTS.has(b) && !(a === 11 && b === 12);
    const conf = Math.min(p.v, q.v);
    ctx.strokeStyle = arm ? `rgba(78,168,222,${clamp(conf, 0.25, 1)})`
                          : `rgba(120,140,165,${clamp(conf * 0.7, 0.15, 0.7)})`;
    ctx.beginPath();
    ctx.moveTo(p.x * W, p.y * H);
    ctx.lineTo(q.x * W, q.y * H);
    ctx.stroke();
  }

  for (const lm of fr.landmarks) {
    if (lm.i === 0) continue;
    const r = ARM_POINTS.has(lm.i) ? Math.max(4, W / 160) : Math.max(3, W / 240);
    // Low-visibility keypoints are drawn hollow: the overlay shows tracking
    // confidence rather than implying every point is equally certain.
    ctx.beginPath();
    ctx.arc(lm.x * W, lm.y * H, r, 0, Math.PI * 2);
    if (lm.v >= 0.5) {
      ctx.fillStyle = ARM_POINTS.has(lm.i) ? "#4ea8de" : "#7d8ea3";
      ctx.fill();
    } else {
      ctx.strokeStyle = "#d29922";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // Elbow angle arc, drawn where the joint actually is
  for (const [side, idx] of [["right", [12, 14, 16]], ["left", [11, 13, 15]]]) {
    const a = pts[idx[0]], b = pts[idx[1]], c = pts[idx[2]];
    const ang = fr.calibrated && fr.calibrated[side];
    if (!a || !b || !c || !ang) continue;
    const a1 = Math.atan2(a.y * H - b.y * H, a.x * W - b.x * W);
    const a2 = Math.atan2(c.y * H - b.y * H, c.x * W - b.x * W);
    ctx.beginPath();
    ctx.strokeStyle = "#f472b6";
    ctx.lineWidth = Math.max(2, W / 400);
    ctx.arc(b.x * W, b.y * H, Math.max(16, W / 26), a1, a2,
            ((a2 - a1 + Math.PI * 2) % (Math.PI * 2)) > Math.PI);
    ctx.stroke();
    ctx.fillStyle = "#f9a8d4";
    ctx.font = `${Math.max(12, Math.round(W / 45))}px ui-monospace, monospace`;
    ctx.fillText(`${ang.elbow_flexion.toFixed(0)}°`,
                 b.x * W + Math.max(18, W / 24), b.y * H);
  }
}

// ---------------------------------------------------------------------------
// panels
// ---------------------------------------------------------------------------

function setPill(node, text, kind) {
  node.className = "pill" + (kind ? " " + kind : "");
  node.innerHTML = text;
}

function render(fr) {
  renderPills(fr);
  renderLegend();
  renderArms(fr);
  renderCharts();
  renderTrace(fr);
  renderLatency(fr);
  renderFilterTable();
  renderWire(fr);
  syncControls(fr);
}

function renderPills(fr) {
  const m = S.metrics;
  if (m) {
    const fps = m.fps;
    setPill($("pillFps"), `<b>${fmt(fps, 1)}</b> fps`,
            fps >= 20 ? "ok" : fps > 0 ? "warn" : "");
    const rtt = S.lastRtt;
    const shown = rtt !== undefined ? rtt : m.latency.total_ms.p95;
    setPill($("pillLat"), `<b>${fmt(shown, 0)}</b> ms end-to-end`,
            shown <= 100 ? "ok" : "warn");
    $("winFrames").textContent = m.latency.total_ms.n || 0;
  }
  const udp = fr.status && fr.status.udp;
  if (udp) {
    setPill($("pillUdp"),
      udp.enabled ? `Unity <b>${udp.host}:${udp.port}</b> · ${udp.packets_sent}`
                  : "Unity off",
      udp.enabled ? (udp.send_errors ? "bad" : "ok") : "");
  }
  const tracked = ["right", "left"].filter((s) => fr.calibrated && fr.calibrated[s]);
  setPill($("pillTrack"),
    fr.detected ? (tracked.length ? `tracking ${tracked.join(" + ")}` : "pose, no arms")
                : "no pose",
    fr.detected && tracked.length ? "ok" : "warn");
}

function armCard(side) {
  const card = el("div", "arm");
  card.id = `arm-${side}`;
  card.appendChild(el("h3", null,
    `${side} arm <span class="badge" id="badge-${side}">—</span>`));
  for (const ch of CHANNELS) {
    const row = el("div", "angle-row");
    row.id = `row-${side}-${ch.key}`;
    row.innerHTML = `
      <div class="angle-head">
        <span class="name">${ch.label}</span>
        <span class="raw" id="raw-${side}-${ch.key}">raw —</span>
      </div>
      <div class="angle-value c-${ch.cls}" id="val-${side}-${ch.key}">—<span class="unit">°</span></div>
      <div class="gauge"><div class="zero" id="zero-${side}-${ch.key}"></div>
        <div class="fill bg-${ch.cls}" id="fill-${side}-${ch.key}"></div></div>`;
    card.appendChild(row);
  }
  return card;
}

function renderArms(fr) {
  const grid = $("armGrid");
  if (!grid.childElementCount) {
    grid.appendChild(armCard("right"));
    grid.appendChild(armCard("left"));
  }
  for (const side of ["right", "left"]) {
    const out = fr.calibrated ? fr.calibrated[side] : null;
    const raw = fr.raw ? fr.raw[side] : null;
    const badge = $(`badge-${side}`);
    badge.className = "badge " + (out ? "live" : "lost");
    badge.textContent = out ? "tracked" : "not tracked";

    for (const ch of CHANNELS) {
      const valNode  = $(`val-${side}-${ch.key}`);
      const rawNode  = $(`raw-${side}-${ch.key}`);
      const fillNode = $(`fill-${side}-${ch.key}`);
      const zeroNode = $(`zero-${side}-${ch.key}`);
      const rowNode  = $(`row-${side}-${ch.key}`);

      if (!out) {
        valNode.innerHTML = `—<span class="unit">°</span>`;
        valNode.classList.add("stale");
        rawNode.textContent = "raw —";
        fillNode.style.width = "0%";
        rowNode.classList.remove("unreliable");
        continue;
      }
      const v = out[ch.key];
      valNode.classList.remove("stale");
      valNode.innerHTML = `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}<span class="unit">°</span>`;
      rawNode.textContent = raw ? `raw ${raw[ch.key].toFixed(1)}°` : "raw —";

      const unreliable = ch.key === "shoulder_rotation" && !out.rotation_reliable;
      rowNode.classList.toggle("unreliable", unreliable);

      const [lo, hi] = ch.range;
      const zeroPct = clamp((0 - lo) / (hi - lo), 0, 1) * 100;
      const valPct  = clamp((v - lo) / (hi - lo), 0, 1) * 100;
      zeroNode.style.left = `${zeroPct}%`;
      if (valPct >= zeroPct) {
        fillNode.style.left = `${zeroPct}%`;
        fillNode.style.width = `${valPct - zeroPct}%`;
      } else {
        fillNode.style.left = `${valPct}%`;
        fillNode.style.width = `${zeroPct - valPct}%`;
      }
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
    const wrap = el("div", "chart-wrap");
    const canvas = document.createElement("canvas");
    wrap.appendChild(canvas);
    wrap.appendChild(el("div", "chart-label", ch.label));
    const scale = el("div", "chart-scale", "");
    wrap.appendChild(scale);
    host.appendChild(wrap);
    return { ch, canvas, scale };
  });
  renderLegend();
}

let _legendKey = "";

function renderLegend() {
  const active = S.frame && S.frame.status ? S.frame.status.filter_type : "kalman";
  const key = `${S.allFilters}|${active}`;
  if (key === _legendKey) return;
  _legendKey = key;

  const parts = [`<span><i style="background:${RAW_COLOR}"></i>raw estimate</span>`];
  if (S.allFilters) {
    for (const ft of Object.keys(FILTER_COLORS)) {
      parts.push(`<span><i style="background:${FILTER_COLORS[ft]}"></i>${FILTER_NAMES[ft]}</span>`);
    }
  } else {
    parts.push(`<span><i style="background:${FILTER_COLORS[active] || "#4ea8de"}"></i>${FILTER_NAMES[active] || active} (active → Unity)</span>`);
  }
  $("chartLegend").innerHTML = parts.join("");
}

function renderCharts() {
  const hist = S.history[S.chartSide];
  if (!hist.length) return;
  const now = hist[hist.length - 1].t;
  const t0 = now - TRACE_SECONDS;
  const win = hist.filter((h) => h.t >= t0);
  const active = S.frame && S.frame.status ? S.frame.status.filter_type : "kalman";

  for (const { ch, canvas, scale } of S.charts) {
    const dpr = globalThis.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 400;
    const cssH = 110;
    if (canvas.width !== Math.round(cssW * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    // auto-scale to the data actually present, with a floor so a still arm
    // does not get magnified into apparent violent motion
    let lo = Infinity, hi = -Infinity;
    for (const h of win) {
      for (const s of [h.raw, h.filt]) {
        if (s) { lo = Math.min(lo, s[ch.key]); hi = Math.max(hi, s[ch.key]); }
      }
    }
    if (!isFinite(lo)) { lo = -10; hi = 10; }
    const mid = (lo + hi) / 2, span = Math.max(hi - lo, 12);
    lo = mid - span * 0.62; hi = mid + span * 0.62;
    scale.textContent = `${lo.toFixed(0)}…${hi.toFixed(0)}°`;

    const X = (t) => ((t - t0) / TRACE_SECONDS) * cssW;
    const Y = (v) => cssH - ((v - lo) / (hi - lo)) * cssH;

    // zero line
    if (lo < 0 && hi > 0) {
      ctx.strokeStyle = "#2a3543";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(cssW, Y(0)); ctx.stroke();
    }

    const drawSeries = (get, color, width) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
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

    drawSeries((h) => h.raw, RAW_COLOR, 1);
    if (S.allFilters) {
      for (const ft of Object.keys(FILTER_COLORS)) {
        drawSeries((h) => h.banks[ft], FILTER_COLORS[ft], 1.5);
      }
    } else {
      drawSeries((h) => h.filt, FILTER_COLORS[active] || "#4ea8de", 1.8);
    }
  }
}

// ---------------------------------------------------------------------------
// derivation trace
// ---------------------------------------------------------------------------

function vecStr(v) {
  return v ? `[${v.map((c) => c.toFixed(3)).join(", ")}]` : "—";
}

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
    <div class="trace-step">
      <h4>1 · Keypoints read from this frame</h4>
      <p>Keypoints from the pose network with its own confidence for each
         point. Angles are computed in ${fr.trace.coordinate_space
           ? fr.trace.coordinate_space.label : "pixel space"}.</p>
      <table>
        <thead><tr><th>joint</th><th>index</th><th>visibility</th></tr></thead>
        <tbody>
          <tr><td>shoulder</td><td>${idx.shoulder}</td><td class="${confCls(tr.visibility.shoulder)}">${tr.visibility.shoulder.toFixed(2)}</td></tr>
          <tr><td>elbow</td><td>${idx.elbow}</td><td class="${confCls(tr.visibility.elbow)}">${tr.visibility.elbow.toFixed(2)}</td></tr>
          <tr><td>wrist</td><td>${idx.wrist}</td><td class="${confCls(tr.visibility.wrist)}">${tr.visibility.wrist.toFixed(2)}</td></tr>
        </tbody>
      </table>
    </div>

    <div class="trace-step">
      <h4>2 · Torso reference frame</h4>
      <p>Built from the shoulder and hip keypoints and orthonormalised, so the
         angles do not change when the subject turns relative to the camera.</p>
      <div class="kv">lateral&nbsp;X <b>${vecStr(tf.x_axis_lateral)}</b></div>
      <div class="kv">superior&nbsp;Y <b>${vecStr(tf.y_axis_superior)}</b></div>
      <div class="kv">anterior&nbsp;Z <b>${vecStr(tf.z_axis_anterior)}</b></div>
    </div>

    <div class="trace-step">
      <h4>3 · Segment vectors in that frame</h4>
      <p>Two-link model: upper arm (shoulder→elbow) and forearm (elbow→wrist).</p>
      <div class="kv">upper arm <b>${vecStr(tr.upper_arm_torso)}</b>
        &nbsp;·&nbsp;length ${tr.segment_lengths.upper_arm.toFixed(3)} ${unit}</div>
      <div class="kv">forearm &nbsp;&nbsp;<b>${vecStr(tr.forearm_torso)}</b>
        &nbsp;·&nbsp;length ${tr.segment_lengths.forearm.toFixed(3)} ${unit}</div>
    </div>

    <div class="trace-step">
      <h4>4 · Angles from geometry</h4>
      ${CHANNELS.map((ch) => {
        const doc = angleDoc(ch.key);
        const grey = ch.key === "shoulder_rotation" && a && !a.rotation_reliable;
        return `<div class="kv${grey ? " unreliable" : ""}" style="margin-bottom:3px">
            <code>${doc ? doc.formula : ""}</code>
            → <b class="c-${ch.cls}">${a ? a[ch.key].toFixed(1) : "—"}°</b>
            <span class="muted">${ch.short}${grey ? " · not observable at this elbow angle" : ""}</span>
          </div>`;
      }).join("")}
    </div>

    <div class="trace-step">
      <h4>5 · Filtering, calibration, transmission</h4>
      <table>
        <thead><tr><th>channel</th><th>raw</th><th>filtered</th><th>to Unity</th></tr></thead>
        <tbody>
          ${CHANNELS.map((ch) => `
            <tr><td>${ch.short}</td>
              <td>${a ? a[ch.key].toFixed(2) : "—"}</td>
              <td>${filt ? filt[ch.key].toFixed(2) : "—"}</td>
              <td class="c-${ch.cls}">${cal ? cal[ch.key].toFixed(2) : "—"}</td></tr>`).join("")}
        </tbody>
      </table>
      <p class="muted" style="margin-top:6px">The right-hand column is exactly what
         appears in the UDP packet below.</p>
    </div>`;
}

function confCls(v) { return v >= 0.7 ? "good" : v >= 0.4 ? "warn" : "bad"; }

function angleDoc(key) {
  if (!S.explain) return null;
  return S.explain.angles.find((a) => a.key === key) || null;
}

// ---------------------------------------------------------------------------
// latency
// ---------------------------------------------------------------------------

function renderLatency(fr) {
  const t = fr.timings || {};
  const stages = S.explain ? S.explain.stages : [];
  const host = $("stageBars");
  if (!host.childElementCount && stages.length) {
    for (const st of stages) {
      const b = el("div", "stagebar");
      b.title = st.description;
      b.innerHTML = `<div class="top"><span>${st.label}</span><b id="ms-${st.key}">—</b></div>
                     <div class="track"><div id="bar-${st.key}" style="width:0%"></div></div>`;
      host.appendChild(b);
    }
  }
  const total = Math.max(t.total_ms || 0, 1);
  for (const st of stages) {
    const v = t[st.key] || 0;
    const msNode = $(`ms-${st.key}`), barNode = $(`bar-${st.key}`);
    if (msNode) msNode.textContent = `${v.toFixed(1)} ms`;
    if (barNode) barNode.style.width = `${clamp((v / total) * 100, 0, 100)}%`;
  }

  const m = S.metrics;
  if (!m) return;
  const rows = [
    ["Server pipeline", m.latency.total_ms],
    ["Pose inference", m.latency.pose_ms],
    ["Frame decode", m.latency.decode_ms],
  ];
  const body = $("latencyTable");
  body.innerHTML = rows.map(([name, s]) =>
    `<tr><td>${name}</td><td>${fmt(s.mean, 1)}</td><td>${fmt(s.p95, 1)}</td><td>${fmt(s.max, 1)}</td></tr>`
  ).join("") + (S.lastRtt !== undefined
    ? `<tr><td>Browser round trip</td><td colspan="3">${fmt(S.lastRtt, 1)} ms (latest)</td></tr>`
    : "");

  const pct = (m.within_budget * 100).toFixed(1);
  $("budgetNote").innerHTML =
    `<b>${pct}%</b> of the last ${m.latency.total_ms.n} frames completed within the
     ${m.latency_budget_ms} ms budget; measured throughput ${fmt(m.fps, 1)} fps
     against a ≥ 20 fps requirement. Uptime ${fmt(m.uptime_s, 0)} s over
     ${m.frames} frames, pose detected in ${(m.detection_rate * 100).toFixed(1)}%.`;
}

// ---------------------------------------------------------------------------
// filter comparison
// ---------------------------------------------------------------------------

function renderFilterTable() {
  const m = S.metrics;
  if (!m) return;
  const side = S.chartSide;
  $("cmpSide").textContent = side;
  const active = S.frame && S.frame.status ? S.frame.status.filter_type : "kalman";
  const order = [["raw", "Raw (unfiltered)"], ["kalman", "Kalman"],
                 ["ma", "Moving average"], ["sg", "Savitzky–Golay"]];
  const body = $("filterTable");
  body.innerHTML = order.map(([key, label]) => {
    const s = m.angle_std[`${key}:${side}`];
    const cells = CHANNELS.map((ch) => {
      if (!s) return "<td>—</td>";
      const v = s[ch.key];
      const cls = key === "raw" ? "" : (v <= 3 ? "good" : v <= 5 ? "warn" : "bad");
      return `<td class="${cls}">${v.toFixed(2)}</td>`;
    }).join("");
    return `<tr class="${key === active ? "active" : ""}"><td>${label}${key === active ? " ←" : ""}</td>${cells}</tr>`;
  }).join("");

  const kp = m.keypoint_jitter;
  $("kpJitter").textContent = kp ? kp.mean_rms_px.toFixed(2) : "—";
}

// ---------------------------------------------------------------------------
// Unity wire panel
// ---------------------------------------------------------------------------

let lastPacketCount = 0, lastPacketTime = 0;

function renderWire(fr) {
  const udp = fr.status && fr.status.udp;
  if (!udp) return;
  const wire = $("wirePacket");
  if (udp.last_packet) {
    wire.className = "wire";
    wire.textContent = udp.last_packet;
  } else {
    wire.className = "wire idle";
    wire.textContent = udp.enabled ? "waiting for the first transmitted packet…"
                                   : "Unity stream disabled";
  }
  setPill($("pillPackets"), `<b>${udp.packets_sent}</b> packets`);
  const now = performance.now();
  if (lastPacketTime && now > lastPacketTime) {
    const rate = (udp.packets_sent - lastPacketCount) / ((now - lastPacketTime) / 1000);
    if (rate >= 0) setPill($("pillPacketRate"), `<b>${rate.toFixed(1)}</b> pkt/s`,
                           rate >= 20 ? "ok" : "warn");
  }
  lastPacketCount = udp.packets_sent;
  lastPacketTime = now;
  setPill($("pillPacketErr"), `<b>${udp.send_errors}</b> errors`,
          udp.send_errors ? "bad" : "");

  if (udp.history && udp.history.length) {
    $("wireHistory").innerHTML = udp.history.slice().reverse()
      .map((h) => `<div>${new Date(h.t * 1000).toLocaleTimeString()}  ${esc(h.packet)}</div>`)
      .join("");
  }
}

// ---------------------------------------------------------------------------
// control synchronisation
// ---------------------------------------------------------------------------

function syncControls(fr) {
  const st = fr.status || {};
  if (st.filter_type && $("selFilter").value !== st.filter_type) {
    $("selFilter").value = st.filter_type;
    renderLegend();
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
    $("inUdpHz").value = st.udp.target_hz || st.config.stream_hz;
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

const POSE_LABELS = {
  arm_down:    "Arm hanging straight down at your side",
  arm_forward: "Arm straight out in front, horizontal",
  arm_side:    "Arm straight out to the side, horizontal",
  elbow_bent:  "Upper arm down, elbow bent to about 90°",
};

function renderCalibration(cal) {
  S.calibration = cal;
  const side = $("selCalSide").value;

  $("calSteps").innerHTML = (cal.required_poses || []).map((p) => {
    const done = (cal.captured || []).includes(p);
    const cur = cal.active && cal.pose === p;
    return `<span class="step ${done ? "done" : cur ? "current" : ""}">${p.replace("_", " ")}</span>`;
  }).join("");

  const notice = $("calNotice");
  if (cal.active) {
    notice.className = "notice";
    notice.innerHTML = `<b>${POSE_LABELS[cal.pose] || cal.pose}</b><br>
      <span class="muted">Hold the pose steadily, then press Capture. The last
      15 filtered frames are averaged, so a single noisy frame cannot skew it.</span>`;
  } else if (cal.calibrated && cal.calibrated[side]) {
    const warn = (cal.warnings && cal.warnings[side]) || [];
    if (warn.length) {
      notice.className = "notice err";
      notice.innerHTML = `<b>${side} arm calibrated with problems.</b><br>` +
        warn.map((w) => `• ${esc(w)}`).join("<br>");
    } else {
      notice.className = "notice ok";
      notice.textContent = `${side} arm calibrated — offsets and scales are applied to every frame.`;
    }
  } else {
    notice.className = "notice";
    notice.textContent = "Not calibrated. Raw geometric estimates are being sent to Unity.";
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

async function refreshSessions() {
  const data = await api("/api/sessions");
  const body = $("sessionTable");
  if (!data.sessions.length) {
    body.innerHTML = `<tr><td colspan="3" class="muted">No sessions recorded yet.</td></tr>`;
    return;
  }
  body.innerHTML = data.sessions.map((s) => `
    <tr><td>${esc(s.name)}</td><td>${(s.size_bytes / 1024).toFixed(1)} kB</td>
      <td style="text-align:right;white-space:nowrap">
        <button class="small" data-summary="${esc(s.name)}">inspect</button>
        <a href="/api/sessions/${encodeURIComponent(s.name)}" download>csv</a>
      </td></tr>`).join("");

  body.querySelectorAll("[data-summary]").forEach((b) => {
    b.onclick = () => showSession(b.dataset.summary);
  });
}

async function showSession(name) {
  const host = $("sessionDetail");
  host.innerHTML = `<p class="muted">Computing statistics from ${name}…</p>`;
  try {
    const s = await api(`/api/sessions/${encodeURIComponent(name)}/summary`);
    const rows = CHANNELS.map((ch) => {
      const r = s.channels[`right_${ch.key}`] || { n: 0 };
      const l = s.channels[`left_${ch.key}`] || { n: 0 };
      return `<tr><td>${ch.short}</td>
        <td>${r.n ? r.mean.toFixed(1) : "—"}</td><td>${r.n ? r.std.toFixed(2) : "—"}</td>
        <td>${l.n ? l.mean.toFixed(1) : "—"}</td><td>${l.n ? l.std.toFixed(2) : "—"}</td></tr>`;
    }).join("");
    host.innerHTML = `
      <p class="muted" style="margin:10px 0 4px">
        <b>${esc(s.name)}</b> — ${s.rows} frames over ${s.duration_s} s
        (${s.mean_rate_hz} Hz mean). Right arm tracked in
        ${(s.tracked_fraction.right * 100).toFixed(1)}% of frames, left in
        ${(s.tracked_fraction.left * 100).toFixed(1)}%.</p>
      <table>
        <thead><tr><th>channel</th><th>R mean</th><th>R σ</th><th>L mean</th><th>L σ</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <img class="plot" src="/api/sessions/${encodeURIComponent(name)}/plot.png?side=right"
           alt="joint angle time series">`;
  } catch (err) {
    host.innerHTML = `<p class="notice err">${esc(err.message)}</p>`;
  }
}

// ---------------------------------------------------------------------------
// explanation panel
// ---------------------------------------------------------------------------

async function loadExplain() {
  S.explain = await api("/api/explain");

  $("dataFlow").innerHTML = S.explain.data_flow
    .map((s, i) => `${i ? '<i>→</i>' : ''}<span>${s}</span>`).join("");

  $("explainAngles").innerHTML = S.explain.angles.map((a) => `
    <div class="explain-item">
      <h4>${a.label}</h4>
      <p>${a.description}</p>
      <div class="explain-meta">${a.formula} · ${a.sign} · neutral: ${a.neutral}
        · typical ${a.typical_range[0]}…${a.typical_range[1]}°</div>
    </div>`).join("");

  $("explainFilters").innerHTML = S.explain.filters.map((f) => `
    <div class="explain-item">
      <h4>${f.label}</h4>
      <p>${f.description}</p>
      <div class="explain-meta">${f.parameters} — ${f.tradeoff}</div>
    </div>`).join("");

  $("explainReq").innerHTML = S.explain.requirements.map((r) => `
    <tr><td>${r.key.replace("_", " ")}</td><td>${r.target}</td><td>${r.measured_by}</td></tr>`
  ).join("");

  const p = S.explain.protocol;
  $("protocolNote").innerHTML =
    `${p.transport}, ${p.encoding}, ${p.rate}. Formats: ` +
    p.packets.map((k) => `<code class="mono">${esc(k.format)}</code>`).join(" · ") +
    `. ${p.hold_behaviour} Unity receiver: <span class="mono">${p.unity_receiver}</span>.`;
}

// ---------------------------------------------------------------------------
// source selection
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
      : `<option value="">no videos in ${data.video_dir}</option>`;
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
    $("sourceHint").textContent = "browser webcam";
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
    $("sourceHint").textContent = mode === "camera"
      ? `server camera ${st.camera_index}` : `replay: ${st.path.split(/[\\/]/).pop()}`;
  } catch (err) {
    showPlaceholder(err.message);
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

  $("selRate").onchange = (e) => { S.sendInterval = 1000 / Number(e.target.value); };
  $("chkMirror").onchange = (e) => api("/api/mirror", "POST", { enabled: e.target.checked });

  $("selFilter").onchange = async (e) => {
    await api("/api/filter", "POST", { type: e.target.value });
    renderLegend();
  };
  $("btnResetFilters").onclick = () => api("/api/reset", "POST");

  $("selChartSide").onchange = (e) => { S.chartSide = e.target.value; renderCharts(); };
  $("selTraceSide").onchange = (e) => {
    S.traceSide = e.target.value;
    if (S.frame) renderTrace(S.frame);
  };
  $("chkAllFilters").onchange = (e) => {
    S.allFilters = e.target.checked;
    renderLegend();
    renderCharts();
  };

  $("btnUdpApply").onclick = async () => {
    const st = await api("/api/udp", "POST", {
      enabled: $("chkUdp").checked,
      host: $("inUdpHost").value.trim(),
      port: Number($("inUdpPort").value),
      hz: Number($("inUdpHz").value),
    });
    onStatus({ udp: st, config: { stream_hz: st.target_hz } });
  };
  $("chkUdp").onchange = () => $("btnUdpApply").click();

  $("btnCalBegin").onclick = async () => {
    renderCalibration(await api("/api/calibration/begin", "POST",
                                { side: $("selCalSide").value }));
  };
  $("btnCalCapture").onclick = async () => {
    try {
      renderCalibration(await api("/api/calibration/capture", "POST"));
    } catch (err) {
      const n = $("calNotice"); n.className = "notice err"; n.textContent = err.message;
    }
  };
  $("btnCalCancel").onclick = async () => {
    renderCalibration(await api("/api/calibration/cancel", "POST"));
  };
  $("btnCalClear").onclick = async () => {
    renderCalibration(await api("/api/calibration/clear", "POST",
                                { side: $("selCalSide").value }));
  };
  $("selCalSide").onchange = () => { if (S.calibration) renderCalibration(S.calibration); };

  $("btnLogStart").onclick = async () => {
    const st = await api("/api/logging/start", "POST", { label: $("inLogLabel").value.trim() });
    onStatus({ logging: st });
  };
  $("btnLogStop").onclick = async () => {
    const st = await api("/api/logging/stop", "POST");
    onStatus({ logging: st });
    refreshSessions();
  };

  $("selSource").onchange = async () => { await refreshSources(); applySource(); };
  $("selSourceItem").onchange = applySource;

  window.addEventListener("resize", renderCharts);
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

(async function boot() {
  wire();
  buildCharts();
  try { await loadExplain(); } catch (e) { console.warn("explain:", e); }
  try { onStatus(await api("/api/status")); } catch (e) { console.warn("status:", e); }
  try { await refreshSessions(); } catch (e) { console.warn("sessions:", e); }
  try { await refreshSources(); } catch (e) { console.warn("sources:", e); }
  connect();
  setInterval(() => {
    if (S.ws && S.wsReady) S.ws.send(JSON.stringify({ type: "status" }));
  }, 5000);
})();
