const counters = { replay: 0, yolo: 0, audio: 0, ce: 0 };
const maxLogEntries = 50;
let logEntries = [];
let dataRoots = [
  "/media/brianw/Extreme SSD/West Point Experimentation",
  "/media/brianw/Extreme SSD/GQ Data",
];

let replaySession = {
  scenario: null,
  start: 0,
  end: -1,
  syncStartAt: null,
  commandWallTime: null,
  playbackMode: "max",
  speed: 1.0,
};
let replayProgress = {};

const $ = (id) => document.getElementById(id);

function asPretty(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function topicKind(topic) {
  if (topic.startsWith("/replay/")) return "replay";
  if (topic.includes("/analytics/yolo/bbox") || topic.includes("/analytics/yolo/annotated/compressed")) return "yolo";
  if (topic.includes("/audio_detector/detections") || topic.includes("/audio_detector/status")) return "audio";
  if (topic.startsWith("/complex_events/")) return "ce";
  return "other";
}

function updateBadge(connected) {
  const badge = $("mqttBadge");
  badge.textContent = connected ? "MQTT: connected" : "MQTT: disconnected";
  badge.classList.toggle("ok", connected);
  badge.classList.toggle("bad", !connected);
}

function updateCounters() {
  $("replayCount").textContent = counters.replay;
  $("yoloCount").textContent = counters.yolo;
  $("audioCount").textContent = counters.audio;
  $("ceCount").textContent = counters.ce;
}

function scenarioDate(scenario) {
  return String(scenario || "").trim().split("_")[0];
}

function updateCandidateFolders() {
  const scenario = $("scenario").value.trim();
  const date = scenarioDate(scenario);
  const out = $("candidateFolders");
  if (!date) {
    out.textContent = "Enter a scenario ID to preview candidate folders.";
    return;
  }
  const rows = dataRoots.map((root) => `${root}/${date}/`);
  out.textContent = rows.join("\n") + "\n\nThese parent roots are mounted into the persistent replay containers; supervisors choose the matching scenario/date at replay time.";
}

function replaySourceFromTopic(topic) {
  let m = topic.match(/^\/replay\/status\/zed\/(.+)$/);
  if (m) return `zed:${m[1]}`;
  m = topic.match(/^\/replay\/status\/respeaker\/(.+)$/);
  if (m) return `respeaker:${m[1]}`;
  if (topic === "/replay/status/gps") return "gps";
  return null;
}

function resetProgressView() {
  replayProgress = {};
  updateReplayProgressUI();
}

function maybeUpdateReplaySession(item) {
  if (!item || !item.topic) return;
  const payload = item.payload || {};
  if (item.topic === "/replay/config" && typeof payload === "object") {
    replaySession.scenario = payload.scenario || replaySession.scenario;
    replaySession.start = Number(payload.start_time ?? payload.start ?? replaySession.start ?? 0);
    replaySession.end = Number(payload.end_time ?? payload.end ?? replaySession.end ?? -1);
    replaySession.playbackMode = payload.playback_mode || payload.mode || replaySession.playbackMode || "max";
    replaySession.speed = Number(payload.speed ?? payload.playback_speed ?? replaySession.speed ?? 1.0);
    replaySession.commandWallTime = item.ts;
    resetProgressView();
  }
  if (item.topic === "/replay/sync" && typeof payload === "object") {
    replaySession.scenario = payload.scenario || replaySession.scenario;
    replaySession.syncStartAt = Number(payload.start_at || payload.sync_start_at || payload.t || item.ts);
    replaySession.playbackMode = payload.playback_mode || payload.mode || replaySession.playbackMode || "max";
    replaySession.speed = Number(payload.speed ?? payload.playback_speed ?? replaySession.speed ?? 1.0);
    replaySession.commandWallTime = item.ts;
  }
}

function maybeUpdateReplayProgress(item, opts = {}) {
  const source = replaySourceFromTopic(item.topic);
  if (!source || !item.payload || typeof item.payload !== "object") return;
  const p = item.payload;
  if (!Number.isFinite(Number(p.current)) || !Number.isFinite(Number(p.duration))) return;

  // Use client receipt time for freshness. Server timestamps from the initial
  // /api/state snapshot can be old and made active rows look stale even when
  // live progress messages were arriving.
  const now = opts.live ? Date.now() / 1000 : Number(item.ts || Date.now() / 1000);
  const current = Number(p.current);
  const duration = Math.max(Number(p.duration), 0.000001);
  const previous = replayProgress[source];
  let speed = Number(p.speed_x ?? p.effective_speed_x ?? NaN);

  if (!Number.isFinite(speed) && previous && Number.isFinite(previous.current) && Number.isFinite(previous.ts)) {
    const dc = current - previous.current;
    const dt = now - previous.ts;
    if (dt > 0.2 && dc >= 0) speed = dc / dt;
  }
  if (!Number.isFinite(speed) && replaySession.syncStartAt) {
    const wallElapsed = now - replaySession.syncStartAt;
    if (wallElapsed > 0.2 && current >= 0) speed = current / wallElapsed;
  }
  if (!Number.isFinite(speed) || speed < 0) speed = 0;

  const remaining = Math.max(0, duration - current);
  const etaSeconds = speed > 0.0001 ? remaining / speed : null;

  replayProgress[source] = {
    source,
    service: p.service || source.split(":")[0],
    node: p.node || source,
    current,
    duration,
    pct: Number.isFinite(Number(p.pct)) ? Number(p.pct) : (100 * current / duration),
    speed,
    etaSeconds,
    finishAt: etaSeconds == null ? null : now + etaSeconds,
    ts: now,
    raw: p,
  };
  updateReplayProgressUI();
}

function formatSeconds(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const sign = seconds < 0 ? "-" : "";
  seconds = Math.abs(seconds);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${sign}${h}h ${m}m ${s}s`;
  if (m > 0) return `${sign}${m}m ${s}s`;
  return `${sign}${s}s`;
}

function formatClock(ts) {
  if (ts == null || !Number.isFinite(ts)) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
}

function updateReplayProgressUI() {
  const now = Date.now() / 1000;
  const rows = Object.values(replayProgress)
    .sort((a, b) => a.source.localeCompare(b.source));
  const active = rows.filter((r) => now - r.ts < 20);

  const tbody = $("replayProgressBody");
  tbody.innerHTML = "";
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">No replay progress yet.</td></tr>';
    $("progressSpeed").textContent = "—";
    $("progressEta").textContent = "—";
    $("progressTime").textContent = "—";
    $("progressSources").textContent = "0";
    if ($("progressMode")) $("progressMode").textContent = replaySession.playbackMode === "scaled" ? `${replaySession.speed}×` : replaySession.playbackMode;
    $("progressNote").textContent = replaySession.scenario
      ? `Waiting for replay status for ${replaySession.scenario}. Status usually appears about every 1 second after playback starts.`
      : "No replay status received yet. Start replay, then wait for a replay container to publish status.";
    return;
  }

  for (const r of rows) {
    const stale = now - r.ts > 20;
    const tr = document.createElement("tr");
    tr.className = stale ? "stale" : "";
    const progressText = `${formatSeconds(r.current)} / ${formatSeconds(r.duration)} (${Math.max(0, Math.min(100, r.pct)).toFixed(1)}%)`;
    const speedText = r.speed > 0 ? `${r.speed.toFixed(2)}×` : "—";
    const etaText = r.etaSeconds == null ? "—" : `${formatSeconds(r.etaSeconds)} (${formatClock(r.finishAt)})`;
    const ageText = `${formatSeconds(now - r.ts)} ago`;
    tr.innerHTML = `
      <td><strong>${escapeHtml(r.source)}</strong><br><span class="muted">${escapeHtml(r.node || "")}</span></td>
      <td>${escapeHtml(progressText)}<div class="bar"><div style="width:${Math.max(0, Math.min(100, r.pct))}%"></div></div></td>
      <td>${escapeHtml(speedText)}</td>
      <td>${escapeHtml(etaText)}</td>
      <td>${escapeHtml(ageText)}${stale ? '<br><span class="warn">stale</span>' : ""}</td>
    `;
    tbody.appendChild(tr);
  }

  const considered = active.length ? active : rows;
  const speeds = considered.map((r) => r.speed).filter((v) => Number.isFinite(v) && v > 0);
  const avgSpeed = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null;
  const maxEta = Math.max(...considered.map((r) => r.etaSeconds ?? 0));
  const maxFinish = Math.max(...considered.map((r) => r.finishAt ?? 0));
  const maxDuration = Math.max(...considered.map((r) => r.duration));
  const maxCurrent = Math.max(...considered.map((r) => r.current));

  $("progressSpeed").textContent = avgSpeed == null ? "—" : `${avgSpeed.toFixed(2)}×`;
  $("progressEta").textContent = maxEta > 0 ? `${formatSeconds(maxEta)} (${formatClock(maxFinish)})` : "done/idle";
  $("progressTime").textContent = `${formatSeconds(maxCurrent)} / ${formatSeconds(maxDuration)}`;
  $("progressSources").textContent = String(active.length || rows.length);
  if ($("progressMode")) $("progressMode").textContent = replaySession.playbackMode === "scaled" ? `${replaySession.speed}×` : replaySession.playbackMode;

  const scenario = replaySession.scenario ? ` for ${replaySession.scenario}` : "";
  const requested = replaySession.playbackMode === "scaled"
    ? `${Number(replaySession.speed || 1).toFixed(2)}× requested`
    : replaySession.playbackMode === "realtime"
      ? "1.00× real-time requested"
      : "max-speed requested";
  const speedNote = avgSpeed == null
    ? "Waiting for enough status updates to estimate effective speed."
    : replaySession.playbackMode === "max"
      ? `Effective max-mode speed is ${avgSpeed.toFixed(2)}×. This is bounded by I/O, decoding, GPU, and detector load.`
      : avgSpeed < 0.85 * Number(replaySession.speed || 1)
        ? "Replay appears slower than the requested timing, likely because processing or I/O is falling behind."
        : "Replay is close to the requested timing.";
  $("progressNote").textContent = `${considered.length} source(s) reporting${scenario}. ${requested}. ${speedNote}`;
}

function addLog(item, opts = {}) {
  maybeUpdateReplaySession(item);
  maybeUpdateReplayProgress(item, opts);

  const kind = topicKind(item.topic);
  if (kind in counters && item.direction === "in") counters[kind] += 1;
  updateCounters();

  if (kind === "replay") $("latestReplay").textContent = asPretty(item.payload);
  if (kind === "ce") $("latestCE").textContent = asPretty(item.payload);

  logEntries.push(item);
  if (logEntries.length > maxLogEntries) logEntries.shift();
  renderLog();
}

function renderLog() {
  const filter = $("filter").value.trim().toLowerCase();
  const container = $("log");
  const shouldStick = container.scrollTop + container.clientHeight >= container.scrollHeight - 12;
  container.innerHTML = "";

  for (const item of logEntries) {
    if (filter && !item.topic.toLowerCase().includes(filter)) continue;
    const entry = document.createElement("div");
    entry.className = `log-entry ${item.direction}`;
    const ts = new Date(item.ts * 1000).toLocaleTimeString();
    entry.innerHTML = `<span class="timestamp">${ts}</span> <span class="topic">${item.direction.toUpperCase()} ${item.topic}</span>\n${escapeHtml(asPretty(item.payload))}`;
    container.appendChild(entry);
  }
  if (shouldStick) container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function postJson(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(payload.detail || `HTTP ${resp.status}`);
  return payload;
}

async function loadInitialState() {
  const resp = await fetch("/api/state");
  const state = await resp.json();
  updateBadge(state.mqtt.connected);
  dataRoots = state.data_roots || dataRoots;
  updateCandidateFolders();
  for (const item of state.recent || []) addLog(item, { initial: true });
  updateReplayProgressUI();
}

function connectEvents() {
  const source = new EventSource("/api/events");
  source.addEventListener("mqtt", (event) => {
    const item = JSON.parse(event.data);
    addLog(item, { live: true });
    if (item.topic === "$web_ui/status" && item.payload && "connected" in item.payload) {
      updateBadge(Boolean(item.payload.connected));
    }
  });
  source.addEventListener("heartbeat", (event) => {
    const status = JSON.parse(event.data);
    updateBadge(Boolean(status.connected));
    updateReplayProgressUI();
  });
  source.onerror = () => updateBadge(false);
}

$("replayForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const playbackMode = $("playbackMode").value || "max";
  const body = {
    scenario: $("scenario").value.trim(),
    start: Number($("start").value || 0),
    end: Number($("end").value || -1),
    sync_delay: Number($("syncDelay").value || 1),
    playback_mode: playbackMode,
    send_control: $("sendControl").checked,
  };
  // Only send a custom multiplier when the user explicitly selected Custom speed.
  // Max-speed and realtime modes should not require or validate this field.
  if (playbackMode === "scaled") {
    body.speed = Number($("speed").value || 1);
  }
  try {
    resetProgressView();
    const res = await postJson("/api/replay/start", body);
    $("lastAction").textContent = res.message;
  } catch (err) {
    $("lastAction").textContent = `Error: ${err.message}`;
  }
});


$("resendSyncBtn").addEventListener("click", async () => {
  try {
    const res = await postJson("/api/replay/resend-sync?delay=0.5", {});
    $("lastAction").textContent = res.message || "Resent replay sync.";
  } catch (err) {
    $("lastAction").textContent = `Error: ${err.message}`;
  }
});

$("clearReplayBtn").addEventListener("click", async () => {
  try {
    const res = await postJson("/api/replay/clear", {});
    resetProgressView();
    $("lastAction").textContent = res.message || "Cleared retained replay command.";
  } catch (err) {
    $("lastAction").textContent = `Error: ${err.message}`;
  }
});

$("stopBtn").addEventListener("click", async () => {
  try {
    const res = await postJson("/api/control", { action: "collection-stop" });
    $("lastAction").textContent = res.note || "Published collection-stop.";
  } catch (err) {
    $("lastAction").textContent = `Error: ${err.message}`;
  }
});

$("shutdownBtn").addEventListener("click", async () => {
  try {
    const res = await postJson("/api/control", { action: "collection-shutdown" });
    $("lastAction").textContent = res.note || "Published collection-shutdown. Recreate containers before replaying again.";
  } catch (err) {
    $("lastAction").textContent = `Error: ${err.message}`;
  }
});

$("publishForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const topic = $("pubTopic").value.trim();
  let payload = $("pubPayload").value;
  try { payload = JSON.parse(payload); } catch (_) { /* keep as string */ }
  try {
    await postJson("/api/publish", { topic, payload });
    $("pubPayload").value = "";
  } catch (err) {
    alert(`Publish failed: ${err.message}`);
  }
});


function updateSpeedControlVisibility() {
  const mode = $("playbackMode") ? $("playbackMode").value : "max";
  const label = $("speedLabel");
  const speedInput = $("speed");
  if (!label || !speedInput) return;

  const custom = mode === "scaled";
  label.classList.toggle("hidden", !custom);
  label.classList.toggle("dimmed", !custom);
  speedInput.disabled = !custom;
  speedInput.required = custom;

  if (mode === "realtime") speedInput.value = "1.0";
}

$("scenario").addEventListener("input", updateCandidateFolders);
$("filter").addEventListener("input", renderLog);
$("clearBtn").addEventListener("click", () => {
  logEntries = [];
  renderLog();
});
$("resetProgressBtn").addEventListener("click", resetProgressView);
if ($("playbackMode")) $("playbackMode").addEventListener("change", updateSpeedControlVisibility);
updateSpeedControlVisibility();

loadInitialState().then(connectEvents).catch((err) => {
  console.error(err);
  connectEvents();
});
