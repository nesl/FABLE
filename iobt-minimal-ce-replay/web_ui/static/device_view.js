const $ = (id) => document.getElementById(id);
const maxLogEntries = 160;
let selectedDevice = "orin11";
let knownDevices = new Set(["orin11"]);
let selectedLogs = [];
let lastYoloImageTs = null;
let lastAudioTs = null;
let lastYoloStatusTs = null;
let lastReplayConfig = null;
let lastReplaySync = null;
let latestYoloStatus = null;
let latestYoloFrame = null;
let lastYoloFrameTs = null;
let latestAudioStatus = null;
let lastSelectedMessageTs = null;
let evidence = resetEvidence();

function resetEvidence() {
  return {
    yoloStatus: 0,
    yoloFrame: 0,
    yoloBbox: 0,
    yoloImage: 0,
    audioStatus: 0,
    audioDetection: 0,
    replayStatus: 0,
    other: 0,
  };
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pretty(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function norm(s) {
  return String(s || "").trim().toLowerCase();
}

function updateBadge(connected) {
  const badge = $("mqttBadge");
  badge.textContent = connected ? "MQTT: connected" : "MQTT: disconnected";
  badge.classList.toggle("ok", connected);
  badge.classList.toggle("bad", !connected);
}

function nodeAliases(name) {
  const raw = String(name || "").trim();
  if (!raw) return new Set();
  const out = new Set([norm(raw), norm(raw.replaceAll("-", "_")), norm(raw.replaceAll("_", "-"))]);
  const m = raw.match(/(\d+)$/);
  if (m) {
    const n = m[1];
    for (const alias of [`orin${n}`, `node${n}`, `dvpg_gq_orin_${n}`, `dvpg-gq-orin-${n}`, `orin_${n}`]) {
      out.add(norm(alias));
    }
  }
  return out;
}

function topicNode(topic) {
  const parts = String(topic || "").split("/").filter(Boolean);
  return parts.length > 0 ? parts[0] : "";
}

function payloadNode(payload) {
  if (!payload || typeof payload !== "object") return "";
  return payload.source_host || payload.hostname || payload.node || "";
}

function itemMatchesSelected(item) {
  const aliases = nodeAliases(selectedDevice);
  const topic = norm(topicNode(item.topic));
  if (aliases.has(topic)) return true;
  const pnode = norm(payloadNode(item.payload));
  if (aliases.has(pnode)) return true;
  if (Array.isArray(item.payload) && item.payload.some((d) => aliases.has(norm(payloadNode(d))))) return true;
  return false;
}

function discoverDevice(item) {
  const candidates = [topicNode(item.topic), payloadNode(item.payload)];
  if (Array.isArray(item.payload)) {
    for (const det of item.payload) candidates.push(payloadNode(det));
  }
  for (const c of candidates) {
    const m = String(c || "").match(/(\d+)$/);
    if (m) knownDevices.add(`orin${m[1]}`);
  }
  renderDeviceSelect();
}

function renderDeviceSelect() {
  const select = $("deviceSelect");
  const current = select.value || selectedDevice;
  const items = Array.from(knownDevices).sort((a, b) => {
    const na = Number((a.match(/\d+$/) || [0])[0]);
    const nb = Number((b.match(/\d+$/) || [0])[0]);
    return na - nb || a.localeCompare(b);
  });
  select.innerHTML = "";
  for (const dev of items) {
    const opt = document.createElement("option");
    opt.value = dev;
    opt.textContent = dev;
    select.appendChild(opt);
  }
  if (items.includes(current)) select.value = current;
  else select.value = selectedDevice;
}

function setSelectedDevice(dev) {
  selectedDevice = String(dev || "orin11").trim();
  if (!selectedDevice) selectedDevice = "orin11";
  knownDevices.add(selectedDevice);
  renderDeviceSelect();
  $("deviceSelect").value = selectedDevice;
  $("manualDevice").value = selectedDevice;
  $("deviceStatus").textContent = `Showing messages matching ${Array.from(nodeAliases(selectedDevice)).join(", ")}`;
  selectedLogs = [];
  evidence = resetEvidence();
  latestYoloStatus = null;
  latestYoloFrame = null;
  lastYoloFrameTs = null;
  latestAudioStatus = null;
  lastYoloImageTs = null;
  lastYoloStatusTs = null;
  lastAudioTs = null;
  lastSelectedMessageTs = null;
  $("yoloJson").textContent = "No YOLO bbox message yet.";
  $("yoloStatusJson").textContent = "No YOLO status message yet.";
  $("yoloFrameJson").textContent = "No YOLO frame-probe message yet.";
  $("audioJson").textContent = "No audio message yet.";
  $("yoloImage").style.display = "none";
  $("imagePlaceholder").style.display = "block";
  renderLog();
  renderHealth();
}

function fmtAge(ts) {
  if (!ts) return null;
  const now = Date.now() / 1000;
  return `${Math.max(0, now - ts).toFixed(1)}s ago`;
}

function updateAges() {
  $("yoloAge").textContent = lastYoloImageTs ? `${fmtAge(lastYoloImageTs)}` : "no image yet";
  $("audioAge").textContent = lastAudioTs ? `${fmtAge(lastAudioTs)}` : "no audio yet";
  renderHealth();
}

function diagnosisText(diag) {
  const map = {
    waiting_for_zed_ipc_socket: "No ZED IPC socket yet",
    waiting_for_video_frames: "YOLO is alive; no video frames received",
    video_frames_stale: "Video frames arrived before, but are now stale",
    video_active_no_yolo_detections: "Video is active; YOLO sees no matching objects",
    active_detections: "YOLO detections active",
    model_loading: "YOLO model is loading",
    model_not_loaded: "YOLO model did not load",
    softfail_enabled: "YOLO soft-fail enabled",
    waiting_for_respeaker_ipc_socket: "No ReSpeaker IPC socket yet",
    waiting_for_audio_frames: "Audio detector is alive; no audio frames received",
    audio_frames_stale: "Audio frames arrived before, but are now stale",
    active_audio_detection: "Audio detections active",
    audio_active_below_threshold: "Audio is active; below detection threshold",
  };
  return map[diag] || diag || "unknown";
}

function renderReplayState() {
  const state = $("replayState");
  const detail = $("replayDetail");
  if (!lastReplayConfig && !lastReplaySync) {
    state.textContent = "No replay request seen";
    detail.textContent = "Click Start Replay on the Replay UI, or start replay from the CLI. This panel now keeps replay state separately from the rolling MQTT log.";
    return;
  }
  const scenario = (lastReplayConfig && lastReplayConfig.scenario) || (lastReplaySync && lastReplaySync.scenario) || "unknown";
  state.textContent = `Requested ${scenario}`;
  if (lastReplaySync && lastReplaySync.start_at) {
    const delta = Number(lastReplaySync.start_at) - Date.now() / 1000;
    detail.textContent = delta > 0 ? `Replay sync starts in ${delta.toFixed(1)}s.` : `Replay sync was ${Math.abs(delta).toFixed(1)}s ago.`;
  } else {
    detail.textContent = "Config seen; waiting for /replay/sync.";
  }
}

function renderEvidence() {
  const h = $("evidenceHealth");
  const d = $("evidenceDetail");
  if (!h || !d) return;
  if (!lastSelectedMessageTs) {
    h.textContent = "No selected-node messages";
    d.textContent = "This can mean the selected node is not running, the compose was generated for a different device, or no detector messages have arrived yet.";
    return;
  }
  h.textContent = `Last selected-node message ${fmtAge(lastSelectedMessageTs)}`;
  d.textContent = `YOLO debug status=${evidence.yoloStatus}, frame probes=${evidence.yoloFrame}, bbox=${evidence.yoloBbox}, debug images=${evidence.yoloImage}, audio detections=${evidence.audioDetection}, audio debug status=${evidence.audioStatus}, other=${evidence.other}`;
}

function renderHealth() {
  renderReplayState();
  renderEvidence();

  const yh = $("yoloHealth");
  const yhd = $("yoloHealthDetail");
  if (!latestYoloStatus) {
    yh.textContent = "No YOLO status yet";
    yhd.textContent = "No YOLO bbox or debug-status message has arrived for this selected device. Normal runs do not publish synthetic status; use --yolo-debug-status if you need to distinguish no video from no objects.";
  } else {
    const diag = latestYoloStatus.diagnosis;
    yh.textContent = diagnosisText(diag);
    const pieces = [];
    pieces.push(`frames=${latestYoloStatus.input_frames_total ?? 0}`);
    pieces.push(`last_dets=${latestYoloStatus.last_detections ?? 0}`);
    pieces.push(`model_loaded=${Boolean(latestYoloStatus.model_loaded)}`);
    if (latestYoloStatus.yolo_device !== undefined) pieces.push(`device=${latestYoloStatus.yolo_device}`);
    if (latestYoloStatus.local_ipc_exists !== null && latestYoloStatus.local_ipc_exists !== undefined) pieces.push(`ipc=${latestYoloStatus.local_ipc_exists ? "yes" : "no"}`);
    if (latestYoloStatus.last_frame_age_sec !== null && latestYoloStatus.last_frame_age_sec !== undefined) pieces.push(`last_frame=${latestYoloStatus.last_frame_age_sec}s ago`);
    if (latestYoloStatus.last_frame_shape) pieces.push(`shape=${latestYoloStatus.last_frame_shape}`);
    if (latestYoloStatus.last_error) pieces.push(`error=${latestYoloStatus.last_error}`);
    if (latestYoloStatus.last_publish_error) pieces.push(`publish_error=${latestYoloStatus.last_publish_error}`);
    yhd.textContent = pieces.join(" · ");
  }

  const yf = $("yoloFrameHealth");
  const yfd = $("yoloFrameDetail");
  if (yf && yfd) {
    if (!latestYoloFrame) {
      yf.textContent = "No frame probe yet";
      yfd.textContent = "No /debug/<node>/analytics/yolo/frame message for this device. Use --yolo-frame-debug or --yolo-debug-status, then start replay.";
    } else {
      const age = fmtAge(lastYoloFrameTs);
      yf.textContent = `Frames received (${age})`;
      const pieces = [];
      pieces.push(`frames=${latestYoloFrame.input_frames_total ?? 0}`);
      if (latestYoloFrame.last_frame_shape) pieces.push(`shape=${latestYoloFrame.last_frame_shape}`);
      if (latestYoloFrame.last_rgb_bytes !== undefined && latestYoloFrame.last_rgb_bytes !== null) pieces.push(`rgb_bytes=${latestYoloFrame.last_rgb_bytes}`);
      if (latestYoloFrame.model_loaded !== undefined) pieces.push(`model_loaded=${Boolean(latestYoloFrame.model_loaded)}`);
      if (latestYoloFrame.last_detections !== undefined) pieces.push(`last_dets=${latestYoloFrame.last_detections}`);
      if (latestYoloFrame.decode_failures) pieces.push(`decode_failures=${latestYoloFrame.decode_failures}`);
      if (latestYoloFrame.last_error) pieces.push(`error=${latestYoloFrame.last_error}`);
      yfd.textContent = pieces.join(" · ");
    }
  }

  const ah = $("audioHealth");
  const ahd = $("audioHealthDetail");
  if (!latestAudioStatus) {
    ah.textContent = "No audio messages yet";
    ahd.textContent = "No selected-node /audio_detector/detections message has arrived. Normal runs do not publish synthetic audio status; use --audio-debug-status only for troubleshooting.";
  } else {
    const diag = latestAudioStatus.diagnosis || (latestAudioStatus.kind === "detection" ? "active_audio_detection" : "unknown");
    ah.textContent = diagnosisText(diag);
    const pieces = [];
    pieces.push(`frames_total=${latestAudioStatus.frames_total ?? "?"}`);
    pieces.push(`detections_total=${latestAudioStatus.detections_total ?? (latestAudioStatus.kind === "detection" ? "≥1" : 0)}`);
    if (latestAudioStatus.local_ipc_exists !== null && latestAudioStatus.local_ipc_exists !== undefined) pieces.push(`ipc=${latestAudioStatus.local_ipc_exists ? "yes" : "no"}`);
    const db = latestAudioStatus.last_db ?? latestAudioStatus.avg_db ?? latestAudioStatus.db;
    if (db !== null && db !== undefined) pieces.push(`last_db=${Number(db).toFixed(1)}dB`);
    if (latestAudioStatus.last_frame_age_sec !== null && latestAudioStatus.last_frame_age_sec !== undefined) pieces.push(`last_frame=${latestAudioStatus.last_frame_age_sec}s ago`);
    if (latestAudioStatus.last_error) pieces.push(`error=${latestAudioStatus.last_error}`);
    ahd.textContent = pieces.join(" · ");
  }
}

function handleReplay(item) {
  if (item.topic === "/replay/config") lastReplayConfig = item.payload;
  if (item.topic === "/replay/sync") lastReplaySync = item.payload;
  renderReplayState();
}

function handleYoloImage(payload) {
  if (!payload || typeof payload !== "object" || !payload.image) return;
  const img = $("yoloImage");
  const fmt = payload.format || "jpg";
  img.src = `data:image/${fmt};base64,${payload.image}`;
  img.style.display = "block";
  $("imagePlaceholder").style.display = "none";
  lastYoloImageTs = Date.now() / 1000;
  evidence.yoloImage += 1;
  if (!latestYoloStatus) {
    latestYoloStatus = {
      diagnosis: payload.detections > 0 ? "active_detections" : "video_active_no_yolo_detections",
      last_detections: payload.detections ?? 0,
      input_frames_total: payload.frame_index ?? "?",
      model_loaded: true,
      last_frame_shape: payload.width && payload.height ? `${payload.width}x${payload.height}` : undefined,
    };
  }
  updateAges();
}

function handleYoloBbox(payload) {
  evidence.yoloBbox += 1;
  $("yoloJson").textContent = pretty(payload);
  latestYoloStatus = {
    ...(latestYoloStatus || {}),
    diagnosis: "active_detections",
    last_detections: Array.isArray(payload) ? payload.length : 1,
    model_loaded: true,
  };
  renderHealth();
}

function handleYoloStatus(payload) {
  if (!payload || typeof payload !== "object") return;
  evidence.yoloStatus += 1;
  latestYoloStatus = payload;
  lastYoloStatusTs = Date.now() / 1000;
  $("yoloStatusJson").textContent = pretty(payload);
  renderHealth();
}

function handleYoloFrame(payload) {
  if (!payload || typeof payload !== "object") return;
  evidence.yoloFrame += 1;
  latestYoloFrame = payload;
  lastYoloFrameTs = Date.now() / 1000;
  $("yoloFrameJson").textContent = pretty(payload);
  // A frame probe proves images are reaching and decoding inside YOLO, even
  // when there are no bbox detections.
  latestYoloStatus = {
    ...(latestYoloStatus || {}),
    diagnosis: payload.last_detections > 0 ? "active_detections" : "video_active_no_yolo_detections",
    input_frames_total: payload.input_frames_total,
    last_frame_shape: payload.last_frame_shape,
    last_rgb_bytes: payload.last_rgb_bytes,
    last_detections: payload.last_detections ?? 0,
    model_loaded: payload.model_loaded,
    model_loading: payload.model_loading,
    yolo_device: payload.yolo_device,
    decode_failures: payload.decode_failures,
    process_failures: payload.process_failures,
    last_error: payload.last_error,
  };
  renderHealth();
}

function meterWidth(db) {
  if (db === null || db === undefined || Number.isNaN(Number(db))) return 0;
  return Math.max(0, Math.min(100, ((Number(db) + 80) / 80) * 100));
}

function handleAudio(payload) {
  if (!payload || typeof payload !== "object") return;
  lastAudioTs = Date.now() / 1000;
  if (payload.kind === "status" || payload.kind === "debug_status") {
    evidence.audioStatus += 1;
    latestAudioStatus = payload;
  } else if (payload.kind === "detection" || payload.db !== undefined) {
    evidence.audioDetection += 1;
    latestAudioStatus = {
      ...payload,
      kind: "detection",
      diagnosis: payload.diagnosis || "active_audio_detection",
      last_db: payload.db ?? payload.last_db,
      last_detection: true,
    };
  }
  const db = payload.last_db ?? payload.avg_db ?? payload.db;
  if (db !== undefined && db !== null) {
    $("audioDb").textContent = `${Number(db).toFixed(1)} dB`;
    $("audioMeter").style.width = `${meterWidth(db)}%`;
  }
  $("audioThreshold").textContent = payload.threshold_db !== undefined ? `${Number(payload.threshold_db).toFixed(1)} dB` : "--";
  $("audioFrameRate").textContent = payload.frame_rate !== undefined ? payload.frame_rate : "--";
  $("audioDetections").textContent = payload.detections !== undefined ? payload.detections : (payload.kind === "detection" ? "1" : "--");
  $("audioLastDet").textContent = payload.last_detection !== undefined ? String(payload.last_detection) : (payload.kind === "detection" ? "true" : "--");
  $("audioJson").textContent = pretty(payload);
  renderHealth();
  updateAges();
}

function addSelectedLog(item) {
  selectedLogs.push(item);
  if (selectedLogs.length > maxLogEntries) selectedLogs.shift();
  lastSelectedMessageTs = item.ts || Date.now() / 1000;
  renderLog();
}

function renderLog() {
  const log = $("log");
  const shouldStick = log.scrollTop + log.clientHeight >= log.scrollHeight - 12;
  log.innerHTML = "";
  for (const item of selectedLogs) {
    const div = document.createElement("div");
    div.className = `log-entry ${item.direction}`;
    const ts = new Date(item.ts * 1000).toLocaleTimeString();
    div.innerHTML = `<span class="timestamp">${ts}</span> <span class="topic">${escapeHtml(item.topic)}</span>\n${escapeHtml(pretty(item.payload))}`;
    log.appendChild(div);
  }
  if (shouldStick) log.scrollTop = log.scrollHeight;
}

function classifySelectedTopic(topic, payload) {
  if (topic.includes("/analytics/yolo/status")) return "yoloStatus";
  if (topic.includes("/analytics/yolo/frame")) return "yoloFrame";
  if (topic.includes("/analytics/yolo/annotated/compressed")) return "yoloImage";
  if (topic.includes("/analytics/yolo/bbox")) return "yoloBbox";
  if (topic.includes("/audio_detector/status")) return "audioStatus";
  if (topic.includes("/audio_detector/detections")) return "audioDetection";
  if (topic.includes("/replay/status")) return "replayStatus";
  return "other";
}

function handleItem(item) {
  if (item.topic.startsWith("/replay/")) {
    handleReplay(item);
    // Replay topics are global, so do not return; status topics may include node names in payload.
  }

  discoverDevice(item);
  if (!itemMatchesSelected(item)) {
    renderHealth();
    return;
  }
  const klass = classifySelectedTopic(item.topic, item.payload);
  if (klass in evidence) evidence[klass] += 1;
  addSelectedLog(item);

  if (item.topic.includes("/analytics/yolo/status")) handleYoloStatus(item.payload);
  else if (item.topic.includes("/analytics/yolo/frame")) handleYoloFrame(item.payload);
  else if (item.topic.includes("/analytics/yolo/annotated/compressed")) handleYoloImage(item.payload);
  else if (item.topic.includes("/analytics/yolo/bbox")) handleYoloBbox(item.payload);
  else if (item.topic.includes("/audio_detector/")) handleAudio(item.payload);
  else renderHealth();
}

async function loadInitialState() {
  const resp = await fetch("/api/state");
  const state = await resp.json();
  updateBadge(state.mqtt.connected);
  if (state.replay) {
    if (state.replay.config) lastReplayConfig = state.replay.config;
    if (state.replay.sync) lastReplaySync = state.replay.sync;
  }
  renderReplayState();
  for (const item of state.recent || []) handleItem(item);
}

function connectEvents() {
  const source = new EventSource("/api/events");
  source.addEventListener("mqtt", (event) => {
    const item = JSON.parse(event.data);
    if (item.topic === "$web_ui/status" && item.payload && "connected" in item.payload) updateBadge(Boolean(item.payload.connected));
    handleItem(item);
  });
  source.addEventListener("heartbeat", (event) => {
    const status = JSON.parse(event.data);
    updateBadge(Boolean(status.connected));
  });
  source.onerror = () => updateBadge(false);
}

$("deviceSelect").addEventListener("change", (e) => setSelectedDevice(e.target.value));
$("useManual").addEventListener("click", () => setSelectedDevice($("manualDevice").value));
$("clearBtn").addEventListener("click", () => { selectedLogs = []; evidence = resetEvidence(); lastSelectedMessageTs = null; renderLog(); renderHealth(); });

renderDeviceSelect();
setSelectedDevice(selectedDevice);
loadInitialState().then(connectEvents).catch((err) => {
  console.error(err);
  connectEvents();
});
setInterval(updateAges, 1000);
