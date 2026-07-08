const counters = { replay: 0, yolo: 0, audio: 0, ce: 0 };
const maxLogEntries = 300;
let logEntries = [];
let dataRoots = [
  "/media/brianw/Extreme SSD/West Point Experimentation",
  "/media/brianw/Extreme SSD/GQ Data",
];

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

function addLog(item) {
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


function prefillScenarioFromUrlOrStorage() {
  const params = new URLSearchParams(window.location.search);
  const selected = params.get("scenario") || localStorage.getItem("iobt_selected_scenario") || "";
  if (selected && $("scenario")) {
    $("scenario").value = selected;
    updateCandidateFolders();
  }
}

async function loadInitialState() {
  const resp = await fetch("/api/state");
  const state = await resp.json();
  updateBadge(state.mqtt.connected);
  dataRoots = state.data_roots || dataRoots;
  updateCandidateFolders();
  for (const item of state.recent || []) addLog(item);
}

function connectEvents() {
  const source = new EventSource("/api/events");
  source.addEventListener("mqtt", (event) => {
    const item = JSON.parse(event.data);
    addLog(item);
    if (item.topic === "$web_ui/status" && item.payload && "connected" in item.payload) {
      updateBadge(Boolean(item.payload.connected));
    }
  });
  source.addEventListener("heartbeat", (event) => {
    const status = JSON.parse(event.data);
    updateBadge(Boolean(status.connected));
  });
  source.onerror = () => updateBadge(false);
}

$("replayForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = {
    scenario: $("scenario").value.trim(),
    start: Number($("start").value || 0),
    end: Number($("end").value || -1),
    sync_delay: Number($("syncDelay").value || 10),
    send_control: $("sendControl").checked,
  };
  try {
    const res = await postJson("/api/replay/start", body);
    $("lastAction").textContent = res.message;
  } catch (err) {
    $("lastAction").textContent = `Error: ${err.message}`;
  }
});

$("stopBtn").addEventListener("click", async () => {
  try {
    await postJson("/api/control", { action: "collection-stop" });
    $("lastAction").textContent = "Published collection-stop.";
  } catch (err) {
    $("lastAction").textContent = `Error: ${err.message}`;
  }
});

$("shutdownBtn").addEventListener("click", async () => {
  try {
    await postJson("/api/control", { action: "collection-shutdown" });
    $("lastAction").textContent = "Published collection-shutdown.";
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

$("scenario").addEventListener("input", updateCandidateFolders);
$("filter").addEventListener("input", renderLog);
$("clearBtn").addEventListener("click", () => {
  logEntries = [];
  renderLog();
});

prefillScenarioFromUrlOrStorage();
loadInitialState().then(connectEvents).catch((err) => {
  console.error(err);
  connectEvents();
});
