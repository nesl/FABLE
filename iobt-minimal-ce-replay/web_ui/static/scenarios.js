let scenarios = [];

const $ = (id) => document.getElementById(id);

function fmt(value) {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "duration unknown";
  const s = Math.round(Number(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h) return `${h}h ${m}m ${sec}s`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function haystack(row) {
  return [
    row.scenario_id,
    row.source_label,
    row.date,
    ...(row.nodes || []),
    ...(row.zed_nodes || []),
    ...(row.respeaker_nodes || []),
    ...(row.gps_objects || []),
    ...(row.modalities || []),
    row.date_dir,
  ].join(" ").toLowerCase();
}

function filteredRows() {
  const f = $("filter").value.trim().toLowerCase();
  const dev = $("deviceFilter").value.trim().toLowerCase();
  const modality = $("modalityFilter").value;
  return scenarios.filter((row) => {
    const h = haystack(row);
    if (f && !h.includes(f)) return false;
    if (dev) {
      const devices = [...(row.nodes || []), ...(row.gps_objects || [])].join(" ").toLowerCase();
      if (!devices.includes(dev)) return false;
    }
    if (modality && !(row.modalities || []).includes(modality)) return false;
    return true;
  });
}

function render() {
  const rows = filteredRows();
  $("countBadge").textContent = `${rows.length}/${scenarios.length} scenarios`;
  const tbody = $("scenarioRows");
  tbody.innerHTML = "";
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8">No matching scenarios.</td></tr>';
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    const end = row.observed_end_datetime || null;
    const start = row.observed_start_datetime || row.start_datetime;
    const devices = [
      row.nodes && row.nodes.length ? `nodes: ${row.nodes.join(", ")}` : "",
      row.gps_objects && row.gps_objects.length ? `gps: ${row.gps_objects.join(", ")}` : "",
    ].filter(Boolean).join("\n");
    tr.innerHTML = `
      <td><code>${row.scenario_id}</code></td>
      <td>${fmt(row.source_label)}</td>
      <td>${fmt(start)}</td>
      <td>${fmt(end)}<br><span class="muted">${fmtDuration(row.duration_seconds)}</span></td>
      <td>${(row.modalities || []).map((m) => `<span class="pill">${m}</span>`).join(" ")}</td>
      <td><pre class="compact-pre">${devices || "—"}</pre></td>
      <td>${row.file_count}<br><span class="muted">zed ${row.zed_file_count || 0}, audio ${row.respeaker_file_count || 0}, gps ${row.gps_file_count || 0}</span></td>
      <td><button type="button" data-scenario="${row.scenario_id}">Use</button></td>
    `;
    tbody.appendChild(tr);
  }
  tbody.querySelectorAll("button[data-scenario]").forEach((button) => {
    button.addEventListener("click", () => {
      const scenario = button.getAttribute("data-scenario");
      localStorage.setItem("iobt_selected_scenario", scenario);
      window.location.href = `/?scenario=${encodeURIComponent(scenario)}`;
    });
  });
}

async function loadCatalog(refresh = false) {
  $("catalogMeta").textContent = refresh ? "Refreshing catalog…" : "Loading catalog…";
  const resp = await fetch(refresh ? "/api/scenarios/refresh" : "/api/scenarios", { method: refresh ? "POST" : "GET" });
  const payload = await resp.json();
  if (!resp.ok) throw new Error(payload.detail || `HTTP ${resp.status}`);
  scenarios = payload.scenarios || [];
  const meta = payload.metadata || {};
  const roots = (meta.roots || []).join("; ");
  $("catalogMeta").innerHTML = `Catalog contains <strong>${scenarios.length}</strong> scenarios. Roots: <code>${roots || "none"}</code>. JSON: <code>${(payload.paths || {}).json || "generated/scenario_catalog.json"}</code>`;
  render();
}

$("filter").addEventListener("input", render);
$("deviceFilter").addEventListener("input", render);
$("modalityFilter").addEventListener("change", render);
$("refreshBtn").addEventListener("click", () => loadCatalog(true).catch((err) => {
  $("catalogMeta").textContent = `Refresh failed: ${err.message}`;
}));

loadCatalog(false).catch((err) => {
  $("catalogMeta").textContent = `Catalog load failed: ${err.message}`;
});
