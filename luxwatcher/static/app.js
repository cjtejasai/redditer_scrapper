async function fetchJSON(path) {
  const res = await fetch(path, { headers: { "cache-control": "no-cache" } });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

function norm(s) {
  return (s ?? "").toString().toLowerCase();
}

function setPill({ running, last_error }) {
  const el = document.getElementById("runningPill");
  el.classList.remove("running", "ok", "err");
  if (running) {
    el.textContent = "Running";
    el.classList.add("running");
    return;
  }
  if (last_error) {
    el.textContent = "Error";
    el.classList.add("err");
    return;
  }
  el.textContent = "Idle";
  el.classList.add("ok");
}

function renderRows(items, filterText) {
  const body = document.getElementById("leadsBody");
  body.innerHTML = "";

  const f = norm(filterText);
  const filtered = !f
    ? items
    : items.filter((it) => {
        const blob = [
          it.market,
          it.vertical,
          it.lead_type,
          it.query,
          it.title,
          it.url,
          it.reason,
          it.evidence,
        ]
          .map(norm)
          .join(" ");
        return blob.includes(f);
      });

  for (const it of filtered) {
    const runId = it.__run_id ?? "";
    const category = it.vertical ?? it.category ?? "—";
    const intent = it.lead_type ?? it.intent ?? it.type ?? "—";
    const confidence = it.confidence ?? "—";
    const query = it.query ?? "—";
    const title = it.title ?? it.post_title ?? "";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="nowrap small">${runId}</td>
      <td class="nowrap">${it.market ?? ""}</td>
      <td class="nowrap">${category}</td>
      <td class="nowrap">${intent}</td>
      <td class="nowrap">${confidence}</td>
      <td class="small">${(query ?? "").toString().replaceAll("<", "&lt;")}</td>
      <td>${(title ?? "").toString().replaceAll("<", "&lt;")}</td>
      <td class="nowrap"><a href="${it.url ?? ""}" target="_blank" rel="noreferrer">open</a></td>
      <td>${(it.reason ?? "").replaceAll("<", "&lt;")}</td>
      <td class="small">${(it.evidence ?? "").replaceAll("<", "&lt;")}</td>
      <td class="nowrap small">${it.ts_utc ?? ""}</td>
    `;
    body.appendChild(tr);
  }
}

function runLabel(meta) {
  const id = meta?.run_id ?? "";
  const leads = meta?.summary?.leads;
  const total = meta?.summary?.total;
  const err = meta?.error;
  const suffix = err ? "error" : `${leads ?? "?"}/${total ?? "?"} leads/total`;
  return `${id} (${suffix})`;
}

function populateRunSelect(runs, latestRunId) {
  const sel = document.getElementById("runSelect");
  const prev = sel.value;
  sel.innerHTML = "";

  if (!runs || runs.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No previous runs yet";
    sel.appendChild(opt);
    sel.disabled = true;
    return "";
  }

  sel.disabled = false;
  for (const r of runs) {
    const opt = document.createElement("option");
    opt.value = r.run_id ?? "";
    opt.textContent = runLabel(r);
    sel.appendChild(opt);
  }

  const target =
    prev && runs.some((r) => r.run_id === prev)
      ? prev
      : latestRunId ?? runs[0].run_id ?? "";
  sel.value = target;
  return target;
}

async function refresh() {
  const [status, runsResp] = await Promise.all([
    fetchJSON("/api/status"),
    fetchJSON("/api/runs?limit=50"),
  ]);
  setPill(status);
  document.getElementById("lastStarted").textContent = status.last_started ?? "—";
  document.getElementById("lastFinished").textContent = status.last_finished ?? "—";
  document.getElementById("lastError").textContent = status.last_error ?? "—";
  document.getElementById("stage").textContent = status.stage ?? "—";
  document.getElementById("stageDetail").textContent = status.stage_detail ?? "—";

  const runs = runsResp.items ?? [];
  window.__runs = runs;
  const selectedRunId = populateRunSelect(runs, runsResp.latest_run_id);
  document.getElementById("shownRun").textContent = selectedRunId || "—";

  const meta = runs.find((r) => r.run_id === selectedRunId);
  document.getElementById("leadsPath").textContent =
    meta?.outputs?.leads_out ?? status.leads_path ?? "—";

  const showAll = document.getElementById("showAll").checked;
  const includeAllParam = showAll ? "&include_all=true" : "";
  const leadsUrl = selectedRunId
    ? `/api/leads?limit=500&run_id=${encodeURIComponent(selectedRunId)}${includeAllParam}`
    : `/api/leads?limit=500${includeAllParam}`;

  const leads = await fetchJSON(leadsUrl);
  window.__leads = (leads.items ?? []).map((it) => ({
    ...it,
    __run_id: selectedRunId,
  }));
  renderRows(window.__leads, document.getElementById("filter").value);
}

async function runNow() {
  const btn = document.getElementById("runNow");
  btn.disabled = true;
  try {
    const res = await fetch("/api/run", { method: "POST" });
    if (res.status === 409) return; // already running
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    await refresh();
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("refresh").addEventListener("click", refresh);
document.getElementById("runNow").addEventListener("click", runNow);
document.getElementById("runSelect").addEventListener("change", refresh);
document.getElementById("showAll").addEventListener("change", refresh);
document.getElementById("filter").addEventListener("input", (e) => {
  renderRows(window.__leads ?? [], e.target.value);
});

refresh().catch((e) => {
  document.getElementById("lastError").textContent = `UI error: ${e}`;
});
