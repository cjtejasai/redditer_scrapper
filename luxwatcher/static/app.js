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

// CSV column definitions matching the expected format
const CSV_COLUMNS = [
  { key: "url", header: "url" },
  { key: "source", header: "source" },
  { key: "buyer_stage", header: "buyer_stage" },
  { key: "confidence", header: "confidence" },
  { key: "property_type", header: "property_type" },
  { key: "budget", header: "budget" },
  { key: "evidence", header: "evidence" },
  { key: "reason", header: "reason_for_sales" },
  { key: "thread_type", header: "thread_type" },
  { key: "thread_summary", header: "thread_summary" },
];

function getFilteredLeads(items, filterText) {
  const f = norm(filterText);
  if (!f) return items;
  return items.filter((it) => {
    const blob = [
      it.url,
      it.source,
      it.buyer_stage,
      it.confidence,
      it.property_type,
      it.budget,
      it.evidence,
      it.reason,
      it.thread_type,
      it.thread_summary,
    ]
      .map(norm)
      .join(" ");
    return blob.includes(f);
  });
}

function renderRows(items, filterText) {
  const body = document.getElementById("leadsBody");
  body.innerHTML = "";

  const filtered = getFilteredLeads(items, filterText);
  window.__filteredLeads = filtered;

  for (const it of filtered) {
    const url = it.url ?? "";
    const source = it.source ?? "op";
    const buyerStage = it.buyer_stage ?? "—";
    const confidence = it.confidence ?? "—";
    const propertyType = it.property_type ?? "unknown";
    const budget = it.budget && it.budget !== "not_mentioned" ? it.budget : "not_mentioned";
    const evidence = it.evidence ?? "";
    const reason = it.reason ?? "";
    const threadType = it.thread_type ?? it.lead_type ?? "—";
    const threadSummary = it.thread_summary ?? it.title ?? "";

    const tr = document.createElement("tr");

    // Add stage badge styling
    const stageBadge = buyerStage !== "—" ? `<span class="badge badge-${buyerStage.replace('_', '-')}">${buyerStage}</span>` : "—";

    tr.innerHTML = `
      <td class="nowrap"><a href="${url}" target="_blank" rel="noreferrer">${url ? "Link" : "—"}</a></td>
      <td class="nowrap">${source}</td>
      <td class="nowrap">${stageBadge}</td>
      <td class="nowrap">${confidence}</td>
      <td class="nowrap">${propertyType}</td>
      <td class="nowrap">${budget}</td>
      <td class="cell-evidence">${evidence.replaceAll("<", "&lt;")}</td>
      <td class="cell-reason">${reason.replaceAll("<", "&lt;")}</td>
      <td class="nowrap">${threadType}</td>
      <td class="cell-summary">${threadSummary.replaceAll("<", "&lt;")}</td>
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
    fetchJSON("/watcher/api/status"),
    fetchJSON("/watcher/api/runs?limit=50"),
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
    ? `/watcher/api/leads?limit=500&run_id=${encodeURIComponent(selectedRunId)}${includeAllParam}`
    : `/watcher/api/leads?limit=500${includeAllParam}`;

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
    const res = await fetch("/watcher/api/run", { method: "POST" });
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

// CSV export functionality
function escapeCsvValue(value) {
  if (value == null) return "";
  const str = String(value);
  // If contains comma, newline, or quote, wrap in quotes and escape inner quotes
  if (str.includes(",") || str.includes("\n") || str.includes('"')) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

function generateCsv(items) {
  const headers = CSV_COLUMNS.map((c) => c.header);
  const rows = [headers.join("\t")]; // Use tab separator for better readability

  for (const it of items) {
    const row = CSV_COLUMNS.map((col) => {
      let value = it[col.key];
      if (col.key === "source") value = value ?? "op";
      if (col.key === "budget" && (!value || value === "not_mentioned")) value = "not_mentioned";
      if (col.key === "property_type") value = value ?? "unknown";
      if (col.key === "thread_type") value = value ?? it.lead_type ?? "";
      if (col.key === "thread_summary") value = value ?? it.title ?? "";
      return escapeCsvValue(value ?? "");
    });
    rows.push(row.join("\t"));
  }

  return rows.join("\n");
}

function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function downloadCsv() {
  const items = window.__filteredLeads ?? window.__leads ?? [];
  if (items.length === 0) {
    alert("No leads to download");
    return;
  }
  const csv = generateCsv(items);
  const runId = document.getElementById("shownRun").textContent || "leads";
  downloadFile(csv, `leads_${runId}.csv`, "text/csv;charset=utf-8;");
}

function downloadExcel() {
  const items = window.__filteredLeads ?? window.__leads ?? [];
  if (items.length === 0) {
    alert("No leads to download");
    return;
  }
  // Generate tab-separated values which Excel can open
  const tsv = generateCsv(items);
  const runId = document.getElementById("shownRun").textContent || "leads";
  // Use .xls extension with TSV content - Excel handles this well
  downloadFile(tsv, `leads_${runId}.xls`, "application/vnd.ms-excel;charset=utf-8;");
}

document.getElementById("downloadCsv").addEventListener("click", downloadCsv);
document.getElementById("downloadExcel").addEventListener("click", downloadExcel);

refresh().catch((e) => {
  document.getElementById("lastError").textContent = `UI error: ${e}`;
});
