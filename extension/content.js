/**
 * Probability Cup Crowd Capture (content script)
 *
 * Watches the platform page, harvests every visible block that looks like a
 * question card (contains "?" and a "%"), and POSTs the raw text to the local
 * logger (tools/crowd_server.py). Parsing happens server-side in Python so
 * format changes never require an extension update.
 *
 * Read-only by design: never clicks, never types, never submits.
 * Panel pattern borrowed from CoreProp's extension.
 */

const LOGGER_URL = "http://127.0.0.1:8765/capture";
const DEBOUNCE_MS = 2500;
const MIN_LEN = 20;
const MAX_LEN = 700;

function log(...args) { console.log("[CrowdCapture]", ...args); }

// ── status panel ───────────────────────────────────────────────────────────
let panel = null;
function setStatus(text, ok) {
  if (!panel) {
    panel = document.createElement("div");
    Object.assign(panel.style, {
      position: "fixed", bottom: "16px", right: "16px", zIndex: "2147483647",
      background: "#0f172a", border: "1px solid #1e3a5f", borderRadius: "10px",
      padding: "8px 12px", fontFamily: "system-ui, sans-serif", fontSize: "12px",
      color: "#e2e8f0", boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
      pointerEvents: "none", opacity: "0.92",
    });
    document.body.appendChild(panel);
  }
  panel.style.borderColor = ok ? "#14532d" : "#7f1d1d";
  panel.textContent = `Crowd Capture: ${text}`;
}

// ── harvesting ─────────────────────────────────────────────────────────────
function qualifies(text) {
  return text && text.length >= MIN_LEN && text.length <= MAX_LEN
      && text.includes("?") && text.includes("%");
}

function collectBlocks() {
  const candidates = document.querySelectorAll("div, li, section, article");
  const blocks = new Set();
  for (const el of candidates) {
    const t = (el.innerText || "").trim();
    if (!qualifies(t)) continue;
    // keep only leaf-most qualifying containers (skip wrappers)
    let childQualifies = false;
    for (const c of el.querySelectorAll("div, li, section, article")) {
      if (qualifies((c.innerText || "").trim())) { childQualifies = true; break; }
    }
    if (!childQualifies) blocks.add(t);
  }
  return [...blocks];
}

async function capture(reason) {
  const blocks = collectBlocks();
  if (blocks.length === 0) {
    setStatus("no question cards visible", false);
    return { stored: 0, blocks: 0 };
  }
  try {
    const res = await fetch(LOGGER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: location.href,
        title: document.title,
        ts: new Date().toISOString(),
        reason,
        blocks,
      }),
    });
    const data = await res.json();
    setStatus(`${data.stored} questions logged (${reason})`, true);
    log("captured", data);
    return data;
  } catch (e) {
    setStatus("logger offline — run tools/crowd_server.py", false);
    log("logger unreachable", e);
    return { error: String(e) };
  }
}

// ── triggers: page load + DOM changes (debounced) + popup button ──────────
let timer = null;
function schedule(reason) {
  clearTimeout(timer);
  timer = setTimeout(() => capture(reason), DEBOUNCE_MS);
}

new MutationObserver(() => schedule("auto")).observe(document.body,
  { childList: true, subtree: true });
schedule("load");

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === "pc-capture") {
    capture("manual").then(sendResponse);
    return true; // async response
  }
});
