/**
 * SFU Library PDF Capture - Background Service Worker
 *
 * Listens for PDF responses via webRequest, grabs cookies for the domain,
 * and POSTs capture requests to the local capture server.
 */

const DEFAULT_SERVER_URL = "http://localhost:8787";
const CAPTURE_HISTORY_MAX = 50;

// ─── State ────────────────────────────────────────────────────

let autoCapture = true;
let serverUrl = DEFAULT_SERVER_URL;
let captureHistory = [];

// Load saved settings on startup
chrome.storage.local.get(
  ["autoCapture", "serverUrl", "captureHistory"],
  (data) => {
    if (data.autoCapture !== undefined) autoCapture = data.autoCapture;
    if (data.serverUrl) serverUrl = data.serverUrl;
    if (data.captureHistory) captureHistory = data.captureHistory;
  }
);

// ─── PDF Detection via webRequest ─────────────────────────────

chrome.webRequest.onHeadersReceived.addListener(
  (details) => {
    if (!autoCapture) return;

    const contentType = details.responseHeaders?.find(
      (h) => h.name.toLowerCase() === "content-type"
    );

    const isPdf =
      contentType?.value?.includes("application/pdf") ||
      details.url.toLowerCase().endsWith(".pdf");

    if (isPdf && details.statusCode >= 200 && details.statusCode < 400) {
      handlePdfDetected(details.url, details.tabId);
    }
  },
  { urls: ["<all_urls>"] },
  ["responseHeaders"]
);

// ─── PDF Capture Logic ────────────────────────────────────────

async function handlePdfDetected(url, tabId) {
  try {
    // Get all cookies for the URL domain
    const urlObj = new URL(url);
    const cookies = await chrome.cookies.getAll({ domain: urlObj.hostname });
    const cookieDict = {};
    cookies.forEach((c) => {
      cookieDict[c.name] = c.value;
    });

    // Get tab title for context
    let tabTitle = "";
    try {
      const tab = await chrome.tabs.get(tabId);
      tabTitle = tab.title || "";
    } catch {
      // Tab may have closed
    }

    await sendCapture(url, cookieDict, tabTitle);
  } catch (err) {
    console.error("PDF capture error:", err);
    updateBadge("!", "#ff0000");
  }
}

async function sendCapture(url, cookies, tabTitle) {
  const payload = {
    url,
    cookies,
    filename: tabTitle || undefined,
  };

  try {
    const response = await fetch(`${serverUrl}/capture`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    const entry = {
      url: url.substring(0, 100),
      timestamp: new Date().toISOString(),
      success: data.success || false,
      error: data.error || null,
    };

    captureHistory.unshift(entry);
    if (captureHistory.length > CAPTURE_HISTORY_MAX) {
      captureHistory = captureHistory.slice(0, CAPTURE_HISTORY_MAX);
    }
    chrome.storage.local.set({ captureHistory });

    if (data.success) {
      updateBadge("+1", "#4caf50");
    } else {
      updateBadge("err", "#ff9800");
    }
  } catch (err) {
    console.error("Capture server request failed:", err);

    const entry = {
      url: url.substring(0, 100),
      timestamp: new Date().toISOString(),
      success: false,
      error: `Server unreachable: ${err.message}`,
    };
    captureHistory.unshift(entry);
    if (captureHistory.length > CAPTURE_HISTORY_MAX) {
      captureHistory = captureHistory.slice(0, CAPTURE_HISTORY_MAX);
    }
    chrome.storage.local.set({ captureHistory });

    updateBadge("!", "#ff0000");
  }
}

// ─── Badge helpers ────────────────────────────────────────────

function updateBadge(text, color) {
  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color });
  // Clear badge after 3 seconds
  setTimeout(() => chrome.action.setBadgeText({ text: "" }), 3000);
}

// ─── Message handler (from popup and content scripts) ─────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "getState") {
    sendResponse({ autoCapture, serverUrl, captureHistory });
  } else if (msg.type === "setAutoCapture") {
    autoCapture = msg.value;
    chrome.storage.local.set({ autoCapture });
    sendResponse({ ok: true });
  } else if (msg.type === "setServerUrl") {
    serverUrl = msg.value;
    chrome.storage.local.set({ serverUrl });
    sendResponse({ ok: true });
  } else if (msg.type === "manualCapture") {
    handlePdfDetected(msg.url, sender.tab?.id || 0);
    sendResponse({ ok: true });
  } else if (msg.type === "captureFromContent") {
    // Content script detected PDF links - capture them
    handlePdfDetected(msg.url, sender.tab?.id || 0);
    sendResponse({ ok: true });
  } else if (msg.type === "checkHealth") {
    fetch(`${serverUrl}/health`)
      .then((r) => r.json())
      .then((data) => sendResponse({ connected: true, data }))
      .catch(() => sendResponse({ connected: false }));
    return true; // async response
  }
});
