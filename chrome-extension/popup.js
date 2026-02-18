/**
 * SFU Library PDF Capture - Popup Script
 */

document.addEventListener("DOMContentLoaded", () => {
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const autoCaptureToggle = document.getElementById("auto-capture");
  const manualUrl = document.getElementById("manual-url");
  const manualBtn = document.getElementById("manual-btn");
  const historyList = document.getElementById("history-list");

  // ─── Load state from background ─────────────────────────────

  chrome.runtime.sendMessage({ type: "getState" }, (state) => {
    if (state) {
      autoCaptureToggle.checked = state.autoCapture;
      renderHistory(state.captureHistory || []);
    }
  });

  // ─── Check server health ────────────────────────────────────

  chrome.runtime.sendMessage({ type: "checkHealth" }, (result) => {
    if (result?.connected) {
      statusDot.className = "status-dot connected";
      statusText.textContent = "Server connected";
    } else {
      statusDot.className = "status-dot disconnected";
      statusText.textContent = "Server disconnected";
    }
  });

  // ─── Auto-capture toggle ────────────────────────────────────

  autoCaptureToggle.addEventListener("change", () => {
    chrome.runtime.sendMessage({
      type: "setAutoCapture",
      value: autoCaptureToggle.checked,
    });
  });

  // ─── Manual capture ─────────────────────────────────────────

  manualBtn.addEventListener("click", () => {
    const url = manualUrl.value.trim();
    if (!url) return;

    manualBtn.textContent = "Capturing...";
    manualBtn.disabled = true;

    chrome.runtime.sendMessage({ type: "manualCapture", url }, () => {
      manualBtn.textContent = "Capture URL";
      manualBtn.disabled = false;
      manualUrl.value = "";

      // Refresh history after a short delay
      setTimeout(() => {
        chrome.runtime.sendMessage({ type: "getState" }, (state) => {
          if (state) renderHistory(state.captureHistory || []);
        });
      }, 1000);
    });
  });

  manualUrl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") manualBtn.click();
  });

  // ─── History rendering ──────────────────────────────────────

  function renderHistory(history) {
    if (!history.length) {
      historyList.innerHTML = '<div style="color:#999;font-size:11px;">No captures yet</div>';
      return;
    }

    historyList.innerHTML = history
      .slice(0, 10)
      .map((entry) => {
        const cls = entry.success ? "success" : "error";
        const status = entry.success ? "OK" : entry.error || "Failed";
        const time = new Date(entry.timestamp).toLocaleTimeString();
        return `
          <div class="history-item ${cls}">
            <div class="url">${escapeHtml(entry.url)}</div>
            <div class="meta">${time} - ${escapeHtml(status)}</div>
          </div>
        `;
      })
      .join("");
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
});
