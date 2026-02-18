/**
 * SFU Library PDF Capture - Content Script
 *
 * Scans the page for PDF links and highlights them.
 * Sends detected PDF links to the background service worker for capture.
 */

(function () {
  "use strict";

  const PDF_LINK_CLASS = "sfu-pdf-capture-highlight";
  const STYLE_ID = "sfu-pdf-capture-styles";

  // ─── Inject styles ─────────────────────────────────────────

  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .${PDF_LINK_CLASS} {
        outline: 2px solid #4caf50 !important;
        outline-offset: 2px !important;
        position: relative;
      }
      .${PDF_LINK_CLASS}::after {
        content: "PDF";
        position: absolute;
        top: -8px;
        right: -8px;
        background: #4caf50;
        color: white;
        font-size: 10px;
        padding: 1px 4px;
        border-radius: 3px;
        font-family: sans-serif;
        pointer-events: none;
      }
    `;
    document.head.appendChild(style);
  }

  // ─── Scan for PDF links ────────────────────────────────────

  function findPdfLinks() {
    const links = [];
    const anchors = document.querySelectorAll("a[href]");

    anchors.forEach((a) => {
      const href = a.href.toLowerCase();
      const text = a.textContent.toLowerCase();

      const isPdfLink =
        href.endsWith(".pdf") ||
        href.includes("/pdf/") ||
        href.includes("format=pdf") ||
        href.includes("type=pdf") ||
        text.includes("download pdf") ||
        text.includes("full text (pdf)") ||
        text.includes("view pdf") ||
        (text.includes("pdf") && a.closest(".download, .fulltext, .access"));

      if (isPdfLink) {
        links.push(a);
      }
    });

    return links;
  }

  // ─── Highlight PDF links ───────────────────────────────────

  function highlightPdfLinks() {
    const links = findPdfLinks();

    links.forEach((link) => {
      if (!link.classList.contains(PDF_LINK_CLASS)) {
        link.classList.add(PDF_LINK_CLASS);

        // Add click handler to send to capture server
        link.addEventListener("click", (e) => {
          // Don't prevent default — let the browser handle the download
          chrome.runtime.sendMessage({
            type: "captureFromContent",
            url: link.href,
          });
        });
      }
    });

    return links.length;
  }

  // ─── Run on load and observe mutations ─────────────────────

  const count = highlightPdfLinks();
  if (count > 0) {
    console.log(`[SFU PDF Capture] Found ${count} PDF links on page`);
  }

  // Re-scan when DOM changes (SPA navigation, lazy loading)
  const observer = new MutationObserver(() => {
    highlightPdfLinks();
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
})();
