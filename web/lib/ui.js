/* SFU Library Suite — shared UI helpers (plain JS, no JSX).
 * Provides a global demo-data banner so every app gives the SAME, unmissable
 * signal whenever it is rendering mock/fallback data instead of live results. */
(function () {
  const BANNER_ID = "sfu-demo-banner";
  const STYLE_ID = "sfu-demo-banner-style";

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = [
      "#" + BANNER_ID + "{position:fixed;top:0;left:0;right:0;z-index:2147483000;",
      "display:flex;align-items:center;justify-content:center;gap:8px;",
      "padding:6px 14px;font-family:'DM Sans',system-ui,sans-serif;font-size:12.5px;",
      "font-weight:500;color:oklch(28% 0.10 75);background:oklch(95% 0.06 75);",
      "border-bottom:1px solid oklch(80% 0.10 75);box-shadow:0 1px 4px oklch(0% 0 0 / 0.08);",
      "letter-spacing:0.01em;}",
      "#" + BANNER_ID + " .sfu-dot{width:8px;height:8px;border-radius:50%;",
      "background:oklch(68% 0.16 60);flex-shrink:0;animation:sfu-pulse 1.6s ease-in-out infinite;}",
      "#" + BANNER_ID + " b{font-weight:600;}",
      "@keyframes sfu-pulse{0%,100%{opacity:1}50%{opacity:0.35}}",
      "body.sfu-has-banner{padding-top:30px;}",
    ].join("");
    document.head.appendChild(s);
  }

  // show=true renders the banner; show=false removes it. Idempotent.
  function setMockBanner(show, message) {
    ensureStyle();
    const existing = document.getElementById(BANNER_ID);
    if (!show) {
      if (existing) existing.remove();
      document.body && document.body.classList.remove("sfu-has-banner");
      return;
    }
    const text = message ||
      "Demo data — the live search backend is unavailable. Showing sample results.";
    if (existing) {
      existing.querySelector(".sfu-msg").innerHTML = text;
      return;
    }
    const bar = document.createElement("div");
    bar.id = BANNER_ID;
    bar.setAttribute("role", "status");
    bar.innerHTML = '<span class="sfu-dot"></span><span class="sfu-msg"><b>Demo mode.</b> ' +
      text + "</span>";
    document.body.appendChild(bar);
    document.body.classList.add("sfu-has-banner");
  }

  // Convenience: pass any api result(s); banner shows if ANY is from mock.
  function reflectSource() {
    const sources = Array.prototype.slice.call(arguments);
    const anyMock = sources.some(function (r) { return r && r.source === "mock"; });
    setMockBanner(anyMock);
    return anyMock;
  }

  window.SFUUI = { setMockBanner: setMockBanner, reflectSource: reflectSource };
})();
