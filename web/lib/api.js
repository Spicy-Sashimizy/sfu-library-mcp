/* SFU Library Suite — shared API client.
 * Plain JS (no JSX); loaded before each app's babel script so window.SFUApi is
 * available. Every method tries the live backend first and falls back to
 * window.SFUMock when the backend is unreachable or errors. The returned object
 * always carries `source: "live" | "mock"` so the UI can surface a demo banner.
 *
 * Endpoints (served by src/sfu_library_mcp_http.py):
 *   POST /api/search        -> { results:[...], meta:{count}, notes:[...] }
 *   GET  /analytics[?panel] -> { panels:{...} } | { panel, data }
 *   GET  /api/index_status  -> { metrics, pack, unpack_jobs }
 *   GET  /api/personas      -> { personas, sections, live_sections, ... }
 *   POST /api/unpack        -> { ok, message }
 *   POST /engagement        -> { recorded }
 *   GET  /health            -> { status:"ok", tools:N }
 */
(function () {
  const BASE = ""; // same origin as the served page
  const TIMEOUT_MS = 12000;

  function withTimeout(promise, ms) {
    let t;
    const timeout = new Promise((_, rej) => { t = setTimeout(() => rej(new Error("timeout")), ms); });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(t));
  }

  async function req(path, opts) {
    const res = await withTimeout(fetch(BASE + path, opts), TIMEOUT_MS);
    if (!res.ok) throw new Error("HTTP " + res.status);
    return res.json();
  }

  // OpenAlex/retriever dict -> the card shape the Search prototype renders.
  function normalizeResult(w, i) {
    const date = w.date || w.publication_date || "";
    const year = w.year || w.publication_year ||
      (typeof date === "string" && date.length >= 4 ? parseInt(date.slice(0, 4), 10) : undefined);
    const doiRaw = (w.doi || "").replace(/^https?:\/\/(dx\.)?doi\.org\//i, "");
    const isOA = !!(w.is_oa || w.oa_status);
    const cited = w.cited_by != null ? w.cited_by : (w.cited_by_count || 0);
    const type = w.type || "article";
    const topics = w.topics || [];
    const abstract = w.abstract || "";
    const access = isOA
      ? { type: "open_access", url: w.oa_url || (doiRaw ? "https://doi.org/" + doiRaw : "#"), label: "Open Access" }
      : { type: "sfu_proxy", url: doiRaw ? "https://doi.org/" + doiRaw : "#", label: "SFU Library" };
    return {
      id: w.id || w.openalex_id || ("R" + i),
      title: w.title || "Untitled",
      authors: w.authors || [],
      year: year,
      source: w.source || w.venue || "",
      volume: w.volume || "",
      issue: w.issue || "",
      doi: doiRaw,
      cited_by: cited,
      type: type,
      peer_reviewed: w.peer_reviewed != null ? w.peer_reviewed : (type === "article" || type === "book"),
      abstract: abstract,
      tldr: w.tldr || (abstract ? abstract.slice(0, 220) + (abstract.length > 220 ? "…" : "") : ""),
      access: w.access || access,
      expanded_terms: w.expanded_terms || [],
      topics: topics,
      oa_status: isOA,
    };
  }

  // Map the prototype's filter UI values to search_academic args.
  function buildSearchArgs(query, filters) {
    filters = filters || {};
    const args = { query: query, limit: 50 };
    const t = filters.type;
    if (t && t !== "All") args.type = t.toLowerCase();
    const a = filters.access;
    if (a === "Open Access" || a === "Unpaywall") args.open_access_only = true;
    const now = 2026;
    const d = filters.date;
    if (d === "Last year") args.year_from = now - 1;
    else if (d === "Last 5 years") args.year_from = now - 5;
    else if (d === "Last 10 years") args.year_from = now - 10;
    return args;
  }

  const SFUApi = {
    _online: null, // last known backend reachability (null=unknown)

    isOnline() { return this._online; },

    async health() {
      try {
        const data = await req("/health", { method: "GET" });
        this._online = true;
        return { data: data, source: "live" };
      } catch (e) {
        this._online = false;
        return { data: { status: "offline" }, source: "mock" };
      }
    },

    async search(query, filters) {
      try {
        const data = await req("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(buildSearchArgs(query, filters)),
        });
        this._online = true;
        const raw = (data && data.results) || [];
        return {
          results: raw.map(normalizeResult),
          notes: (data && data.notes) || [],
          meta: (data && data.meta) || { count: raw.length },
          source: "live",
        };
      } catch (e) {
        this._online = false;
        const mock = (window.SFUMock && window.SFUMock.results) || [];
        return { results: mock, notes: [], meta: { count: mock.length }, source: "mock" };
      }
    },

    async analytics(panel) {
      try {
        const q = panel ? ("?panel=" + encodeURIComponent(panel)) : "";
        const data = await req("/analytics" + q, { method: "GET" });
        this._online = true;
        return { data: data, source: "live" };
      } catch (e) {
        this._online = false;
        return { data: (window.SFUMock && window.SFUMock.analytics) || {}, source: "mock" };
      }
    },

    async indexStatus() {
      try {
        const data = await req("/api/index_status", { method: "GET" });
        this._online = true;
        return { data: data, source: "live" };
      } catch (e) {
        this._online = false;
        return { data: (window.SFUMock && window.SFUMock.indexStatus) || {}, source: "mock" };
      }
    },

    async personas() {
      try {
        const data = await req("/api/personas", { method: "GET" });
        this._online = true;
        return { data: data, source: "live" };
      } catch (e) {
        this._online = false;
        return { data: (window.SFUMock && window.SFUMock.personas) || {}, source: "mock" };
      }
    },

    async unpack(section) {
      try {
        const data = await req("/api/unpack", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ section: section }),
        });
        return { data: data, source: "live" };
      } catch (e) {
        return { data: { ok: false, message: "Backend unavailable (demo mode)." }, source: "mock" };
      }
    },

    async engagement(evt) {
      try {
        await req("/engagement", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(evt),
        });
        return { source: "live" };
      } catch (e) {
        return { source: "mock" };
      }
    },
  };

  window.SFUApi = SFUApi;
  window.SFUApi.normalizeResult = normalizeResult;
})();
