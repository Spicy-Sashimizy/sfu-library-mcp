# SFU Library Suite — web GUI

The web GUI for the thin-client stack. Implementation of the `claude.ai/design`
handoff bundle. Served as **static files at `/app`** by the HTTP entry point —
no Node/npm build step.

## Run it

```bash
cd src
MCP_HTTP_PORT=8080 ../.venv/bin/python3 -m uvicorn sfu_library_mcp_http:app --host 0.0.0.0 --port 8080
# open http://localhost:8080/app
```

`/app` redirects to `suite.html` (the shell), which mounts one app at a time.

## Layout

| Path | What |
|---|---|
| `suite.html` | shell; iframes an app, coordinates via `postMessage` |
| `search.html` | search app (search / saved / history / databases / settings / admin) |
| `analytics.html` | "About this index" |
| `index_manager.html` | personas / warm cache / storage budget |
| `tweaks-panel.jsx` | design-time tweak panel (loaded by `search.html`) |
| `lib/api.js` | `window.SFUApi` — live backend calls with mock fallback |
| `lib/mock.js` | `window.SFUMock` — sample/fallback data |
| `lib/ui.js` | `window.SFUUI` — the demo-data banner |
| `vendor/` | React 18.3.1 + ReactDOM + Babel-standalone (vendored, no CDN) |

## Live vs. demo data

Every `SFUApi` method tries the live backend first and falls back to `SFUMock`
on error/timeout, tagging the result `source: "live" | "mock"`. Whenever mock
data is shown — backend down **or** a live call returned nothing — `SFUUI`
renders a **demo-data banner** at the top of the app so it's never mistaken for
live results. The Search app's AI-status pill also reflects backend reachability.

## Backend endpoints used

`POST /api/search`, `GET /api/index_status`, `GET /api/personas`,
`POST /api/unpack`, `GET /analytics`, `POST /engagement`, `GET /health`.
See `docs/THIN_CLIENT_SWAP.md` → "Web GUI" for details.

## Editing

JSX compiles in the browser via vendored Babel — just edit and reload, no build.
Keep the design system (OKLCH tokens, DM Sans / Lora fonts) consistent across
apps. Sample data lives only in `lib/mock.js`.
