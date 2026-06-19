# DO demo droplet — boot stack (Qdrant on_disk + thin-client MCP, off-BMP)

What the on-demand DigitalOcean droplet runs once the NAS broker wakes it. Companion
to `docs/infrastructure/HYBRID_DEMO_DEPLOYMENT.md` (Phase 1/2) and the broker in
`scripts/demo_broker/`.

## Files
| File | Role |
|---|---|
| `docker-compose.demo.yml` | Qdrant (sparse, on_disk) + thin-client MCP (port 8080), wired `SFU_SPLADE_BACKEND=qdrant` |
| `cloud-init.sh` | boot: mount volume → compose up → **pre-warm** → ready |

## Serving shape
The thin-client serves **BM25F (tantivy) + dense (usearch)** from the volume's
`thinclient_index/`, and **SPLADE from Qdrant** — so the 68.6 GB resident BMP set is
never loaded (the point of the off-BMP migration). The volume (seeded by
`scripts/seed_demo_volume.sh`) carries both `thinclient_index/` and `qdrant_storage/`.

## Pre-warm matters (measured 2026-06-19)
First SPLADE query pays a cold model load: **fp32 ~104.6 s, fp16 ~6.1 s** (warm
~40 ms both; identical top terms). So the stack uses the **fp16 encoder**
(`SFU_SPLADE_MODEL_PATH=/app/models/splade_onnx_fp16`, honored by the retriever as
of this change) and `cloud-init.sh` issues one warm-up query before the droplet is
considered ready — the NAS broker only proxies once `/health` passes. Qdrant warm
search itself was ~66 ms p50 at 30M (see THIN_CLIENT_SWAP.md off-BMP section).

## Bring up (on the droplet / snapshot)
```bash
IDX_VOL=/mnt/idxvol SPLADE_COLLECTION=splade_150m \
  REPO=/opt/sfu bash /opt/sfu/deploy/demo_droplet/cloud-init.sh
```

## Status: authored, NOT yet run on a real droplet
Validated locally: bash/compose syntax, retriever env override + fp16 encode, the
Qdrant SPLADE leg against the **30M** collection. **Live wake time, end-to-end warm
latency, and behaviour at 5–15 concurrent are UNMEASURED** — Phase-5 dry run, which
needs the 150M ingest finished + the volume seeded. The `splade_150m` collection is
still being ingested (do not point a live droplet at it until it reports `green`).
