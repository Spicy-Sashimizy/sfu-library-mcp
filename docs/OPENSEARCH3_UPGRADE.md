# OpenSearch 2.19.5 → 3.7.0 Upgrade Runbook

**Status:** ✅ **COMPLETED 2026-06-10** (executed host-side via ClaudeBox dashboard session).
Cold backup taken (300.2 GB tar, verified size), container recreated on 3.7.0, **all
post-upgrade validation passed**: version 3.7.0 / Lucene 10.4.0; cluster green 14/14 shards;
custom-codecs + knn + neural-search at 3.7.0.0; `openalex_works` count 150,413,098 exact;
codec still zstd/3; BM25 read path (302 ms), write/refresh/search/forcemerge/delete smoke,
kNN on `openalex_works_dense` (471 ms) all OK. `two_phase_search_pipeline` created
(prune 0.4 / expansion 5.0 / window 10000) — re-run the NDCG@10 harness from the container.
Backup tarball deleted post-validation per operator instruction (storage reclaimed; rollback
to 2.19 is no longer possible — Lucene 10 segments now on the volume).
**Why:** Lucene 10, derived source for vectors (~3× vector `_source` saving on new indices),
disk-based binary quantization (matches our validated 32× binary+rescore recipe), two-phase
neural sparse (up to ~9.8× SPLADE speedup, usable on the EXISTING rank_features field),
path to SEISMIC sparse-ANN (3.3+ `sparse_vector` field).

## Verified facts (primary sources, 2026-06-10)

- **Direct restart-upgrade 2.19 → 3.x is the supported path** (we are on the terminal 2.x).
- **The 274 GB zstd index WILL open in 3.x**: custom-codecs is bundled in the official
  3.7.0 image and ships `backward_codecs/lucene912/Zstd912Codec` — exactly what 2.19
  wrote. New/merged segments are written as Lucene 10.4 zstd. `index.codec: zstd` stays valid.
- `DISABLE_SECURITY_PLUGIN=true` still honored; add `DISABLE_INSTALL_DEMO_CONFIG=true`;
  no admin password needed with security disabled.
- k-NN: lucene-engine indices keep working (nmslib is the one removed; we don't use it).
  The removed `index.knn.space_type` index setting is NOT present on our indices (verified).
- **Rollback = cold volume copy taken BEFORE first 3.x start.** Lucene 9 cannot read
  Lucene 10 segments; never point 2.19 at a volume 3.x has touched.
- `plugins.query.size_limit`, `http.max_content_length`, `-Xms2g -Xmx2g`: all unchanged.
- Bulk API in 3.x strictly enforces the 512-byte `_id` limit (OpenAlex IDs are far under).
- GPU (cuVS) index builds = separate remote worker service, faiss-engine fp32 only — not
  applicable to the lucene-engine POC index; revisit when building the full-scale dense index.

## Pre-flight audit — PASSED (run 2026-06-10 against the live cluster)

| Check | Result |
|---|---|
| All indices `version.created` ≥ 2.0 (incl. hidden `.tasks`, `.plugins-ml-config`, …) | ✅ all `136408327` (2.19) |
| No `index.knn.space_type` index setting anywhere | ✅ dense index clean (knn:true only) |
| No knn+zstd codec combination on one index | ✅ separate indices |
| No `master`-prefixed settings, no SQL DELETE/scroll reliance, raw-HTTP client only | ✅ |
| Baseline counts | `openalex_works` 150,413,098 · `openalex_works_dense` 600,000 |

## Compose change — DONE (`.devcontainer/docker-compose.yml`)

Image pinned `opensearchproject/opensearch:2` → `3.7.0`; added
`DISABLE_INSTALL_DEMO_CONFIG=true`. Takes effect on next container recreate.

## HOST-SIDE STEPS (operator: run from the host / ClaudeBox, NOT the devcontainer)

```bash
# 0. Make sure no eval/indexer is running against the cluster.

# 1. Flush + stop (quiesce writes first)
curl -s -X POST http://localhost:9200/_flush
docker stop claudebox-sfu-library-mcp-training-opensearch

# 2. COLD BACKUP of the data volume — THE ONLY ROLLBACK PATH (≈25 min for 274 GB; needs 2x headroom)
docker run --rm \
  -v sfu-library-mcp-training_opensearch_data:/src:ro \
  -v /path/with/space:/dst \
  alpine sh -c 'cd /src && tar cf /dst/os219-data-$(date +%F).tar .'

# 3. Recreate with the new image (compose file already updated in repo)
docker compose -f .devcontainer/docker-compose.yml pull opensearch
docker compose -f .devcontainer/docker-compose.yml up -d opensearch
docker logs -f claudebox-sfu-library-mcp-training-opensearch
# watch for: 3.7.0 banner, plugins incl. opensearch-custom-codecs/knn/neural-search,
# shard recovery of openalex_works with NO codec SPI errors (can take minutes).
```

## Post-upgrade validation (can be run from the devcontainer)

1. `GET /` → version 3.7.0, lucene_version 10.x.
2. `GET /_cluster/health?wait_for_status=green&timeout=10m`.
3. `GET /_cat/plugins?v` → custom-codecs, knn, neural-search present at 3.7.0.x.
4. `GET /openalex_works/_count` == 150,413,098; settings still `codec: zstd / level 3`.
5. Read path: BM25F + SPLADE searches via `src/lib/opensearch_retriever.py`; one scroll.
6. Write/merge path: index 1 test doc, refresh, search, delete; `_forcemerge` a SMALL
   test index to confirm Lucene104 zstd writes.
7. kNN path: `dense_search()` against `openalex_works_dense`.
8. `_bulk` smoke from the indexer code path.
9. Re-run `data/eval_results/post_index_benchmark.json` benchmark for regression.
10. Keep the 2.19 tarball for several days before discarding.

## Rollback

```bash
docker compose down opensearch
docker run --rm -v sfu-library-mcp-training_opensearch_data:/dst -v /path/with/space:/src:ro \
  alpine sh -c 'rm -rf /dst/* && tar xf /src/os219-data-<date>.tar -C /dst'
# revert image to opensearchproject/opensearch:2 (or :2.19.5) in compose; up -d
```

## Immediately after a successful upgrade (the free wins)

1. **Enable two-phase neural sparse** on the existing index (no reindex needed):
   ```json
   PUT /_search/pipeline/two_phase_search_pipeline
   { "request_processors": [ { "neural_sparse_two_phase_processor": {
       "enabled": true,
       "two_phase_parameter": { "prune_ratio": 0.4, "expansion_rate": 5.0,
                                 "max_window_size": 10000 } } } ] }
   ```
   Query via `neural_sparse` with `query_tokens` (raw client-side SPLADE vectors work),
   `?search_pipeline=two_phase_search_pipeline`. Then re-run the NDCG@10 harness.
2. **New dense indices get derived source by default** (3.x creation) — plan the
   full-scale dense index with `"mode": "on_disk", "compression_level": "32x"` per the
   validated recipe in `docs/COMPRESSION_EVAL_RESULTS.md`.
3. Apply the lossless lexical config (`docs/COMPRESSION_EVAL_RESULTS.md`) in the same
   reindex that adopts 3.x-native features.

Sources: docs.opensearch.org/latest/breaking-changes/ · opensearch.org/blog/opensearch-3-0-what-to-expect/ ·
github.com/opensearch-project/opensearch-build manifests/3.7.0 · github.com/opensearch-project/custom-codecs
(backward_codecs/lucene912) · docs.opensearch.org/latest/search-plugins/search-pipelines/neural-sparse-query-two-phase-processor/ ·
opensearch.org/blog/do-more-with-less-save-up-to-3x-on-storage-with-derived-vector-source/ ·
docs.opensearch.org/latest/vector-search/optimizing-storage/disk-based-vector-search/ ·
docs.opensearch.org/latest/upgrade-or-migrate/
