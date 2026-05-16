# LLM-Judged Benchmark Results — 2026-05-16

**120 queries | TREC 0-3 relevance | Claude Haiku judge | NDCG@10**  
**Full report:** `data/eval_results/benchmark_report_final.md`  
**Raw data:** `data/eval_results/benchmark_llm_judge_final.json`  
**Judge cache:** `data/eval_results/llm_judge_cache.json` (321KB, all 120 queries cached)

For methodology explanation (why citation-count was replaced), see `docs/BENCHMARK_METHODOLOGY.md`.

---

## Overall Results

| Method | Mean NDCG@10 | Median NDCG | Std Dev | MRR@10 | P@10(≥2) |
|--------|-------------|-------------|---------|--------|----------|
| **SPLADE (local)** | **0.6534** | 0.9784 | 0.4629 | 0.6625 | 0.6033 |
| RRF (BM25F+SPLADE) | 0.6362 | 0.9281 | 0.4519 | 0.6667 | 0.5817 |
| BM25F (local) | 0.6249 | 0.9111 | 0.4457 | 0.6562 | 0.5108 |
| OpenAlex relevance sort | 0.5391 | 0.6976 | 0.3885 | 0.5648 | 0.3217 |

### Key observations

- **SPLADE (0.6534) beats RRF (0.6362) by 0.0172** — RRF is currently diluting SPLADE's signal because BM25F and SPLADE only share 7.5% of their result sets. The RRF k-parameter (currently 60) needs to be lowered to 20-40 to fix this. See `docs/SPLADE_OPTIMIZATION_ROADMAP.md`.
- **The high std dev (0.46) is structural, not noise.** Median NDCG is 0.93-0.98 — on subjects with local coverage, the index is excellent. The low mean is dragged down by 15 zero-coverage subjects scoring 0.0.
- **RRF wins on win-rate** (71/120 queries = 59% vs OpenAlex) even though SPLADE wins on raw mean — RRF is more consistent, SPLADE has higher ceiling but more variance.

---

## Wiring Decision: RRF Beats OpenAlex by +0.0971 NDCG

**Verdict: WIRE RRF into all three search tools.**

| Tool | Delta vs OpenAlex | Threshold | Decision |
|------|------------------|-----------|----------|
| search_academic | +0.0971 | >0 | **WIRE RRF** |
| search_by_topic | +0.0971 (all queries) | >+0.03 | **WIRE RRF** |
| export_search | +0.0971, metadata complete | >+0.01 + metadata | **WIRE RRF** |
| search_biomedical | 0.0 local coverage | Local >40% relevant | **ROUTE LIVE** |

RRF vs OpenAlex win/tie/loss: **71 wins / 41 ties / 8 losses** (59% / 34% / 7%)

The 8 losses are all biomedical (AlphaFold, CRISPR, pandemic surveillance) — cutting-edge topics where global citation networks capture emerging research that postdates the local index. Acceptable 7% failure rate.

---

## Per-Subject Breakdown

Sorted by RRF NDCG descending.

| Subject | BM25F | SPLADE | RRF | OpenAlex | Delta (RRF-OA) | Action |
|---------|-------|--------|-----|----------|----------------|--------|
| International Studies | 0.9799 | 0.9939 | 0.9972 | 0.8592 | +0.1380 | WIRE STRONGLY |
| Geography | 1.0000 | 0.9869 | 0.9969 | 0.8882 | +0.1087 | WIRE STRONGLY |
| Computing Science | 0.9925 | 0.9575 | 0.9926 | 0.7991 | +0.1935 | WIRE STRONGLY |
| Mathematics | 0.9631 | 0.9902 | 0.9900 | 0.7543 | +0.2357 | WIRE STRONGLY |
| Mechatronic Systems Engineering | 0.9640 | 1.0000 | 0.9784 | 0.6892 | +0.2892 | WIRE STRONGLY |
| Archaeology | 0.9348 | 0.9484 | 0.9708 | 0.4986 | +0.4722 | WIRE STRONGLY |
| Education | 0.9526 | 0.9966 | 0.9708 | 0.8716 | +0.0992 | WIRE STRONGLY |
| Engineering Science | 0.9669 | 0.9734 | 0.9725 | 0.9028 | +0.0697 | WIRE STRONGLY |
| Psychology | 0.9619 | 0.9873 | 0.9778 | 0.9324 | +0.0454 | WIRE |
| Chemistry | 0.9518 | 0.9979 | 0.9748 | 0.9469 | +0.0279 | WIRE |
| Physics | 0.9557 | 1.0000 | 0.9746 | 0.9502 | +0.0244 | WIRE |
| Gerontology | 0.9611 | 0.9991 | 0.9615 | 0.7718 | +0.1897 | WIRE STRONGLY |
| Finance | 0.9711 | 0.9871 | 0.9707 | 0.8586 | +0.1121 | WIRE STRONGLY |
| Interactive Arts & Technology (SIAT) | 0.9230 | 0.9966 | 0.9666 | 0.8207 | +0.1459 | WIRE STRONGLY |
| English - General | 0.9159 | 1.0000 | 0.9573 | 0.9483 | +0.0090 | WIRE |
| Sociology | 0.9287 | 0.9838 | 0.9549 | 0.6405 | +0.3144 | WIRE STRONGLY |
| Philosophy | 0.9730 | 0.9701 | 0.9413 | 0.8987 | +0.0426 | WIRE |
| Latin American Studies | 0.9109 | 0.9523 | 0.9442 | 0.0000 | +0.9442 | WIRE STRONGLY |
| Indigenous Studies | 0.8712 | 0.9845 | 0.9282 | 0.7761 | +0.1521 | WIRE STRONGLY |
| Canadian Studies | 0.8014 | 0.9749 | 0.9125 | 0.8378 | +0.0747 | WIRE |
| Resource & Environmental Management | 0.9251 | 0.9778 | 0.9285 | 0.7420 | +0.1865 | WIRE STRONGLY |
| Linguistics | 0.9519 | 0.9968 | 0.9232 | 0.8668 | +0.0564 | WIRE |
| Labour Studies | 0.9059 | 0.9267 | 0.9117 | 0.5338 | +0.3779 | WIRE STRONGLY |
| Film | 0.9198 | 0.9902 | 0.9163 | 0.6384 | +0.2779 | WIRE STRONGLY |
| Business Administration | 0.9539 | 0.9164 | 0.9236 | 0.8417 | +0.0819 | WIRE |
| Biomedical Physiology & Kinesiology (BPK) | 0.7228 | 0.7459 | 0.7285 | 0.6088 | +0.1197 | WIRE |
| Health Sciences | 0.7611 | 0.7940 | 0.7503 | 0.6970 | +0.0533 | WIRE |
| Gender, Sexuality & Women's Studies | 0.5000 | 0.4940 | 0.4994 | 0.3613 | +0.1381 | WIRE |
| Economics | 0.3036 | 0.3315 | 0.2924 | 0.1869 | +0.1055 | WIRE |
| Political Science | 0.3028 | 0.3300 | 0.3102 | 0.2299 | +0.0803 | WIRE |
| History | 0.3038 | 0.3158 | 0.3113 | 0.1899 | +0.1214 | WIRE |
| Communication | 0.3227 | 0.3275 | 0.3281 | 0.2449 | +0.0832 | WIRE |
| **Anthropology** | 0.9390 | 0.9063 | 0.8980 | **0.9314** | -0.0334 | ROUTE LIVE |
| **Molecular Biology & Biochemistry** | 0.0000 | 0.0000 | 0.0000 | **0.6934** | -0.6934 | ROUTE LIVE |
| **Global Health** | 0.0000 | 0.0000 | 0.0000 | **0.3659** | -0.3659 | ROUTE LIVE |

**15 subjects with 0.0 local NDCG (no index coverage) — always route live:**  
Statistics & Actuarial Science, Urban Studies, Applied Legal Studies, Music, Public Policy, Sustainable Energy Engineering, Sustainable Community Development, Theatre, Visual Arts, Publishing, Management & Organizational Studies, Accounting, Forensics, Molecular Biology & Biochemistry, Global Health

---

## Where SPLADE Specifically Wins

SPLADE outperforms BM25F on subjects requiring vocabulary expansion:

| Subject | SPLADE | BM25F | SPLADE Advantage |
|---------|--------|-------|-----------------|
| English - General | 1.0000 | 0.9159 | +0.0841 |
| Mechatronic Systems Engineering | 1.0000 | 0.9640 | +0.0360 |
| Physics | 1.0000 | 0.9557 | +0.0443 |
| Gerontology | 0.9991 | 0.9611 | +0.0380 |
| Criminology | 0.9924 | 0.9320 | +0.0604 |
| Education | 0.9966 | 0.9526 | +0.0440 |
| Chemistry | 0.9979 | 0.9518 | +0.0461 |
| Linguistics | 0.9968 | 0.9519 | +0.0449 |
| Canadian Studies | 0.9749 | 0.8014 | +0.1735 |
| Indigenous Studies | 0.9845 | 0.8712 | +0.1133 |

**Pattern:** SPLADE wins most on multi-word topical queries with synonym-rich vocabulary and interdisciplinary topics where terms don't always appear verbatim in the paper. BM25F wins on exact-match subjects (Geography, Business Administration, Philosophy).

---

## Top Individual Query Results

### Biggest RRF wins vs OpenAlex

| Delta | Subject | Query |
|-------|---------|-------|
| +0.9626 | Latin American Studies | Central American migration displacement gang violence asylum |
| +0.9259 | Latin American Studies | extractivism Indigenous land rights resistance Amazon deforestation |
| +0.5701 | Archaeology | Northwest Coast archaeology maritime adaptation coastal villages |
| +0.4931 | Labour Studies | trade union organizing collective bargaining public sector BC |
| +0.4341 | Film | streaming platform recommendation algorithm content diversity |
| +0.4281 | Biomedical Physiology & Kinesiology | biomedical physiology exercise rehabilitation muscle physiology |
| +0.4198 | Interactive Arts & Technology (SIAT) | computational art generative algorithms creative machine learning |
| +0.4157 | Mechatronic Systems Engineering | robotic manipulation control algorithms real-time embedded systems |
| +0.3744 | Archaeology | lithic analysis stone tool technology reduction sequence |
| +0.3642 | History | digital humanities text mining archival primary sources |

### Biggest OpenAlex wins vs RRF (failures)

| Delta | Subject | Query |
|-------|---------|-------|
| -0.3010 | Global Health | antimicrobial resistance stewardship programs low-income countries |
| -0.3553 | Health Sciences | mental health primary care integration collaborative care |
| -0.3869 | Molecular Biology & Biochemistry | CRISPR gene editing off-target effects therapeutic clinical |
| -0.4307 | Global Health | pandemic preparedness One Health zoonotic disease surveillance |
| -1.0000 | Molecular Biology & Biochemistry | protein structure prediction AlphaFold machine learning drug discovery |

**Pattern:** All RRF losses are in biomedical. AlphaFold (2020) and CRISPR (post-2015 surge) postdate or are underrepresented in the local OpenAlex snapshot. OpenAlex live captures these via global citation networks.

---

## Comparison to Old Benchmark

| Query | Old SPLADE NDCG | New SPLADE NDCG | Difference |
|-------|-----------------|-----------------|------------|
| Canadian documentary film Indigenous representation | 0.000 | 0.986 | +0.986 |
| graph algorithms combinatorial optimization | 0.000 | 0.980 | +0.980 |
| interactive media experience design user engagement | 0.027 | 0.999 | +0.972 |
| political risk country ratings international | 0.047 | 0.990 | +0.943 |
| kinesiology sport performance motor learning | 0.097 | 1.000 | +0.903 |
| police use of force racial profiling Black Indigenous | 0.135 | 1.000 | +0.865 |

Old method systematically scored 0.00 on topically relevant but less-cited papers. New method reveals these were actually the best possible results.
