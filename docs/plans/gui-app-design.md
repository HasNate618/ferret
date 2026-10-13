# Ferret GUI App — Engine-Only Design (ferret frozen)

> Source: design subagent (delegate), grounded in `docs/specs/ferret-design.md`, `README.md`, and the v1 progress ledger. Filed for the ferret-gap evaluation.
> Constraint: GUI never patches ferret. It only shells `ferret [--config F] [--index P] <cmd>`, parses `--json`/NDJSON stdout, stderr text, exit codes 0/1/2.

## 1. Product flows

**F1 First-run onboarding:** wizard picks roots, backend, model paths → run `ferret init [--root D]…` (refuse-overwrite unless `--force` after confirm) → `ferret config show` to render merged view → `ferret status --json` (empty index) → probe model server (§2) → start initial `ferret index --json` with progress UI. Warn on `openai`/`sglang` upload (stderr text).
**F2 As-you-type search:** 150ms debounce, cancel in-flight child on new keystroke, `ferret search --json [QUERY] --limit 20 [--mode hybrid] + filters`. Empty-query + filters = browse (vec arm skipped). `mode=semantic` + filter-only = exit 2 → GUI must block that combo. Render `scores.vec` as 0–1 bar; never `%`-render `score` (RRF ~0.01–0.05). `snippet` ≤200 chars or basename fallback; `page` (1-based) + `ordinal` (0-based) for PDF open-at-page.
**F3 Folder-scoped search:** scope bar maps to `--under DIR` (must exist, canonicalized, tilde-expanded; missing=exit 2), `--collection NAME`, `--name GLOB` (case-sensitive `GLOB`, document it), `--kind/--ext` (comma-union), `--after/--before YYYY-MM-DD`. Warn: `--under/--name` starve vec arm (`k=min(candidates*8,400)`, no second KNN pass) — GUI shows "semantic starved, showing keyword" when `scores.vec` all null.
**F4 Background watcher + re-index:** OS watcher (§2) coalesces FS events → debounce 5–30s → `ferret index [CHANGED_PATHS]… --json` (extra-paths inherit collection; off-root paths need unique last-component) → parse NDJSON per-file `ok/skip/error` → `ferret gc` nightly + after deletes/moves. Full `index` (no args) on config-pipeline change or daily.
**F5 Model-server lifecycle:** default `managed=false`: GUI owns `llama-server` subprocess (from `gguf/mmproj/endpoint`), health-gates index/search. If user sets `managed=true`, ferret self-spawns once (never shuts down) — GUI only monitors.

## 2. Architecture

```
GUI (single instance) ─┬─ spawn `ferret search --json` per keystroke (cancelable, 1 concurrent)
                       ├─ queue `ferret index --json` (max 1; searches run concurrent, WAL + busy_timeout 5s)
                       ├─ own: llama-server child (supervise/restart, port from endpoint)
                       └─ own: recursive inotify watcher + scheduler + SQLite read of status (no direct DB writes)
```

**Supervision:** GUI = parent. Ferret children are short-lived, killed via SIGKILL on cancel/replace (`search` is read-only → safe; `index` rolls back open txn, committed files/pages stay, PDFs resume at `next_page`). Model server supervised with backoff; health = `GET {endpoint}/health`, fallback `POST {endpoint}/v1/embeddings {"input":"ok"}` on 404 (mirror ferret). Never run two `index` concurrently (SQLite lock + duplicate embed work).

**Inotify at OS scale:** recursive `inotify` on roots only (not whole `$HOME`); watch dirs, not files; `max_user_watches` check at onboarding (`/proc/sys/fs/inotify/max_user_watches`, typical 524k — 500k-file tree ≈ 30–80k dirs, OK; if exceeded, degrade to 60s mtime poll + manual rescan). Coalesce with 2s quiet-window per dir, batch paths into one `index PATH…` call (argv cap → chunk 500 paths). Ignore ferret's own `cache_dir/pages/*` and `index.sqlite{-wal,-shm}`. `respect_gitignore/include/exclude_globs` stay ferret-side; watcher pre-filters only obvious temp (`*.swp,~*,.git/`).

**Scheduler:** priority queue: user-visible dirs first; PDFs already get ferret-side two-pass (`first_pass_pages=30` then resume). Nightly full `index` catches fingerprint drift, missed events, `complete=0` retries. `gc` after any delete burst.

## 3. Performance budget

| Op | Budget | Notes |
|---|---|---|
| Keystroke → render p50 / p95 | <250ms / <600ms | query embed 60–180ms (local GGUF q8, ctx 2048/4096) + vec KNN 5–30ms + FTS 5–20ms + spawn 5–15ms. Debounce 150ms, kill prior child, cache last 50 query→JSON. Remote `openai/sglang` +150–400ms. |
| Initial index, ~500k files | 6–14h dominated by embedder | Assume ~1M chunks (2/file avg). `batch_size=8`, ~15–40 batches/s text-only → 7–18 chunks/s/W? Realistic ~30–80 files/s text, ~3–8 PDFs/s (pdftotext+pdftoppm+raster+1 embed/page), ~10–30 imgs/s. 500k ≈ 3h crawl/extract + 5–10h embed. |
| Incremental (100 changed files) | <30s | fingerprint fast-path skips `complete=1` unchanged; only changed + `complete=0` resume. |
| Memory/disk | GUI <300MB; index ≈ dims×4B×chunks + FTS (~1–3KB/chunk text) → 1M×768f ≈ 3GB vec + FTS. | Show disk-free gate before full index. |

**Bottlenecks:** (1) embedder HTTP throughput (single `llama-server`, batch 8) — nothing GUI-side fixes it except smaller roots/`max_pages`/`max_chars`; (2) PDF raster (`pdftoppm -r 120` + PNG) CPU/IO; (3) SQLite WAL write amp on per-page PDF txns; (4) process-spawn per keystroke (mitigate: keep `--limit ≤20`, `candidates=50` default).

## 4. Ferret touchpoints per flow

- **F1:** `init [--root]… [--force]` (exit 2 if exists w/o force); `config show` (merged TOML, `api_key="***"`) / `config path`; `status --json` → `{index,model_id,dimensions,backend,images_supported,files{text,image,pdf},chunks,incomplete,errors,indexed_at_max}` (no embedder call). Config keys read: `index.roots/path, embedder.{backend,endpoint,model,dimensions,multimodal,managed,managed_command,gguf,mmproj,batch_size,timeout_ms}, search.*, chunk.{text,pdf,image}, extract.*, files.*`.
- **F2/F3:** `search [QUERY] --json [--mode hybrid|semantic|keyword] [--code] [--kind a,b] [--ext a,b] [--under DIR] [--collection N] [--name GLOB] [--after/--before YYYY-MM-DD] [--min-score F] [--limit N]` + globals `--config/--index`. Prefixes inside QUERY (`kind:/ext:/path:/under:/collection:/name:/after:/before:`, quoted `"My Docs"`, `code:`) — flags win. JSON out: `{query(task-stripped),task,mode,limit,min_score,filters{kind,ext,under,collection,name,after,before},hits[{path,kind,chunk_kind(text|image|pdf_page),collection,ext,page|null,ordinal,mtime,score,scores{vec|null,fts|null,filename|null},snippet}]}`. Stdin: no QUERY + piped stdin = 1 line; TTY + no QUERY/flags = exit 2. Zero hits = exit 0 `hits:[]`.
- **F4:** `index [PATH]… --json [--strict]` → stdout NDJSON `{event:"file",path,status:ok|skip|error,reason,error}`, `reason ∈ {unsupported_ext,skip_rule,no_image_backend,heic_skip,no_magick,gitignore,include_globs,…}`; stderr `seen/skipped/embedded/errors` summary (text only); `rm PATH…` (missing=warn exit 0); `gc` (drop gone rows). Exit 0 with per-file errors; `--strict` → exit 1 (GUI: don't use; parse NDJSON instead).
- **F5:** no ferret status API for server — GUI replicates health check + reads `embedder.{endpoint,managed,managed_command,gguf,mmproj}` from `config show`; `images_supported` + `backend` from `status --json`; 1×1-PNG probe semantics (`multimodal` override) inferred, not reimplemented.

## 5. Failure handling

- **Server down:** `index`/`search` exit 1 + stderr pointing at `endpoint`. GUI: pre-flight health check, banner + "Start server" (spawn `managed_command` or synthesized `llama-server -m {gguf} --mmproj {mmproj} --embeddings --pooling mean -c/-b/-ub 4096`); fallback search `mode=keyword` (no embed) with "semantic offline" chip; queue index until healthy. Retry policy mirror (429/503/reset ×3, 200/400/800ms).
- **Locked/corrupt index:** SQLite busy → backoff + serialize index to 1 slot; corrupt DB (sqlite-vec init fail, schema/model `model_id/dimensions/prefix_profile` mismatch exit 1) → offer new `--index` path or delete sqlite (destructive confirm); never auto-delete.
- **Permission storms:** per-file `error` rows (`complete=0`), walk continues — GUI aggregates NDJSON `status:error` counts, collapses repeats by errno/dir, "exclude" one-click writes `exclude_globs`.
- **Disk full:** abort on SQLite error exit 1; pre-index `statvfs` gate (<5GB warn, <1GB block); `cache_dir/pages` rasters cleaned per-page by ferret, GUI sweeps orphans on crash.
- **Deleted folders:** `index` leaves missing rows; GUI watches `DELETE/MOVE_SELF` → `rm` gone paths + periodic `gc`; root missing → disable scope, keep hits with "missing" badge, offer remove-root.

## 6. Non-goals

No in-GUI extraction/embedding/ranking (ferret owns it); no direct SQLite writes (read `status` only); no audio/video/OCR/docx (ferret walk-skips); no second model/dim in one index; no telemetry/auto-download of GGUFs; no Quickshell-QML-in-ferret; no AUR/deb/brew work.

## 7. Ferret gap list (most important)

| Need | Flow | Workaround | Severity |
|---|---|---|---|
| Single-file cheap upsert: `index PATH` still walks roots; no `index --only PATH` / mtime-manifest API | F4 | batch changed paths, accept full-walk cost; nightly full scan | High |
| Progress %/ETA/chunks in NDJSON (only per-file `ok/skip/error`, summary on stderr text) | F1/F4 | count walk paths first (`find`), divide; fragile | High |
| Cancel-safe resumable `index` token (rely on `complete/next_page` implicitly) | F4 | kill + rerun; parse `incomplete` from `status` | Med |
| Offline keyword search guarantee: unclear if `mode=keyword` skips embedder call | F2/F5 | try keyword on server-down; if exit 1, ship own FTS fallback (duplicates engine) | High |
| No snippet offsets / no `start_byte/end_byte` in v1 hits | F2 | open file at top / PDF at `page`; can't scroll to match | High |
| No thumbnail/preview API (page rasters deleted post-commit) | F2 | GUI re-renders PDFs/images itself (poppler/QL) | Med |
| No collections-list or roots-list JSON (`status` has counts only; `config show` TOML parse needed) | F3 | parse `config show` TOML for `index.roots` | Med |
| No index-run summary JSON (stderr text `seen/skipped/embedded/errors`) | F4 | regex-parse stderr; breaks on wording change | Med |
| No concurrent-index lock error code (only SQLite busy after 5s) | F4 | GUI-side mutex; can't coordinate with cron/CLI users | Med |
| Rename/move = delete+re-embed (fingerprint is path+mtime+size, no content hash) | F4 | accept re-embed cost; `gc`+`index` pair | Med |
| No `search --canceled` / streaming hits (one JSON blob at end) | F2 | kill on new keystroke; no progressive render | Med |
| `--under` must exist + `--name` case-sensitive + vec starvation (cap 400) undiscoverable in JSON | F3 | pre-`stat` dirs; lowercase hint; detect all-null `scores.vec` | Low |
| Date filters date-only (`YYYY-MM-DD`), no timestamp/counts per bucket for faceting | F3 | day-granularity UI only | Low |
| No model-server status endpoint in ferret (`health` logic duplicated in GUI) | F5 | reimplement `GET /health` + `"ok"` probe fallback | Low |
| `api_key` 0600 enforcement kills GUI-spawned runs on mis-chmodded TOML | F1 | pre-check perms, prompt `chmod 600` | Low |
