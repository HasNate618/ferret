# Ferret v1 design

**Date:** 2026-10-12
**Status:** draft for review
**Project:** `/home/nate/Projects/ferret`

Ferret is a Linux and macOS CLI for local multimodal semantic search over files the user chooses. Packaging is **Nix-first**. It embeds nothing itself. EmbeddingGemma 2 (or a compatible HTTP embeddings API) lives in another process. Quickshell, fzf, and agents are clients of `ferret search --json`.

This spec is v1 only. Audio, video, a watcher, a GUI, and an in-process model runner are explicitly out of scope. AUR, deb, and Homebrew casks wait until the binary exists.

---

## 1. Goal

Given folders on disk, ferret must:

1. Index **text** (plain text, markdown, HTML, source files), **images**, and **PDF pages**.
2. Search by meaning and by exact words, with filters applied in SQL before ranking.
3. Talk to a local or remote embeddings HTTP server, defaulting to EmbeddingGemma 2 via `llama-server`.
4. Expose a frozen JSON search contract so a personal Quickshell UI can live in the user’s dots, not in this repo.
5. Build and install with Nix (`flake.nix`) on `x86_64-linux`, `aarch64-linux`, `aarch64-darwin`, and `x86_64-darwin`.

Success for v1: `ferret search --json 'the clause about pets in the lease' --kind pdf` returns the right PDF page from a local corpus, with snippet and score, without ferret loading a GGUF.

### 1.1 Non-goals (v1)

- Hosting or bundling EmbeddingGemma 2, llama.cpp, LiteRT, or PyTorch.
- Audio or video. v1 schema file kinds are only `text|image|pdf`; chunk kinds are only `text|image|pdf_page`.
- A menubar/GUI, Quickshell QML in-tree, or a long-running ferret daemon/socket.
- `inotify` watch / incremental daemon (`watch` is post-v1).
- DigUp-style isolated code index, OCR, Office/LibreOffice decks, docx/odt readers.
- Mixing EmbeddingGemma 1, a different model, or a different dimension into an existing index.
- Telemetry, accounts, or automatic model download (the embedder process owns that).
- An `ollama` embedder backend (different URL/shape: `/api/embed`). Post-v1 if someone wants it.
- Bundling GGUFs, llama.cpp, LiteRT, or ImageMagick in the flake by default.

---

## 2. Architecture

Two processes:

```
folders ──► ferret (crawl, extract, SQLite, search)
              │
              │ HTTP embeddings
              ▼
         llama-server | litert-lm serve | SGLang | OpenAI-compatible API
```

Ferret never maps model weights. If the embedder is down, index and search fail with a clear error pointing at `embedder.endpoint`.

Optional: `embedder.managed = true` lets `index` / `search` spawn `llama-server` when the health check fails. Ferret does not shut it down afterwards (orphans are the operator’s). Default is `managed = false`.

Language: **Rust**, one binary crate named `ferret`, edition 2024. Modules, not a workspace of crates:

| Module | Responsibility | Depends on |
|---|---|---|
| `config` | Merge defaults → files → env → flags; dump merged view | filesystem |
| `embedder` | `EmbedInput` → `Vec<f32>` over HTTP; prefixes; capability probe | `config`, HTTP client |
| `extract` | File bytes → `Chunk`s (text / image / pdf page) | poppler CLI, `image` crate, fs |
| `store` | Schema, WAL, sqlite-vec, FTS5, meta, CRUD | SQLite |
| `index` | Walk, skip, fingerprint, upsert; resume PDFs via `next_page` | extract, embedder, store |
| `search` | Parse query, SQL filters, hybrid RRF, JSON/human | store, embedder |
| `cli` | `init` `index` `search` `rm` `gc` `status` `config` | all of the above |

There is no ferret RPC socket in v1. CLI in, stdout/stderr out.

---

## 3. Embedder

### 3.1 Internal API

Ferret’s embedder trait is **not** “a string to `/v1/embeddings`”. Callers pass a typed input so prefixes and modalities stay consistent.

```text
EmbedInput::Query { text, task }          // default task = search
EmbedInput::Document { title, text }      // title = "none" if unknown
EmbedInput::Image { bytes, mime }
EmbedInput::PdfPage { image_png, text }   // one vector: image part + page text
```

`task` for queries is one of: `search` (default), `code`. `code` is only a **prefix switch** (`task: code retrieval | query:`). There is no second index.

`model_id` in index meta **is** `embedder.model` from the merged config. There is no second name.

Outputs: a single L2-normalized vector of `config.embedder.dimensions` finite `f32`s.

- `openai` / `sglang`: send `"dimensions": <n>` in the JSON body when `n != 768`. If the server rejects it, fail the call (do not silently store a 768-d vector in a 256-d index).
- `llama_server` / `litert_lm`: do not send `dimensions`. If the server returns more dims than configured, truncate to `dimensions` and L2-renormalize. If it returns fewer, or any non-finite, the call is an error.

### 3.2 Task prefixes (EmbeddingGemma 2)

Ferret always applies these on **text**. Media-only inputs get no text prefix. Do not use FP16 anywhere ferret controls. The remote server must run BF16 or quantized GGUF; ferret only stores `f32`.

**Prefix profile.** `embedder.prefix_profile` is `embeddinggemma2` (model card) or `litert_lm` (LiteRT-LM docs). Default: `litert_lm` when `backend = "litert_lm"`, otherwise `embeddinggemma2`. Mixing profiles in one index is a hard error (fingerprint includes the profile). Do not use LiteRT’s search-query string against a llama-server index, or the reverse.

| Role | `embeddinggemma2` (model card) | `litert_lm` |
|---|---|---|
| Search query | `task: search result \| query: {text}` | `task: search query \| text: {text}` |
| Code query (`code:` / `--code`) | `task: code retrieval \| query: {text}` | `task: code retrieval \| query: {text}` |
| Document | `title: {title or none} \| text: {text}` | `task: search result \| text: {text}` |
| Image-only | no text prefix | no text prefix |

PDF page with image:

- `llama_server` / `litert_lm`: page text as a **text content part** next to `image_url`; **no** `<|image|>` in the string (the image part is the marker). Text part uses the Document prefix of the active profile, or empty if `embed_text = false`.
- `sglang`: `text` is `<|image|>\n` plus the Document-prefixed page text (or just `<|image|>` if no text), plus `image`.

Frozen request bodies: [Appendix B](#appendix-b-frozen-http-bodies).

**HTTP is standard for text, profiled for images.** Text uses OpenAI `POST {endpoint}/v1/embeddings` (`model`, `input` as a string or string array, `encoding_format: "float"`). llama-server, `litert-lm serve`, SGLang, and other OpenAI-compatible **text** servers speak it. Images are **not** one public standard: official OpenAI embeddings are text-only. Ferret’s `EmbedInput` is internal; Appendix B maps it onto backend dialects. A random multimodal embeddings API does not work unless it matches a profile. Ollama `/api/embed` is not a v1 backend.

### 3.3 Backends

Config `embedder.backend` is one of:

| Value | Transport | Images | Notes |
|---|---|---|---|
| `llama_server` | `POST {endpoint}/v1/embeddings` | yes, if started with `--mmproj` | **v1 default.** See Appendix B. |
| `litert_lm` | same OpenAI-shaped `/v1/embeddings` | yes, if probe succeeds | Same HTTP client as `llama_server`. |
| `sglang` | `/v1/embeddings` items `{text, image}` | yes | One image per item. Audio `/encode` is unused in v1. |
| `openai` | `/v1/embeddings` strings only | **no** | Image files are capability-skipped. PDF pages embed as `Document` (text only). |

`ollama` is not a v1 value. Do not send Ollama `/api/embed` bodies to `/v1/embeddings`.

Auth: optional `embedder.api_key` or `FERRET__EMBEDDER__API_KEY`. Sent as `Authorization: Bearer`. Empty for local servers. Prefer the env var. `ferret init` does **not** write `api_key`. If a loaded TOML file contains a non-empty `api_key` and its mode is more permissive than `0600`, exit 1.

Image support for a process (`images_supported`):

- `backend = "openai"` → always false (no probe).
- `embedder.multimodal = true` / `false` → that value (no probe).
- `embedder.multimodal` omitted (compiled default) → probe once: embed a 1×1 PNG. HTTP 4xx or transport error on that probe → `images_supported = false`, warn once on stderr. Success → true.

`ferret init` does not write `multimodal`; operators set it when they do not want a probe.

Index meta stores `model_id` (`= embedder.model`), `dimensions`, `backend`, `prefix_profile`, `images_supported` (the value used for this index), `text_pipeline`, `image_pipeline`, and `config_fingerprint`. Opening an index whose `model_id`, `dimensions`, or `prefix_profile` differ from the merged config is a hard error (exit 1): use a new `--index` path, or delete the sqlite file. `ferret rm` only deletes file rows; it does not change meta. A `backend` change with the same `model_id`, `dimensions`, and `prefix_profile` is a stderr warning, not an error.

`images_supported` is part of `config_fingerprint` ([Appendix A](#appendix-a-config-fingerprint)). Indexing without mmproj, then later with mmproj, **must** re-embed affected files. Do not store a capability skip as a finished image index that later `index` runs ignore.

### 3.4 llama-server requirements (local default)

Documented for operators, not implemented inside ferret:

- llama.cpp with `gemma-embedding2` (PR #30054, 2026-10-06 or later).
- `--embeddings --pooling mean`.
- `-c`, `-b`, and `-ub` all ≥ the longest input (bidirectional attention). v1 default assumed context: 2048 for text-only, 4096 when multimodal.
- Text GGUF plus `--mmproj` for images/PDF page images.
- Same GGUF for index and query. Ferret stores `model_id` as the configured id string; it cannot detect quant mismatches.

`embedder.managed = true` runs `embedder.managed_command` if set, otherwise:

```text
llama-server -m {gguf} --mmproj {mmproj} --embeddings --pooling mean
  --host 127.0.0.1 --port {from endpoint} -c 4096 -b 4096 -ub 4096
```

Health check: `GET {endpoint}/health`. If the response is HTTP 404, fall back to `POST {endpoint}/v1/embeddings` with input `"ok"`. Any other 4xx/5xx or connection failure means the embedder is down.

If `managed = true` and `mmproj` is empty, the synthesized command omits `--mmproj`. The image probe then fails (`images_supported = false`).

---

## 4. Extraction

Extractors never embed. They return chunks. Index embeds each chunk.

### 4.1 File kinds

| File kind | Extensions (default) | Chunks |
|---|---|---|
| `text` | `.txt .md .markdown .rst .org .html .htm .csv` plus source: `.rs .go .py .js .ts .tsx .jsx .c .h .cpp .java .sh .toml .yaml .yml .json .nix .lua .css .svg` (svg as text) | Sliding windows of text |
| `image` | `.jpg .jpeg .png .webp .gif .bmp`; `.heic .heif` only when `files.heic = "magick"` | One image chunk |
| `pdf` | `.pdf` | One chunk per page |

Unknown extensions are **walk-skip** (no new `files` row) unless the user adds them via `[files.extension_kind]`. If that path is already in the DB, `index` **deletes** it (same as a skip-rule tombstone).

HTML: strip tags and decode entities; remaining text is `text` chunks. JSON/YAML/TOML: raw UTF-8 as text (no pretty-printer).

Unsupported in v1 (walk-skip, including `.docx .odt .pptx`, rasterizing `.svg`, audio, video, archives): same as unknown extensions. `.svg` is indexed as **text** (source), never as an image. `.heic` / `.heif` are walk-skip when `files.heic = "skip"` (default).

Walk order after extension typing:

1. Skip-rule list (§4.5).
2. If `files.include_globs` is non-empty, the relative path from the collection root must match at least one glob (`*` and `**`). Empty list = all typed files.
3. If `files.respect_gitignore` is true (default) and a `.git` directory exists at or above the file (at or below a configured root), skip paths ignored by that repo’s gitignore, including nested `.gitignore`. Uses the `ignore` crate. A path that is a configured root is never gitignored away.

### 4.2 Text chunking

`[chunk.text]`:

| Key | Default | Meaning |
|---|---|---|
| `mode` | `"chars"` | `"chars"`: sliding windows. `"sentences"`: split on Unicode sentence boundaries, then pack sentences until `max_chars`; if one sentence is longer, fall back to char windows for that sentence. |
| `max_chars` | `2000` | Unicode scalar chars, not bytes. |
| `overlap` | `200` | Overlap for `chars` mode and for over-long sentences. Ignored as a pack overlap in `sentences` mode except on the fallback. |
| `skip_letterless` | `true` | Drop a chunk that contains no Unicode letter (Semantra: page-number footers matching every query). |
| `numeric_heavy` | `0.7` | If the share of ASCII digits among non-whitespace scalars is `>=` this, **do not embed** the chunk: FTS + filename only (`chunks` row, FTS yes, no `chunks_vec` row). `0` disables. DigUp does this for data exports. |

Each chunk stores `start_byte` / `end_byte` in the original file (UTF-8 byte offsets). Title for the document prefix is the basename.

If `overlap >= max_chars`, config error, exit 1. If `mode` is not `chars` or `sentences`, exit 1.

Files whose extracted text is empty (or only whitespace), or whose every chunk is letterless, produce no vec/FTS chunks. The `files` row is still written with `complete = 1`, `error` null, `skip_reason` null, `next_page` null.

`mode`, `max_chars`, `overlap`, `skip_letterless`, and `numeric_heavy` belong to **`text_pipeline`**. Changing them re-chunks and re-embeds **text and PDF-text** only; image vectors stay.

### 4.3 Images

Decode with the Rust `image` crate (`png`, `jpeg`, `gif`, `webp`, `bmp` features). Animated GIF/WebP: **first frame only**. Downscale so the long edge is `chunk.image.max_edge_px` (default 1024), PNG-encode for the embedder. One chunk per file, `text` empty (FTS still indexes filename and path via the FTS `filename`/`path` columns).

`files.heic = "skip"` (default): `.heic` / `.heif` are walk-skip (`reason=heic_skip`). `"magick"`: run `magick` (ImageMagick 7) or `convert` (IM6) to PNG in ferret cache, then the same decode path. Missing binary: **walk-skip** `reason=no_magick` (not extract-error). Conversion fail: extract-error. `heic` and `max_edge_px` belong to **`image_pipeline`**.

Corrupt or undecodable images: **extract-error** (`error` set, `complete = 0`, `skip_reason` null), walk continues.

If `images_supported` is false: **capability-skip**. Write a `files` row with `skip_reason = "no_image_backend"`, `complete = 1`, `error` null, zero chunks. Count as skipped in the index summary, not as an error. Because `images_supported` is in the fingerprint, a later successful probe re-embeds these files.

### 4.4 PDFs

Shell out to poppler. Resolve binaries as: `extract.pdfinfo` / `extract.pdftotext` / `extract.pdftoppm` if set, else `{extract.poppler_dir}/<name>` if `poppler_dir` is set, else `PATH`.

- `pdfinfo` — page count (the size estimate is unused in v1)
- `pdftotext -f N -l N -layout` — page text
- `pdftoppm -f N -l N -png -r {render_dpi} -singlefile` — page image when embedding images, **output under ferret cache** (see below), never next to the corpus PDF

`[chunk.pdf]`: `max_pages = 5000`, `first_pass_pages = 30`, `render_dpi = 120`, `embed_text = true`, `embed_page_image = true`.

Page numbers on chunks are **1-based**, matching PDF viewers.

Temp files: `{cache_dir}/pages/` (see §16). `pdftoppm` writes `{cache}/{files.id}-{page}.png`. Delete that file after the page transaction commits. Never write into the corpus tree.

Fingerprint membership (Appendix A): `embed_text`, `first_pass_pages`, `max_pages` → `text_pipeline.pdf`. `embed_page_image`, `render_dpi` → `image_pipeline.pdf`. Changing only `max_chars` re-writes PDF FTS text and does not re-render page images.

Behavior:

1. **Cursor, not a guessed range.** `files.next_page` is the 1-based next page to attempt (`1` on first sight). After a page is embedded **or** skipped as empty, set `next_page = page + 1`.
2. **Pass order (priority only):** in one `index` run, first bring every PDF up through `min(first_pass_pages, n, next_page…)` so short documents become searchable, then continue each file from its `next_page` through `min(max_pages, n)`. This is scheduling, not a second definition of “remaining pages.”
3. **Resume:** embed page `P` if `P < next_page` is already done; never jump from a failed page 14 to page 31. Also, if `chunks` is missing a page `< next_page` (operator deleted rows), backfill that page before advancing.
4. Page embedding:
   - `embed_page_image` and `images_supported` → `EmbedInput::PdfPage` (one vector per page).
   - else if `embed_text` and the page has text → `EmbedInput::Document` from page text.
   - else skip that page (advance `next_page`; not the whole file).
5. `embed_text = false` with images: `EmbedInput::Image` per page (no document prefix). Still store `pdftotext` output in `chunks.text` when poppler returns any, so FTS can match it.
6. Always store page text in `chunks.text` when `pdftotext` returns non-whitespace, even when the vector is image-only or `PdfPage`.
7. Missing poppler: every PDF is an **extract-error** (`files.error = "pdftotext not found"` or whichever binary is missing, `complete = 0`).
8. `complete = 1` only when `error` is null **and** `next_page > min(max_pages, n)`. If `n > max_pages`, still `complete = 1` after page `max_pages`; stderr warns once per file that remaining pages were not indexed.
9. `start_byte` / `end_byte` are **null** for `pdf_page` chunks.

### 4.5 Skip rules (defaults)

Four outcomes, never mixed:

| Outcome | `files` row | When |
|---|---|---|
| **walk-skip** | none (new); **delete** if already present | unknown/unsupported extension; skip-rule match (§4.5 list) |
| **capability-skip** | yes, `skip_reason` set, `complete = 1`, `error` null, 0 chunks | `no_image_backend` on an image file |
| **extract-error** | yes, `error` set, `complete = 0`, `skip_reason` null | corrupt image, missing poppler, embed HTTP 4xx on this file |
| **ok** | yes, `error` null, `skip_reason` null, `complete` per §4.4/§4.2 | indexed |

Skip-rule list (walk-skip). Name rules apply **even when the configured root is hidden** (e.g. `~/.ssh`):

- Any path component that is `.git`, `node_modules`, `.cache`, `target`, `dist`, `__pycache__`, `.venv`, `venv`
- Hidden path components (name starts with `.`) except the configured root itself — the root may be hidden, children that start with `.` are still skipped — **unless** `files.include_hidden = true`
- Basename matches: `*.pem`, `*.key`, `*.p8`, `*.p12`, `*.pfx`, `.netrc`, `*.netrc`, `.env`, `.env.*`, `credentials.json`, `*.sqlite-journal`
- Basename is `id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`, or matches `id_*` and does **not** end in `.pub`
- Size > `files.max_size` (default `40MiB`)
- User `files.exclude_globs`
- User `files.include_globs` non-empty and no match (§4.1)
- `respect_gitignore` hit (§4.1)

Never follow symlinks unless `files.follow_symlinks = true`. Never **mutate** corpus files (no write, move, rename, or chmod of user content). The only network send is embeddings HTTP to `embedder.endpoint` (extracted text/image bytes). Non-local backends therefore **upload** those bytes; say so in `ferret init` stderr if `backend` is `openai` or `sglang`.

Index NDJSON `status=skip` is emitted for walk-skip and capability-skip, with `"reason"` set (`unsupported_ext`, `skip_rule`, `no_image_backend`, `heic_skip`, `no_magick`, `gitignore`, `include_globs`, …). Walk-skip has `"path"` but no DB id.

---

## 5. Store

Single SQLite file at `index.path` (default `{data_dir}/index.sqlite`; see §16). WAL mode, `busy_timeout = 5000`. Register sqlite-vec in-process via the `sqlite-vec` crate (`sqlite3_vec_init`), not by loading a loose `.so`. If registration fails, exit 1 with the rustc/linker error — there is no runtime “install the extension” path.

Require **SQLite ≥ 3.38** (rusqlite `bundled` feature). Multi-value `--kind` / `--ext` use `file_kind IN (…)` / `ext IN (…)` as vec0 metadata constraints.

### 5.1 Schema (version 1)

`meta` key/value includes: `schema_version=1`, `model_id`, `dimensions`, `backend`, `prefix_profile`, `images_supported`, `config_fingerprint`, `text_pipeline`, `image_pipeline`.

**files**

- `id`
- `path` (absolute, canonical, unique)
- `collection` (unique among roots; see §5.3)
- `kind` (`text|image|pdf`)
- `ext` (no dot, lower)
- `size`, `mtime` (unix seconds)
- `fingerprint`
- `complete` (see §4.4 item 8)
- `next_page` (nullable INTEGER; PDF only; 1-based next page to attempt; null for non-PDF)
- `skip_reason` (nullable TEXT)
- `indexed_at` (unix seconds, UTC)
- `error` (nullable)

**chunks**

- `id`, `file_id` ON DELETE CASCADE
- `ordinal` (INTEGER, **0-based**, contiguous per file in page/byte order)
- `page` (nullable, 1-based; PDF only)
- `kind` (`text|image|pdf_page`) — chunk kind, **never** copied into vec0
- `text` (nullable; empty string stored as null)
- `start_byte` / `end_byte` (UTF-8 byte offsets; **null** for `image` and `pdf_page`)

**chunks_fts** — FTS5, `tokenize='unicode61'`, columns `text`, `filename`, `path`. `rowid = chunks.id`. Maintained by triggers from `chunks` + `files`.

**chunks_vec** — `vec0` with one `FLOAT[{dimensions}]` column and metadata:

- `file_kind TEXT` — copy of **`files.kind`** (`text|image|pdf`), never `chunks.kind`. `--kind pdf` is `WHERE file_kind IN ('pdf')`.
- `ext TEXT`, `mtime INTEGER`, `collection TEXT`

`rowid = chunks.id`. Metadata exists so KNN can take `WHERE file_kind = 'pdf' AND mtime >= ?` **during** the neighbor search, not after.

### 5.2 Fingerprint

`files.fingerprint = blake3(canonical_path || 0x1f || mtime ascii || 0x1f || size ascii || 0x1f || config_fingerprint)` as hex.

`config_fingerprint = blake3(canonical JSON in Appendix A)` as hex. That JSON includes `text_pipeline` and `image_pipeline` objects (not a single blob of every knob).

Reindex granularity (Semantra: changing text chunk size must not re-embed PNGs):

| What changed vs stored meta | Action |
|---|---|
| `model_id`, `dimensions`, or `prefix_profile` | Hard error (unchanged): new `--index` or delete the sqlite file |
| `text_pipeline` only | Delete and re-extract **text** chunks and **PDF page text** / `Document` vectors. Keep `image` chunks and `PdfPage`/`Image` vectors if `image_pipeline` and `images_supported` match |
| `image_pipeline` only | Delete and re-embed **image** files and PDF page rasters (`PdfPage` / page `Image`). Keep text chunks |
| Both, or `extension_kind` / `embed_recipe` / `prefix_version` | Full reindex of affected files (`next_page = 1`) |
| Unchanged fingerprint **and** `complete = 1` | Skip extract and embed |
| Unchanged fingerprint but `complete = 0` | Resume from `next_page` (PDFs) or retry the whole file (text/image extract-error). Do not jump to `first_pass_pages + 1` |

On a text-only pipeline bump for a PDF that used `PdfPage` (image+text in one vector): those page vectors **are** image_pipeline. Do not drop them. Re-write `chunks.text` from `pdftotext` if needed for FTS; leave the vec row.

On an image-pipeline bump for that PDF: delete `pdf_page` vec rows and re-embed pages from `next_page = 1`.

Unchanged `config_fingerprint` is the skip fast-path. When only one pipeline object differs, ferret still walks files of that kind (and PDFs if either PDF-related key changed).

### 5.3 Incremental index and collections

Walk configured roots, then extra paths passed to `ferret index PATH`. For each eligible file: compare fingerprint as above.

**Collection names must be unique.** Default collection for a root is the last component of the canonical path (`~/Documents` → `Documents`). Explicit `[[index.roots]] collection = "…"` overrides. Two roots that would share a name are a config error (exit 1) at the start of `index` / `init`. Extra paths under a configured root inherit that root’s collection. Extra paths not under a root use the last component of that path and error if that name is already a collection.

Missing from disk but in DB: left for `gc`. `index` does not delete missing files. It does delete walked paths that now match walk-skip rules.

Per-file extract-errors set `files.error`, leave `complete = 0`, and do not abort the walk. Fatal (DB, embedder unreachable, schema mismatch) abort with exit 1.

---

## 6. Search

### 6.1 Pipeline

1. Parse CLI flags + typed prefixes out of the query string.
2. Remainder is `nl_query`. A **semantic** copy unquotes `"phrases"` (inner text kept) and removes standalone FTS operators `OR`, `AND`, `NOT`. The keyword arm keeps the remainder including quotes and operators.
3. If the semantic copy is empty (query was only filters and/or operators): skip the vec arm; rank FTS + filename (browse/filter). If `mode=semantic` and the semantic copy is empty: error (exit 2).
4. Apply filters in SQL, not in Rust after fetch. Split:
   - **vec0 metadata (true KNN prefilter):** `file_kind IN (…)` / `ext IN (…)` / `mtime` / `collection`. Requires SQLite ≥ 3.38 for `IN` on TEXT metadata.
   - **JOIN `files` (FTS prefilter; vec post-KNN in the same statement):** `--under` (path prefix) and `--name` (basename glob). sqlite-vec cannot `GLOB`/`LIKE` those during KNN. When either is set, the vec arm uses `k = min(candidates * 8, 400)`, then drops rows that fail the JOIN, then takes up to `candidates`. Returning fewer than `limit` after that is accepted; ferret does not loop a second KNN. The keyword arm is not subject to that cap. `ferret search --help` must state: `--under` / `--name` can starve the semantic arm on a large index; hybrid still has FTS.
5. Each arm keeps at most `search.candidates` (default 50) rows after the step above.
6. Fuse with RRF unless `mode` selects a single arm.
7. Drop fused score `< min_score`. Return `limit` hits (default 20).

### 6.2 Modes

`hybrid` (default) | `semantic` | `keyword`

RRF:

```text
score(id) = Σ_arm  weight_arm / (rrf_k + rank_arm)
```

Default `rrf_k = 60`, `weight_vec = 1.0`, `weight_fts = 1.0`, `weight_filename = 0.4`. Rank is 1-based within that arm’s candidate list. Missing arm → that term is 0.

Filename arm: FTS5 `MATCH` on `filename` and `path` only (not body text), same filters.

Vec distance is L2 on normalized vectors. JSON `scores.vec` is cosine similarity `1 - (distance^2)/2` clamped to `[0,1]`. JSON `scores.fts` is SQLite FTS5 `rank` as returned (typically negative BM25); do not scale it. JSON `scores.filename` is that arm’s FTS5 `rank`. RRF uses 1-based rank in each arm, not these display scores. `scores.*` is `null` when that arm did not list the hit.

JSON `score` is fused RRF (typically ~0.01–0.05). Quickshell must **not** render it as a percent. Use `scores.vec` for a 0–1 cosine bar.

### 6.3 Filters

All of these are AND-combined. Flags and prefixes are equivalent; flags win on conflict.

| Flag | Prefix | Meaning |
|---|---|---|
| `--kind text,image,pdf` | `kind:` / `type:` | File kind, comma-separated |
| `--ext pdf,md` | `ext:` | Extension, no dot |
| `--under DIR` | `path:` / `under:` | Canonical path prefix. Tilde-expand, then `std::fs::canonicalize`. If DIR does not exist, exit 2. |
| `--collection NAME` | `collection:` | Root collection name |
| `--name GLOB` | `name:` | Glob on basename only (`*` and `?`; no `**`). SQL `GLOB`, case-sensitive. |
| `--after YYYY-MM-DD` | `after:` | `mtime >=` that date 00:00:00 local |
| `--before YYYY-MM-DD` | `before:` | `mtime <` next day 00:00:00 local |
| `--min-score F` | | Fused RRF score |
| `--limit N` | | |
| `--mode hybrid` | | |
| `--code` | leading `code:` | Query task prefix only |

`after:` / `before:` parse as naive local calendar dates. Invalid dates are a CLI error (exit 2).

FTS operators (`"phrase"`, `OR`, `NOT`, `AND`) apply only to the keyword arm. `+` / `-` are operators only at the start of the remainder or after whitespace (Semantra: `left-wing` and `COVID-19` are not subtraction). Repeatable flags union their values (`--kind pdf --kind text` equals `--kind pdf,text`).

`--name` uses SQL `GLOB`, **case-sensitive**, even on APFS. Document that. Canonicalize `--under` after tilde-expand; missing DIR is exit 2.

**Prefix tokenizer:** a prefix is a token at the start of the query or after whitespace, `name:` then value. `kind:pdf` and `kind: pdf` are the same. Values run until the next whitespace. Values with spaces must be quoted (`path:"/home/nate/My Documents"`). Unknown prefixes stay in `nl_query` (they are search text, not errors).

`ferret search` with neither a query operand nor any filter flag is usage error (exit 2). `ferret search --kind pdf` (filters only) is valid browse. If QUERY is omitted and stdin is not a TTY, read one query line from stdin; if stdin is a TTY, do not prompt — require QUERY or flags.

### 6.4 JSON contract (frozen)

`ferret search --json` prints one JSON object to stdout, nothing else. Progress and warnings go to stderr.

`query` is the remainder after prefix strip (the semantic copy’s source before FTS-operator removal). v1 hits have **no** `start_byte` / `end_byte`; Quickshell opens `path` (and `page` for PDFs).

```json
{
  "query": "pets in the lease",
  "task": "search",
  "mode": "hybrid",
  "limit": 20,
  "min_score": 0.0,
  "filters": {
    "kind": ["pdf"],
    "ext": [],
    "under": null,
    "collection": null,
    "name": null,
    "after": null,
    "before": null
  },
  "hits": [
    {
      "path": "/home/nate/Documents/lease.pdf",
      "kind": "pdf",
      "chunk_kind": "pdf_page",
      "collection": "Documents",
      "ext": "pdf",
      "page": 12,
      "ordinal": 11,
      "mtime": 1717200000,
      "score": 0.0412,
      "scores": { "vec": 0.81, "fts": -5.12, "filename": -1.04 },
      "snippet": "...pets are not permitted without written consent..."
    }
  ],
  "warnings": []
}
```

`warnings` is always present (possibly empty) with zero or more stable string codes:

- `vec_starved` — a path-scoping filter (`--under` or `--name`) was present AND at least one hit was returned AND zero returned hits have a non-null `scores.vec`. The results are keyword-only; widen the filter or raise `search.candidates`.
- `browse_no_vec` — the query carried no searchable text so only the FTS arm ran (filter-only browse, or `--mode keyword`).

No other codes exist in v1.1. Parsers match codes on exact equality.

- `page` is `null` for non-PDF.
- `ordinal` is 0-based (`page` 12 with pages 1–12 present → `ordinal` 11).
- `snippet` is ≤200 characters from `chunks.text` with the first match window. If `text` is null or empty, snippet is the basename (including image-only and empty `pdf_page`).
- Human mode (default): one hit per line, `path[:page]` then a short snippet. Not stable for clients; Quickshell must use `--json`. Do not print fused RRF as a percent.

### 6.5 Empty results and errors

Zero hits: exit 0, stdout is the JSON object with `"hits": []` when `--json`, or no lines in human mode. Fatal errors (embedder, DB, config): stdout empty, stderr one or more text lines, exit 1. Clients must use the exit code, not scrape stdout for `"error"`.

---

## 7. Config

### 7.1 Layering (last wins)

1. Compiled defaults (this spec)
2. `/etc/ferret/config.toml` if present
3. `{config_dir}/config.toml` (see §16). `--config FILE` **replaces** this layer (it does not stack a second user file). `/etc` and `./.ferret.toml` still apply.
4. `./.ferret.toml` from cwd
5. Environment: `FERRET__SECTION__KEY`, nested with `__`, e.g. `FERRET__EMBEDDER__ENDPOINT=http://127.0.0.1:8080`
6. CLI flags

`--config` pointing at a missing file: exit 1. It is not a silent skip.

Special env, not part of the nested map: `FERRET_EMBEDDER=fake` selects the in-process fake embedder (tests and fixtures). It wins over `embedder.backend`. Fake embedder sets `images_supported = true`.

`ferret config show` prints the merged TOML (comments stripped). It does not print secrets: `api_key` is shown as `"***"` when non-empty. `ferret config path` prints loaded file paths, one per line, then `defaults` / `env` / `flags` if those layers contributed.

### 7.2 Default file (`ferret init` writes a short form; omitted keys keep defaults)

Compiled default `index.roots` is empty. `ferret init` writes an empty `roots` array unless `--root` is given. It does not write `api_key` or `multimodal`. Existing user config: refuse (exit 2) unless `--force`.

Two TOML shapes are allowed; mixing both keys in one file is a config error (exit 1):

```toml
# Shorthand: collection name = last path component (must be unique)
[index]
roots = ["~/Documents", "~/Pictures"]

# Or explicit names (must not also set `roots = [string, ...]`):
# [[index.roots]]
# path = "~/Documents"
# collection = "docs"
```

Illustrative full file (not compiled defaults for `roots` / `multimodal`; no `api_key`):

```toml
[index]
# omit `path` → `{data_dir}/index.sqlite` (§16)
roots = ["~/Documents", "~/Pictures"]

[embedder]
backend = "llama_server"          # llama_server | litert_lm | sglang | openai
endpoint = "http://127.0.0.1:8080"
model = "embeddinggemma-2"
# prefix_profile omitted → embeddinggemma2, or litert_lm when backend is litert_lm
dimensions = 768                   # 768 | 512 | 256 | 128
batch_size = 8
timeout_ms = 120000
# multimodal omitted → probe (openai backend: always false)
managed = false
managed_command = []               # if empty, synthesize llama-server argv
gguf = ""
mmproj = ""

[search]
mode = "hybrid"
rrf_k = 60
weight_vec = 1.0
weight_fts = 1.0
weight_filename = 0.4
default_limit = 20
candidates = 50
min_score = 0.0

[chunk.text]
mode = "chars"                     # chars | sentences
max_chars = 2000
overlap = 200
skip_letterless = true
numeric_heavy = 0.7                # 0 disables FTS-only numeric chunks

[chunk.pdf]
max_pages = 5000
first_pass_pages = 30
render_dpi = 120
embed_text = true
embed_page_image = true

[chunk.image]
max_edge_px = 1024

[extract]
poppler_dir = ""                   # prepended to pdfinfo/pdftotext/pdftoppm if set
pdfinfo = ""
pdftotext = ""
pdftoppm = ""

[files]
follow_symlinks = false
include_hidden = false
respect_gitignore = true
heic = "skip"                      # skip | magick
max_size = "40MiB"
include_globs = []                 # empty = all typed files
exclude_globs = [
  "**/.git/**",
  "**/node_modules/**",
  "**/.cache/**",
  "**/target/**",
]
# extension_kind = { "markdown" = "text" }
```

Tilde expansion uses the process home. `max_size` accepts integer bytes or `KiB`/`MiB`/`GiB`.

`ferret init` creates the user config dir, writes the short file with `roots` from `--root` (repeatable, default empty array), and does not contact the network. If `backend` in compiled defaults is local, no upload warning. If the user later sets `openai`/`sglang`, the next `index` prints the upload warning.

---

## 8. CLI

```text
ferret init [--root DIR]... [--force]
ferret index [PATH]...          # extra paths in addition to configured roots
ferret index --only PATH...     # index exactly these paths, skip the roots walk
                                # (--only with no PATHs: exit 2; watcher recipe:
                                # batch changed paths per run, chunk ~500 client-side)
ferret search [QUERY] [--json] [filters...]
ferret rm PATH...               # delete those paths from the index
ferret gc                       # delete DB rows whose files are gone
ferret status [--json]
ferret config show | path
```

Global flags: `--config FILE` (replaces the user `{config_dir}` layer), `--index PATH` (overrides `index.path`).

`ferret status` human output and `ferret status --json` share these fields (JSON names):

```json
{
  "index": "/home/nate/.local/share/ferret/index.sqlite",
  "model_id": "embeddinggemma-2",
  "dimensions": 768,
  "backend": "llama_server",
  "images_supported": true,
  "roots": [{"path": "/home/nate/docs", "collection": "docs"}],
  "files": { "text": 0, "image": 0, "pdf": 0 },
  "chunks": 0,
  "incomplete": 0,
  "errors": 0,
  "indexed_at_max": null
}
```

No embedder call.

`index` writes a summary to stderr: files seen / skipped / embedded / errors. `--json` on index emits NDJSON events to stdout. The first line is always the plan event with progress totals; per-file events follow:

```json
{"event":"plan","files":2,"bytes":29}
{"event":"file","path":"/abs/path","status":"ok","reason":null,"error":null}
{"event":"file","path":"/abs/path","status":"skip","reason":"no_image_backend","error":null}
{"event":"file","path":"/abs/path","status":"error","reason":null,"error":"pdftotext not found"}
```

`plan.files` counts walk File events only (walk-skips excluded); `plan.bytes` sums walked file sizes. The `summary` counters are the exact values the non-JSON stderr human summary prints (with `--json` the event replaces that stderr line). On `--strict` failure neither summary appears — the run errors out first. Parsers MUST ignore unknown `event` values so future additive events never break them.

Exit codes:

- `0` success (including zero search hits; including index with some per-file errors; including `rm` of a path that was not in the index — warn on stderr)
- `1` fatal (config, DB, embedder, schema/model mismatch, `--config` missing file)
- `2` usage / unknown flag / `init` without `--force` onto an existing file / invalid date / `--under` path does not exist

`index --strict`: per-file errors become exit 1.

---

## 9. Error handling

**Transactions:**

- Text and image files: one SQLite transaction for the whole file (extract + embed + insert).
- PDFs: **one transaction per page** (extract page, embed, insert chunk, bump `next_page`). First-pass vs second-pass is only walk order.
- Embedder down or Ctrl-C: roll back the **open** transaction, then abort the command (exit 1). Already committed pages/files stay. PDFs remain `complete = 0` with `next_page` pointing at the first uncommitted page.
- Do not wrap an entire PDF (all pages) in one transaction.

**Partial file:** if some pages committed and a later page fails with a non-fatal extract/embed error (HTTP 4xx on that payload, poppler page fail): keep committed chunks, set `files.error` to that error, leave `complete = 0`, continue the walk to other files.

**Capability skip:** `status=skip`, `reason=no_image_backend`, not `error`.

HTTP: 3 attempts with linear backoff (200ms, 400ms, 800ms) on 429/503/connection reset. Other 4xx are not retried.

---

## 10. Testing

CI does not download models or call a GPU.

| Area | How |
|---|---|
| Config merge | Fixture files + env + flags; golden merged TOML |
| Query parser | Prefixes (`kind:pdf` vs `kind: pdf`), dates, FTS stripping, `code:`, quoted paths |
| RRF | Stub rankings → exact fused order |
| Filters | In-memory SQLite fixture; `file_kind IN`; `--under` JOIN; assert SQL excludes the wrong kinds/dates |
| `--under` hybrid | Fixture with in-dir and out-dir files; hybrid must return the in-dir hit (FTS if vec is starved) |
| Extract text/html | Tiny files in `testdata/` |
| Extract PDF | Skip if poppler missing; otherwise a 2-page fixture |
| Extract image | Tiny PNG in `testdata/`; GIF first frame |
| Index skip/fingerprint | Fake embedder returns deterministic unit vectors |
| Fingerprint vs probe | Index with `images_supported=false`, flip to true, assert image files re-embed |
| PDF resume | Fail after page 1 of a 3-page PDF; resume embeds page 2 next, not page 31 |
| Collection collision | Two roots ending in `docs` → exit 1 |
| Search JSON | Golden hits against the stub index (`ordinal`, `task`, `limit`) |
| Model mismatch | Opening a 768 index with `dimensions=256` errors |
| `overlap >= max_chars` | Config error |
| `api_key` file mode | Non-0600 with a key → exit 1 |
| Text vs image fingerprint | Bump `chunk.text.max_chars` on a mixed corpus; PNGs keep vectors; text files re-embed |
| `chunk.text.mode=sentences` | Fixture with two sentences → two chunks when `max_chars` would keep them as one |
| `numeric_heavy` | File of digits-only lines → FTS chunks, no vec rows |
| `skip_letterless` | Chunk of punctuation only → dropped |
| Hyphen FTS | Query `left-wing` is not parsed as FTS `-wing` |
| `include_globs` / gitignore | Fixture repo: ignored file walk-skip; glob miss walk-skip |
| Path policy | With `XDG_DATA_HOME` set on Darwin, index default is under that dir |
| `heic=magick` | Skip if `magick` missing; convert fixture if present |

The fake embedder is a trait impl used in tests and `FERRET_EMBEDDER=fake` for local fixture indexing without llama-server.

---

## 11. Build order

1. Cargo binary, `config` merge, CLI stubs, tests for layering.
2. `store` schema (`file_kind`, `next_page`, `skip_reason`) + fake embedder + `search` filters/RRF on fixture rows.
3. Text files + real `llama_server` text embeddings + prefixes.
4. Images (mmproj path) + fingerprint-vs-probe test.
5. PDF pages (text, then interleaved image+text) + resume-cursor test.
6. `openai` + `sglang`/`litert_lm` HTTP profiles and capability flags (Appendix B).
7. Freeze `search --json`; document Quickshell as an external client.

Do not add watch, GUI, audio, or video before step 7 is usable.

---

## 12. Quickshell (out of tree)

Not in this repository. The supported integration is:

```text
ferret search --json "$query" --limit 20
```

plus the filter flags the widget wants to expose. Opening a hit (`xdg-open` on Linux, `open` on macOS, jump to PDF page) is the widget’s job. An `examples/` snippet may be added later; it is not v1.

---

## 13. Decisions already made

| Topic | Choice | Rejected |
|---|---|---|
| Language | Rust | Python (venv/torch as the product), Go |
| Model hosting | External HTTP | In-process llama.cpp / sentence-transformers |
| Local default | `llama-server` + GGUF + mmproj | LiteRT as the only default (still a backend) |
| Index | SQLite + sqlite-vec + FTS5 + RRF | Standalone Qdrant/usearch daemon |
| v1 corpus | Text + images + PDF pages | All four media types from day one; text-only MVP |
| GUI | User’s Quickshell, JSON CLI | In-tree QML |
| Code | Same index, optional `code:` prefix | Separate code index |
| Filters | SQL / vec0 metadata prefilter | Post-filter in Rust after KNN |
| PDF commit | One transaction per page + `next_page` | One transaction per file; guessed remaining range |
| Collection | Unique last-component (or explicit); error on collision | Silent shared name `docs` |
| llama-server PDF payload | image_url part + text part, no `<\|image\|>` in the string | Duplicate Gemma marker in the text |
| HTTP | OpenAI `/v1/embeddings` for text; Appendix B dialects for images | One JSON body for every multimodal server; Ollama `/api/embed` in v1 |
| Prefixes | Model card (`embeddinggemma2`); `litert_lm` profile when that backend is set | LiteRT search-query strings against llama-server |
| Packaging | Nix flake first; wrap poppler; no GGUFs in the flake | AUR/deb in v1; vendored llama.cpp |
| Paths | `dirs` crate; honor `XDG_*` everywhere | Linux-only hardcoded `~/.local/share` |
| HEIC | `skip` default; `magick` optional | In-crate HEIF decoder |
| Chunking | `chars` default; optional `sentences`; letter-less drop; numeric-heavy FTS-only | Token-accurate chunking in v1 |

---

## 14. Operator dependencies (v1)

- Rust toolchain
- rusqlite with `bundled` SQLite ≥ 3.38
- `sqlite-vec` crate **0.1.x**, registered via `sqlite3_vec_init`
- poppler-utils (`pdfinfo`, `pdftotext`, `pdftoppm`) for PDFs
- A running embeddings server for real (non-fake) index/search
- llama.cpp recent enough for `gemma-embedding2` when using the default backend
- Nix (optional): `nix develop` / `nix build` from the repo flake
- ImageMagick `magick` only if `files.heic = "magick"`

---

## 15. Nix

v1 ships a `flake.nix` at the repo root. Ferret is a **Nix-first** CLI, not an afterthought tarball.

### 15.1 Outputs

| Output | What |
|---|---|
| `packages.ferret` (default) | The `ferret` binary. `meta.mainProgram = "ferret"` |
| `devShells.default` | rustc, cargo, clippy, rustfmt, pkg-config, poppler-utils |
| `overlays.default` | `ferret` |
| `nixosModules.ferret` | Writes `/etc/ferret/config.toml` from module options (already a config layer). Does **not** start a daemon |

home-manager: user config file only. No `watch` systemd unit in v1.

Systems: `x86_64-linux`, `aarch64-linux`, `aarch64-darwin`, `x86_64-darwin`.

### 15.2 Wrap and what not to vendor

Wrap the binary so `PATH` includes poppler (`pdfinfo`, `pdftotext`, `pdftoppm`). If `files.heic = "magick"` is in the NixOS module options, also wrap ImageMagick; the default package does **not** pull ImageMagick.

Do **not** vendor GGUFs, llama.cpp, LiteRT, or PyTorch in the flake. The embedder is an operator process. Document `llama-server` as a separate input the user already runs.

`nix build` and `nix develop` are v1. AUR, `.deb`, and Homebrew formulae wait until the binary exists.

---

## 16. macOS and path policy

The engine (CLI, SQLite, HTTP embedder, poppler) is portable. Quickshell is a Linux client. macOS users use `ferret search --json` plus whatever launcher they want (`open` instead of `xdg-open`).

Resolve directories with the `dirs` crate:

| Role | Linux default | macOS default | Override |
|---|---|---|---|
| `config_dir` | `~/.config/ferret` | `~/Library/Preferences/ferret` | `XDG_CONFIG_HOME/ferret` if `XDG_CONFIG_HOME` is set |
| `data_dir` | `~/.local/share/ferret` | `~/Library/Application Support/ferret` | `XDG_DATA_HOME/ferret` if set |
| `cache_dir` | `~/.cache/ferret` | `~/Library/Caches/ferret` | `XDG_CACHE_HOME/ferret` if set |

If any `XDG_*` variable is set (nix-darwin, CI), that variable wins on every OS. `index.path` default is `{data_dir}/index.sqlite`. PDF rasters go under `{cache_dir}/pages/`.

poppler on macOS comes from Homebrew or Nix (`poppler`). No Spotlight, Quick Look, or ImageIO in v1.

`--name` SQL `GLOB` stays **case-sensitive** on APFS. Document it; do not fold case.

`heic = "skip"` by default. `heic = "magick"` requires `magick` or `convert` on `PATH`; missing binary → walk-skip `reason=no_magick`, not an extract-error.

---

## Appendix A. Config fingerprint

UTF-8 JSON, object keys sorted lexicographically, no insignificant whitespace, then `blake3`. Field set is exactly:

```json
{
  "backend": "llama_server",
  "dimensions": 768,
  "embed_recipe": "v1",
  "extension_kind": { "svg": "text" },
  "image_pipeline": {
    "heic": "skip",
    "images_supported": true,
    "max_edge_px": 1024,
    "pdf": {
      "embed_page_image": true,
      "render_dpi": 120
    }
  },
  "model_id": "embeddinggemma-2",
  "prefix_profile": "embeddinggemma2",
  "prefix_version": "v1",
  "text_pipeline": {
    "max_chars": 2000,
    "mode": "chars",
    "numeric_heavy": 0.7,
    "overlap": 200,
    "pdf": {
      "embed_text": true,
      "first_pass_pages": 30,
      "max_pages": 5000
    },
    "skip_letterless": true
  }
}
```

`extension_kind` is the **full** merged map of extension (no dot, lower) → file kind, including every compiled default, not the one-key stub in the example above. `embed_recipe` / `prefix_version` bump when Appendix B or §3.2 change. `prefix_profile` is the merged config value. `images_supported` lives inside `image_pipeline` and is also stored in `meta`.

`files.include_globs`, `include_hidden`, `respect_gitignore`, and `exclude_globs` are **not** in this JSON (they change which paths are walked, not how a given file is chunked). Changing them does not rewrite `config_fingerprint`; existing rows for now-excluded paths are deleted as walk-skips (§4.5).

Implement with a BTreeMap (or equivalent) so key order is sorted, then `serde_json` compact output.

---

## Appendix B. Frozen HTTP bodies

`Content-Type: application/json`. `model` is `embedder.model`. Examples below use `prefix_profile = "embeddinggemma2"`. When the profile is `litert_lm`, substitute the LiteRT strings from §3.2.

### B.1 `llama_server` / `litert_lm` — text query

```json
{
  "model": "embeddinggemma-2",
  "encoding_format": "float",
  "input": "task: search result | query: pets in the lease"
}
```

Document:

```json
{
  "model": "embeddinggemma-2",
  "encoding_format": "float",
  "input": "title: lease.pdf | text: pets are not permitted without written consent"
}
```

### B.2 `llama_server` / `litert_lm` — image only

```json
{
  "model": "embeddinggemma-2",
  "encoding_format": "float",
  "input": [
    {
      "content": [
        {
          "type": "image_url",
          "image_url": { "url": "data:image/png;base64,iVBOR..." }
        }
      ]
    }
  ]
}
```

### B.3 `llama_server` / `litert_lm` — PDF page (`PdfPage`)

Image part first, then text part. **Do not** put `<|image|>` in the text; llama.cpp’s multimodal path already substitutes the image part.

```json
{
  "model": "embeddinggemma-2",
  "encoding_format": "float",
  "input": [
    {
      "content": [
        {
          "type": "image_url",
          "image_url": { "url": "data:image/png;base64,iVBOR..." }
        },
        {
          "type": "text",
          "text": "pets are not permitted without written consent"
        }
      ]
    }
  ]
}
```

Whether `gemma-embedding2` injects Gemma’s `<|image|>` token from the `image_url` part is a llama.cpp behavior. v1 freezes this payload; ranking evals can revisit the marker later without a silent dual implementation.

### B.4 `sglang` — PDF page

(`encoding_format: "float"` is always sent, like B.1/B.5; omitted here
for brevity. The page text carries the active profile's Document prefix;
the example below abbreviates it.)

```json
{
  "model": "google/embeddinggemma-2",
  "encoding_format": "float",
  "input": [
    {
      "text": "<|image|>\ntitle: none | text: pets are not permitted without written consent",
      "image": "data:image/png;base64,iVBOR..."
    }
  ]
}
```

Textless pages send `"text": "<|image|>"` alone. Standalone images
send the same marker item. `dimensions` follows the B.5 rule below.

### B.5 `openai` — text only

```json
{
  "model": "text-embedding-3-small",
  "input": "task: search result | query: pets in the lease",
  "encoding_format": "float"
}
```

No image parts. `dimensions` is sent only when it differs from 768
(the example above omits it per this rule).
