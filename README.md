# ferret — local-first semantic file search

Ferret indexes your text, images, and PDF pages into one local SQLite
file and searches it by meaning, keywords, or filename — from the
terminal, from fzf, from Quickshell, or from an agent. One static Rust
binary. No daemon, no cloud, no GUI.

```sh
ferret init --root ~/notes
ferret index
ferret search "pets in the lease"
```

## How it works

1. **Walk** configured roots (gitignore, hidden files, size caps, and
   include/exclude globs honoured; every skip is reported).
2. **Extract** text chunks, image pixels (long edge ≤ `max_edge_px`),
   and PDF pages (text + optional raster per page).
3. **Embed** through an HTTP embedding server (default: local
   `llama-server` running EmbeddingGemma) and store vectors beside an
   FTS5 keyword index in `{data_dir}/ferret/index.sqlite`.
4. **Search** hybrid: vector similarity ∪ keyword ∪ filename, fused
   with RRF, filtered in SQL *before* ranking.

## Requirements

- An embedding server (see below). Ferret never loads model weights
  itself — it only speaks HTTP.
- `poppler-utils` (`pdfinfo`, `pdftotext`, `pdftoppm`) for PDFs.
  Nix wraps them automatically; otherwise install via your package
  manager or Homebrew.
- ImageMagick (`magick`/`convert`) only if you set
  `files.heic = "magick"` for HEIC/HEIF photos.

## Quickstart with Nix

```sh
nix build .#ferret          # ./result/bin/ferret
nix develop                 # rustc, cargo, clippy, rustfmt, poppler
```

Without Nix: `cargo build --release` (needs poppler-utils on `PATH`).

## The embedding server

Ferret needs an OpenAI-compatible `POST {endpoint}/v1/embeddings`
server. The local default is llama.cpp's `llama-server`:

```text
llama-server -m embeddinggemma-2-q8_0.gguf --mmproj mmproj.gguf \
  --embeddings --pooling mean \
  --host 127.0.0.1 --port 8080 -c 4096 -b 4096 -ub 4096
```

Requirements (spec §3.4): llama.cpp with `gemma-embedding2` support,
`--embeddings --pooling mean`, `-c`/`-b`/`-ub` all ≥ your longest
input (2048 text-only, 4096 multimodal), text GGUF plus `--mmproj`
for images/PDF pages. Use the **same** GGUF for index and query —
ferret stores the configured model id string and cannot detect quant
mismatches. Do not vendor GGUFs into this repo.

With `embedder.managed = true`, ferret starts that command itself
(the first time the server is down) from your `embedder.gguf` /
`embedder.mmproj` settings, or runs your `embedder.managed_command`
verbatim. It never shuts the child down. Health is
`GET {endpoint}/health`, falling back to a probe `POST
{v1/embeddings}` with input `"ok"` when `/health` 404s.

### Backends

| `embedder.backend` | Images | Notes |
|---|---|---|
| `llama_server` (default) | yes, with `--mmproj` | v1 default |
| `litert_lm` | yes, if probe succeeds | same HTTP as llama_server |
| `sglang` | yes | `{text, image}` items, `<\|image\|>` marker |
| `openai` | **no** | text-only; image files skip, PDF pages embed as text |

Non-local backends **upload** your file bytes. `ferret init` and
`ferret index` print a stderr warning when the merged backend is
`openai` or `sglang`. Set `embedder.multimodal = true/false` to
override the image probe.

## CLI

```text
ferret [--config FILE] [--index PATH] <command>

init [--root DIR]... [--force]   write starter config.toml (no network)
index [PATH]... [--json] [--strict]
search [QUERY] [filters] [--json] [--mode semantic|keyword|hybrid] [--code]
rm PATH...                       delete rows (missing paths warn, exit 0)
gc                               drop rows whose files are gone
status [--json]                  counts, model, backend, no embedder call
config show | path
```

Exit codes: `0` success (including zero hits, per-file index errors,
and `rm` of a missing path); `1` fatal (config, DB, embedder,
schema/model mismatch, `--strict` with errors); `2` usage.

`ferret index` prints `seen / skipped / embedded / errors` to stderr,
or one NDJSON `{"event":"file",…}` object per file on stdout with
`--json`. Statuses: `ok`, `skip` (with `reason`: `unsupported_ext`,
`skip_rule`, `no_image_backend`, `heic_skip`, `no_magick`,
`gitignore`, `include_globs`), `error` (with message; `complete=0`,
walk continues).

## Search JSON contract (frozen)

`ferret search --json` is the client contract for Quickshell, fzf,
and agents. Quickshell QML intentionally lives **outside** this repo.

```jsonc
{
  "query": "pets in the lease", "task": "search",
  "mode": "hybrid", "limit": 20, "min_score": 0.0,
  "filters": { "kind": [], "ext": [], "under": null, /* … */ },
  "hits": [{
    "path": "/home/nate/notes/lease.md",
    "collection": "notes", "kind": "text", "chunk_kind": "text",
    "ext": "md", "mtime": 1791847599, "ordinal": 0, "page": null,
    "score": 0.0163,
    "scores": { "vec": 0.766, "fts": null, "filename": null },
    "snippet": "The lease forbids pets…"
  }]
}
```

Field notes for client authors:

- `score` is a fused rank weight, **not a percent**. For a 0–1 bar
  use `scores.vec` (cosine-ish similarity) when present.
- `scores.fts` / `scores.filename` are null when that arm did not run
  (e.g. `mode=semantic`, or no keyword terms).
- `--name` is a **case-sensitive** `GLOB` on the basename (even on
  macOS/APFS): `--name '*.MD'` does not match `readme.md`.
- `--under DIR` / `--name GLOB` filter in SQL *before* ranking. With
  very selective filters the semantic arm can starve (zero vector
  candidates): widen the filter or raise `search.candidates`.
- `page` is 1-based for PDF hits, null otherwise. `chunk_kind` is
  `text` | `image` | `pdf_page`.
- No `QUERY` + piped stdin reads one line (never prompts). A bare
  filter with no query text lists matching files (`hybrid` skips the
  vector arm; `mode=semantic` with no text is a usage error).
- fzf example:
  `ferret search --json "$q" | jq -r '.hits[].path' | fzf --preview 'sed -n 1,40p {}'`

## Configuration

Layers, weakest to strongest: compiled defaults → `/etc/ferret/`
→ `{config_dir}/config.toml` → `./.ferret.toml` →
`FERRET__SECTION__KEY` env (e.g. `FERRET__EMBEDDER__BACKEND`) →
`--config` / `--index` flags. `ferret config show` prints the merged
result (`api_key` redacted); `init` never writes `api_key`.

```toml
[index]
roots = ["/home/nate/notes"]   # collection = last path component
# path = "/custom/index.sqlite"

[embedder]
backend = "llama_server"       # llama_server | litert_lm | sglang | openai
endpoint = "http://127.0.0.1:8080"
model = "embeddinggemma-2"
# dimensions = 768             # openai/sglang: sent iff != 768
prefix_profile = "embeddinggemma2"  # or litert_lm (must match backend family)
# multimodal = true            # override the 1x1-PNG image probe
managed = false                # spawn llama-server when down
# gguf = ""  mmproj = ""       # managed-mode weights
# batch_size = 8  timeout_ms = 30000  api_key = ""

[chunk.text]
mode = "sentences"             # or chars
max_chars = 2000
overlap = 200

[chunk.pdf]
max_pages = 5000
first_pass_pages = 30          # scheduler: short docs first
render_dpi = 120
embed_text = true
embed_page_image = true

[chunk.image]
max_edge_px = 1024

[files]
heic = "skip"                  # or "magick" (needs ImageMagick)
max_size = 52428800
# follow_symlinks, include_hidden, respect_gitignore,
# include_globs, exclude_globs, extension_kind …
```

Opening an index whose `model_id`, `dimensions`, or `prefix_profile`
differ from the merged config is a hard error — use a new `--index`
path. Changing text knobs re-embeds text (image vectors are
re-derived byte-identical); changing image knobs re-embeds images and
PDF pages; `images_supported` flips re-embed affected files.

## Paths

- Linux: `~/.config/ferret/`, `~/.local/share/ferret/`,
  `~/.cache/ferret/` (all overridable via `XDG_*`).
- macOS: `~/Library/Preferences/ferret/` for config
  (`dirs` crate); `XDG_*` overrides still win. Default index is
  `{data_dir}/index.sqlite`.

## Testing without a server

`FERRET_EMBEDDER=fake` swaps in a deterministic blake3-based embedder
(any backend): `FERRET_EMBEDDER=fake ferret index`. The test suite
(`cargo test`) uses it plus `httpmock` servers — no network, no GPU.

## NixOS module

```nix
{
  services.ferret = {
    enable = true;
    settings.index.roots = [ "/home/alice/notes" ];
    heicMagick = true;  # default false: adds ImageMagick to PATH
  };
}
```

Writes `/etc/ferret/config.toml`. No daemon — ferret is a CLI; run
`ferret index` from cron/timer yourself.

## Scope (v1 non-goals)

No watcher, GUI, audio/video, OCR, docx/odt/decks, in-process model,
vendored weights, or Ollama backend. See `docs/specs/ferret-design.md`
for the full design and `docs/plans/ferret-v1.md` for the build plan.
