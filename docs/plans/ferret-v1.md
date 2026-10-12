# Ferret v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Nix-first Rust CLI `ferret` that indexes local text, images, and PDF pages into SQLite (sqlite-vec + FTS5), embeds via an external HTTP server (default EmbeddingGemma 2 / llama-server), and serves hybrid search over a frozen `ferret search --json` contract.

**Architecture:** One binary crate, modules not a workspace. Ferret never maps GGUF weights. `config` merges layered TOML; `extract` yields chunks; `embedder` turns `EmbedInput` into L2-normalized `f32` vectors over HTTP (or the fake in-process impl); `store` is one SQLite file with vec0 metadata prefilters; `index` walks, fingerprints, and commits (per-file for text/image, per-page for PDFs); `search` parses prefixes, filters in SQL, fuses RRF, prints JSON or human lines; `cli` is the only user surface.

**Tech Stack:** Rust edition 2024; clap 4.6; rusqlite 0.40 (`bundled`); sqlite-vec 0.1.x (`sqlite3_vec_init`, pin `=0.1.9` or the latest 0.1 that cargo resolves); reqwest 0.13 (`blocking`, `json`, `rustls`); blake3 1; dirs 6; ignore 0.4; globset 0.4; image 0.25; unicode-segmentation 1; serde 1 / toml 0.8 / serde_json 1; chrono 0.4; thiserror 2; Nix flake wrapping poppler-utils.

**Spec:** `docs/superpowers/specs/2026-10-12-ferret-design.md`

## Global Constraints

- Language: **Rust**, one binary crate named `ferret`, edition 2024. Modules, not a workspace of crates.
- Ferret never maps model weights. If the embedder is down, index and search fail pointing at `embedder.endpoint`.
- v1 file kinds: only `text|image|pdf`. Chunk kinds: only `text|image|pdf_page`.
- No in-tree GUI, watcher, daemon/socket, audio, video, OCR, Office readers, Ollama backend, or vendored GGUFs/llama.cpp/LiteRT/ImageMagick (default package).
- Do not use FP16 anywhere ferret controls. Store only finite `f32`. L2-normalize every vector.
- `model_id` in index meta **is** `embedder.model`. Opening an index whose `model_id`, `dimensions`, or `prefix_profile` differ is exit 1.
- Filters apply in SQL before ranking. vec0 metadata: `file_kind`/`ext`/`mtime`/`collection`. `--under`/`--name` JOIN `files` with vec `k = min(candidates*8, 400)`.
- JSON `score` is fused RRF. JSON `scores.vec` is cosine `1 - (distance^2)/2` clamped `[0,1]`. Do not print fused RRF as a percent.
- `--name` SQL `GLOB` is **case-sensitive**, including on APFS.
- Never mutate corpus files. Never write PDF rasters into the corpus tree. Cache under `{cache_dir}/pages/`.
- HTTP: 3 attempts, linear backoff 200ms/400ms/800ms on 429/503/connection reset. Other 4xx are not retried.
- Exit 0: success (zero hits, some per-file index errors, `rm` of a missing path). Exit 1: fatal (config, DB, embedder, schema/model mismatch, `--config` missing file). Exit 2: usage / unknown flag / `init` without `--force` onto an existing file / invalid date / `--under` DIR does not exist.
- SQLite ≥ 3.38 via rusqlite `bundled`. sqlite-vec **0.1.x** registered in-process; no runtime `.so` load.
- `FERRET_EMBEDDER=fake` wins over `embedder.backend`; fake sets `images_supported = true`.
- Path policy: `dirs` crate defaults; if any `XDG_*` is set, that variable wins on **every** OS. macOS `config_dir` default is `~/Library/Preferences/ferret` (not Application Support).
- CI does not download models or call a GPU.
- The repo may not be a git repository yet. Task 1 runs `git init`. Commit steps assume that. If the operator has forbidden commits, skip only the commit step and continue.

---

## File structure

Create these files. Do not add extra crates, a `watch` module, or QML.

| Path | Responsibility |
|---|---|
| `Cargo.toml` / `Cargo.lock` | One package `ferret`, binary + lib |
| `.gitignore` | `/target`, `/result`, `.direnv` |
| `src/main.rs` | `fn main` → `ferret::cli::run()`, map errors to exit 1/2 |
| `src/lib.rs` | Module declarations only |
| `src/error.rs` | `Error` with `exit_code()` → 1 or 2 |
| `src/paths.rs` | `config_dir` / `data_dir` / `cache_dir` (+ XDG override) |
| `src/config.rs` | Structs, compiled defaults, layering, validation, `to_toml` redacting `api_key` |
| `src/fingerprint.rs` | Appendix A canonical JSON + blake3; `text_pipeline` / `image_pipeline` objects |
| `src/cli.rs` | clap: global flags + subcommands; dispatch |
| `src/embedder/mod.rs` | `EmbedInput`, `Embedder` trait, `build_embedder`, health, managed spawn |
| `src/embedder/prefix.rs` | Prefix profiles → prefixed strings |
| `src/embedder/fake.rs` | Deterministic unit vectors from blake3 |
| `src/embedder/http.rs` | Appendix B bodies, retries, truncate+renormalize |
| `src/embedder/probe.rs` | 1×1 PNG probe → `images_supported` |
| `src/extract/mod.rs` | `Chunk` / `Extracted` types; dispatch by `FileKind` |
| `src/extract/text.rs` | UTF-8 read, HTML strip, char/sentence windows, letterless, numeric_heavy |
| `src/extract/image.rs` | `image` crate decode, first GIF frame, downscale, optional magick HEIC |
| `src/extract/pdf.rs` | poppler resolve + per-page text/raster |
| `src/store/mod.rs` | Open, WAL, register vec, schema, meta mismatch, CRUD |
| `src/store/schema.rs` | SQL strings; FTS triggers; vec0 with `FLOAT[{d}]` |
| `src/index/mod.rs` | Walk roots + extra paths; fingerprint; upsert; NDJSON; summary |
| `src/index/walk.rs` | Extension typing, skip rules, gitignore, globs |
| `src/index/skip.rs` | `SkipReason` + NDJSON reason strings |
| `src/search/mod.rs` | Run search: embed query, three arms, RRF, JSON/human |
| `src/search/parse.rs` | Prefix tokenizer, dates, FTS operator strip, hyphen rule |
| `src/search/rrf.rs` | `score = Σ weight / (rrf_k + rank)` |
| `src/search/sql.rs` | Filter SQL: vec0 metadata vs JOIN |
| `tests/common.rs` | Temp HOME/XDG, fake embedder, tiny fixtures helpers |
| `tests/config_merge.rs` | Layering goldens |
| `tests/query_parse.rs` | Prefixes, hyphens, `code:` |
| `tests/search_rrf.rs` | Stub rankings |
| `tests/search_json.rs` | Frozen JSON shape |
| `tests/extract_text.rs` | Chunking modes |
| `tests/extract_image.rs` | PNG + GIF |
| `tests/extract_pdf.rs` | 2-page fixture; skip without poppler |
| `tests/index_pipeline.rs` | Fingerprint split, resume, skip, collections |
| `tests/embedder_http.rs` | httpmock Appendix B |
| `testdata/` | Tiny PNG, GIF, 2-page PDF, html, numeric, letterless |
| `flake.nix` / `flake.lock` | package, devShell, overlay, nixosModule |
| `nix/module.nix` | Writes `/etc/ferret/config.toml` |
| `README.md` | Operator: llama-server argv, JSON client, Nix |

---

### Task 1: Crate scaffold, errors, path policy

**Files:**
- Create: `Cargo.toml`, `.gitignore`, `src/main.rs`, `src/lib.rs`, `src/error.rs`, `src/paths.rs`, `tests/paths.rs`
- Create: `docs/superpowers/plans/2026-10-12-ferret-v1.md` already exists (this file)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ferret::error::{Error, Result}` with `Error::usage(String)`, `Error::fatal(String)`, `fn exit_code(&self) -> i32` (`2` for usage, `1` otherwise)
  - `ferret::paths::{FerretDirs, ferret_dirs}` — `ferret_dirs(env: &dyn Fn(&str) -> Option<String>) -> FerretDirs` where `FerretDirs { config_dir, data_dir, cache_dir: PathBuf }`
  - `pub const INDEX_FILENAME: &str = "index.sqlite";`

- [ ] **Step 1: Write the failing path test**

Create `tests/paths.rs`:

```rust
use ferret::paths::ferret_dirs;
use std::path::PathBuf;

fn env_map<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |k| {
        pairs
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    }
}

#[test]
fn xdg_data_home_wins_on_any_os() {
    let dirs = ferret_dirs(&env_map(&[("XDG_DATA_HOME", "/tmp/xdg-data")]));
    assert_eq!(dirs.data_dir, PathBuf::from("/tmp/xdg-data/ferret"));
}

#[test]
fn xdg_config_and_cache_win() {
    let dirs = ferret_dirs(&env_map(&[
        ("XDG_CONFIG_HOME", "/tmp/xdg-cfg"),
        ("XDG_CACHE_HOME", "/tmp/xdg-cache"),
    ]));
    assert_eq!(dirs.config_dir, PathBuf::from("/tmp/xdg-cfg/ferret"));
    assert_eq!(dirs.cache_dir, PathBuf::from("/tmp/xdg-cache/ferret"));
}

#[test]
fn missing_xdg_uses_platform_defaults() {
    let dirs = ferret_dirs(&|_| None);
    let cfg = dirs.config_dir.to_string_lossy();
    let data = dirs.data_dir.to_string_lossy();
    let cache = dirs.cache_dir.to_string_lossy();
    assert!(cfg.ends_with("ferret"), "{cfg}");
    assert!(data.ends_with("ferret"), "{data}");
    assert!(cache.ends_with("ferret"), "{cache}");
    #[cfg(target_os = "macos")]
    {
        assert!(cfg.contains("Library/Preferences"), "{cfg}");
        assert!(data.contains("Library/Application Support"), "{data}");
        assert!(cache.contains("Library/Caches"), "{cache}");
    }
    #[cfg(target_os = "linux")]
    {
        assert!(cfg.contains(".config/ferret"), "{cfg}");
        assert!(data.contains(".local/share/ferret"), "{data}");
        assert!(cache.contains(".cache/ferret"), "{cache}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nate/Projects/ferret && cargo test --test paths`
Expected: FAIL (no `Cargo.toml` / unresolved crate)

- [ ] **Step 3: Write crate + implementation**

`.gitignore`:

```
/target
/result
.direnv
```

`Cargo.toml` (pin caret ranges; do not add extract/embed crates yet):

```toml
[package]
name = "ferret"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
description = "Local-first semantic multimodal file search"
license = "MIT"
default-run = "ferret"

[dependencies]
thiserror = "2"
dirs = "6"

[dev-dependencies]
```

If `dirs` 7 is what cargo resolves, that is fine. Use `dirs = "6"` or `"7"` — both expose `config_dir`/`data_dir`/`cache_dir`. **Do not** use `dirs::config_dir()` as the macOS config path; spec wants Preferences.

`src/error.rs`:

```rust
use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Usage(String),
    Fatal(String),
}

impl Error {
    pub fn usage(msg: impl Into<String>) -> Self {
        Self::Usage(msg.into())
    }
    pub fn fatal(msg: impl Into<String>) -> Self {
        Self::Fatal(msg.into())
    }
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            Self::Fatal(_) => 1,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(s) | Self::Fatal(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Fatal(e.to_string())
    }
}
```

`src/paths.rs`:

```rust
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FerretDirs {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

pub const INDEX_FILENAME: &str = "index.sqlite";

pub fn ferret_dirs(env: &dyn Fn(&str) -> Option<String>) -> FerretDirs {
    let config_dir = xdg_or(env, "XDG_CONFIG_HOME", platform_config_parent()).join("ferret");
    let data_dir = xdg_or(env, "XDG_DATA_HOME", platform_data_parent()).join("ferret");
    let cache_dir = xdg_or(env, "XDG_CACHE_HOME", platform_cache_parent()).join("ferret");
    FerretDirs {
        config_dir,
        data_dir,
        cache_dir,
    }
}

fn xdg_or(env: &dyn Fn(&str) -> Option<String>, key: &str, fallback: PathBuf) -> PathBuf {
    match env(key) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => fallback,
    }
}

fn platform_config_parent() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        dirs::home_dir()
            .expect("home directory")
            .join("Library/Preferences")
    }
    #[cfg(not(target_os = "macos"))]
    {
        dirs::config_dir().expect("config directory")
    }
}

fn platform_data_parent() -> PathBuf {
    dirs::data_dir().expect("data directory")
}

fn platform_cache_parent() -> PathBuf {
    dirs::cache_dir().expect("cache directory")
}
```

`src/lib.rs`:

```rust
pub mod error;
pub mod paths;
```

`src/main.rs`:

```rust
fn main() {
    eprintln!("ferret: not implemented");
    std::process::exit(1);
}
```

Then `git init` in `/home/nate/Projects/ferret` if `.git` is absent.

- [ ] **Step 4: Run tests**

Run: `cargo test --test paths`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml .gitignore src tests/paths.rs
git commit -m "$(cat <<'EOF'
feat: scaffold ferret crate with XDG-aware path policy

EOF
)"
```

---

### Task 2: Config types, defaults, layering, validation

**Files:**
- Create: `src/config.rs`, `tests/config_merge.rs`, `tests/common.rs`
- Modify: `src/lib.rs` (add `pub mod config;`), `Cargo.toml` (serde, toml, serde_json)

**Interfaces:**
- Consumes: `FerretDirs`, `Error`
- Produces:
  - `config::load(opts: &LoadOpts) -> Result<Config>` (uses real env + filesystem)
  - `config::load_from(opts: &LoadOpts, dirs: &FerretDirs, env: &dyn Fn(&str) -> Option<String>, cwd: &Path, read: &dyn Fn(&Path) -> Result<String>) -> Result<(Config, Vec<LayerSource>)>`
  - `config::compiled_defaults(dirs: &FerretDirs) -> Config`
  - `config::to_display_toml(cfg: &Config) -> String` (`api_key` → `"***"` when non-empty)
  - Types in `src/types.rs` (this task): `FileKind { Text, Image, Pdf }`, `ChunkKind { Text, Image, PdfPage }`, `QueryTask { Search, Code }`, `Backend { LlamaServer, LitertLm, Sglang, Openai }`, `PrefixProfile { EmbeddingGemma2, LitertLm }`, `SearchMode { Hybrid, Semantic, Keyword }`, `TextChunkMode { Chars, Sentences }`, `HeicMode { Skip, Magick }`. `Display`/`FromStr` use the spec strings (`text`, `image`, `pdf`, `pdf_page`, `search`, `code`, `llama_server`, `litert_lm`, `sglang`, `openai`, `embeddinggemma2`, `hybrid`, `chars`, `skip`).

```rust
pub struct Config {
    pub index: IndexConfig,
    pub embedder: EmbedderConfig,
    pub search: SearchConfig,
    pub chunk: ChunkConfig,
    pub extract: ExtractConfig,
    pub files: FilesConfig,
}
pub struct IndexConfig { pub path: PathBuf, pub roots: Vec<RootEntry> }
pub struct RootEntry { pub path: PathBuf, pub collection: String }
pub struct EmbedderConfig {
    pub backend: Backend,
    pub endpoint: String,
    pub model: String,
    pub prefix_profile: PrefixProfile,
    pub dimensions: u32,
    pub batch_size: u32,
    pub timeout_ms: u64,
    pub multimodal: Option<bool>,
    pub managed: bool,
    pub managed_command: Vec<String>,
    pub gguf: String,
    pub mmproj: String,
    pub api_key: String,
}
pub struct SearchConfig {
    pub mode: SearchMode,
    pub rrf_k: f64,
    pub weight_vec: f64,
    pub weight_fts: f64,
    pub weight_filename: f64,
    pub default_limit: usize,
    pub candidates: usize,
    pub min_score: f64,
}
pub struct ChunkConfig {
    pub text: TextChunkConfig,
    pub pdf: PdfChunkConfig,
    pub image: ImageChunkConfig,
}
pub struct TextChunkConfig {
    pub mode: TextChunkMode,
    pub max_chars: usize,
    pub overlap: usize,
    pub skip_letterless: bool,
    pub numeric_heavy: f64,
}
pub struct PdfChunkConfig {
    pub max_pages: u32,
    pub first_pass_pages: u32,
    pub render_dpi: u32,
    pub embed_text: bool,
    pub embed_page_image: bool,
}
pub struct ImageChunkConfig { pub max_edge_px: u32 }
pub struct ExtractConfig {
    pub poppler_dir: String,
    pub pdfinfo: String,
    pub pdftotext: String,
    pub pdftoppm: String,
}
pub struct FilesConfig {
    pub follow_symlinks: bool,
    pub include_hidden: bool,
    pub respect_gitignore: bool,
    pub heic: HeicMode,
    pub max_size: u64,
    pub include_globs: Vec<String>,
    pub exclude_globs: Vec<String>,
    pub extension_kind: BTreeMap<String, FileKind>,
}
```

`LoadOpts`:

```rust
pub struct LoadOpts {
    pub config_flag: Option<PathBuf>, // --config; missing file => Error::fatal
    pub index_flag: Option<PathBuf>,  // --index overrides index.path
}
```

`LayerSource` is an enum: `Defaults`, `Etc`, `UserFile(PathBuf)`, `CwdFile(PathBuf)`, `Env`, `Flags`.

**Root TOML shapes (mixing both in one file is `Error::fatal`):**
- `[index] roots = ["~/Documents"]` → collection = last component after tilde-expand + canonicalize-if-exists, else last component of expanded path
- `[[index.roots]] path = "..." collection = "..."`

**Env:** `FERRET__SECTION__KEY` and nested `__` (`FERRET__EMBEDDER__ENDPOINT`). Not `FERRET_EMBEDDER` (that is special, Task 6).

**Validation (exit 1 / `Error::fatal`):**
- `chunk.text.overlap >= chunk.text.max_chars`
- `chunk.text.mode` not `chars`|`sentences`
- `embedder.dimensions` not in `{768,512,256,128}`
- `embedder.backend` not one of the four
- two roots with the same collection name
- both `roots = [...]` and `[[index.roots]]` in the same file
- unix: a loaded TOML that contains non-empty `api_key` and file mode more permissive than `0600`

**Prefix profile default:** if key omitted, `litert_lm` when `backend = "litert_lm"`, else `embeddinggemma2`.

**`index.path` default:** `{data_dir}/index.sqlite`.

Compiled defaults match spec §7.2: `backend = "llama_server"`, `endpoint = "http://127.0.0.1:8080"`, `model = "embeddinggemma-2"`, `dimensions = 768`, `batch_size = 8`, `timeout_ms = 120000`, `managed = false`, empty `managed_command`/`gguf`/`mmproj`, `multimodal` omitted (`None`), `search.mode = hybrid`, `rrf_k = 60`, weights 1.0/1.0/0.4, `default_limit = 20`, `candidates = 50`, `min_score = 0.0`, text chunk chars/2000/200/skip_letterless true/numeric_heavy 0.7, pdf 5000/30/120/true/true, `max_edge_px = 1024`, empty extract paths, `follow_symlinks = false`, `include_hidden = false`, `respect_gitignore = true`, `heic = skip`, `max_size = 40MiB`, empty `include_globs`, default `exclude_globs` as spec, default `extension_kind` map from spec §4.1.

`max_size` parses integer bytes or `KiB`/`MiB`/`GiB` (1024-based). Store as `u64` bytes.

- [ ] **Step 1: Write failing tests**

`tests/config_merge.rs`:

```rust
use ferret::config::{compiled_defaults, load_from, LoadOpts};
use ferret::paths::FerretDirs;
use ferret::types::PrefixProfile;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn dirs() -> FerretDirs {
    FerretDirs {
        config_dir: PathBuf::from("/tmp/ferret-cfg"),
        data_dir: PathBuf::from("/tmp/ferret-data"),
        cache_dir: PathBuf::from("/tmp/ferret-cache"),
    }
}

#[test]
fn compiled_index_path_uses_data_dir() {
    let c = compiled_defaults(&dirs());
    assert_eq!(c.index.path, PathBuf::from("/tmp/ferret-data/index.sqlite"));
    assert!(c.index.roots.is_empty());
    assert!(matches!(c.embedder.prefix_profile, PrefixProfile::EmbeddingGemma2));
}

#[test]
fn litert_backend_defaults_prefix_profile() {
    let files: HashMap<PathBuf, String> = HashMap::from([(
        PathBuf::from("/tmp/ferret-cfg/config.toml"),
        "[embedder]\nbackend = \"litert_lm\"\n".into(),
    )]);
    let (c, _) = load_from(
        &LoadOpts { config_flag: None, index_flag: None },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|p| files.get(p).cloned().ok_or_else(|| ferret::error::Error::fatal("missing")),
    )
    .unwrap();
    assert!(matches!(c.embedder.prefix_profile, PrefixProfile::LitertLm));
}

#[test]
fn env_overrides_endpoint() {
    let (c, _) = load_from(
        &LoadOpts { config_flag: None, index_flag: None },
        &dirs(),
        &|k| {
            (k == "FERRET__EMBEDDER__ENDPOINT")
                .then(|| "http://127.0.0.1:9999".into())
        },
        Path::new("/tmp/cwd"),
        &|_| Err(ferret::error::Error::fatal("no files")),
    )
    .unwrap();
    assert_eq!(c.embedder.endpoint, "http://127.0.0.1:9999");
}

#[test]
fn overlap_ge_max_chars_is_fatal() {
    let files: HashMap<PathBuf, String> = HashMap::from([(
        PathBuf::from("/tmp/ferret-cfg/config.toml"),
        "[chunk.text]\nmax_chars = 100\noverlap = 100\n".into(),
    )]);
    let err = load_from(
        &LoadOpts { config_flag: None, index_flag: None },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|p| files.get(p).cloned().ok_or_else(|| ferret::error::Error::fatal("missing")),
    )
    .unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
fn missing_config_flag_is_fatal() {
    let err = load_from(
        &LoadOpts {
            config_flag: Some(PathBuf::from("/no/such/ferret.toml")),
            index_flag: None,
        },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|_| Err(ferret::error::Error::fatal("not found")),
    )
    .unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[cfg(unix)]
#[test]
fn api_key_file_mode_0644_is_fatal() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.toml");
    std::fs::write(&p, "[embedder]\napi_key = \"secret\"\n").unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
    let err = load_from(
        &LoadOpts { config_flag: Some(p.clone()), index_flag: None },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|path| std::fs::read_to_string(path).map_err(|e| ferret::error::Error::fatal(e.to_string())),
    )
    .unwrap_err();
    assert_eq!(err.exit_code(), 1);

    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).unwrap();
    let (c, _) = load_from(
        &LoadOpts { config_flag: Some(p), index_flag: None },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|path| std::fs::read_to_string(path).map_err(|e| ferret::error::Error::fatal(e.to_string())),
    )
    .unwrap();
    assert!(ferret::config::to_display_toml(&c).contains("api_key = \"***\""));
}
```

Skippable layers (`/etc/ferret/config.toml`, `{config_dir}/config.toml`, `./.ferret.toml`): `read` returning `Err` means absent — skip. `--config FILE`: `read` `Err` is fatal (exit 1), never a skip.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test config_merge`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement**

Add deps: `serde = { version = "1", features = ["derive"] }`, `toml = "0.8"`, `serde_json = "1"`. Dev-dep: `tempfile = "3"`.

Implement `src/types.rs` (FileKind/ChunkKind/QueryTask/Backend/PrefixProfile/SearchMode/TextChunkMode/HeicMode) with `RenameAll = "snake_case"` / explicit string values: backend `llama_server|litert_lm|sglang|openai`, heic `skip|magick`, mode `chars|sentences`, search `hybrid|semantic|keyword`.

Implement merge last-wins in this order: compiled defaults → `/etc/ferret/config.toml` → `{config_dir}/config.toml` OR `--config FILE` (replace that layer, do not stack) → `./.ferret.toml` → env `FERRET__*` → `--index` into `index.path`.

Tilde-expand roots with `dirs::home_dir()` (`~` and `~/`).

`prefix_profile` omitted: set after backend is known, from the **merged** backend.

Do not write `api_key` in any init path (init is Task 3); only refuse permissive files here.

`tests/common.rs` (Task 11 extends this with `test_env()`):

```rust
use ferret::paths::FerretDirs;
use std::path::PathBuf;

pub fn test_dirs(tmp: &std::path::Path) -> FerretDirs {
    FerretDirs {
        config_dir: tmp.join("cfg"),
        data_dir: tmp.join("data"),
        cache_dir: tmp.join("cache"),
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test config_merge --test paths`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/types.rs src/config.rs src/lib.rs tests/config_merge.rs tests/common.rs
git commit -m "$(cat <<'EOF'
feat: merge layered ferret TOML config with validation

EOF
)"
```

---

### Task 3: CLI stubs — clap, init, config show/path, exit codes

**Files:**
- Create: `src/cli.rs`
- Modify: `src/main.rs`, `Cargo.toml` (clap with `derive`)
- Test: `tests/cli_init.rs`

**Interfaces:**
- Consumes: `config::{load, compiled_defaults, to_display_toml, LoadOpts}`, `paths::ferret_dirs`, `Error`
- Produces: `cli::run() -> Result<()>` (caller in `main` prints `Display` on stderr and `process::exit(err.exit_code())`)
- Produces clap `Cli`: global `--config FILE`, `--index PATH`; subcommands `init { roots: Vec<PathBuf> /* --root, Append */, force: bool }`, `index { paths: Vec<PathBuf>, json: bool, strict: bool }`, `search { query: Option<String>, json: bool, kind: Vec<String>, ext: Vec<String>, under: Option<PathBuf>, collection: Option<String>, name: Option<String>, after: Option<String>, before: Option<String>, min_score: Option<f64>, limit: Option<usize>, mode: Option<SearchMode>, code: bool }`, `rm { paths: Vec<PathBuf> }`, `gc`, `status { json: bool }`, `config { ConfigCmd::Show | Path }`
- Search flags: `--kind`, `--ext`, `--under`, `--collection`, `--name`, `--after`, `--before`, `--min-score`, `--limit`, `--mode`, `--code` (bool). Repeatable `--kind` / `--ext` union values (clap `ArgAction::Append`, then split on comma and merge).
- `ferret search` with neither QUERY nor any filter flag → `Error::usage`. If QUERY omitted and stdin is not a TTY, read one line; if stdin is a TTY, do not prompt.

`main.rs`:

```rust
fn main() {
    if let Err(err) = ferret::cli::run() {
        eprintln!("{err}");
        std::process::exit(err.exit_code());
    }
}
```

- [ ] **Step 1: Write failing CLI tests**

Use `assert_cmd` + `predicates` (add as dev-deps) **or** call `cli::run_with_args(args: &[&str], stdin, is_tty) -> Result<(i32_would, stdout, stderr)>` without spawning. Prefer an internal `run_with` so tests do not need a built binary for every case:

```rust
pub struct RunOutput { pub stdout: String, pub stderr: String }

pub fn try_run(args: &[&str]) -> ferret::error::Result<RunOutput>;
pub fn try_run_with(args: &[&str], stdin: &str, is_tty: bool) -> ferret::error::Result<RunOutput>;

#[test]
fn search_with_no_query_or_filters_is_usage() {
    let err = ferret::cli::try_run(&["ferret", "search"]).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn init_refuses_existing_without_force() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("XDG_CONFIG_HOME", tmp.path());
    ferret::cli::try_run(&["ferret", "init"]).unwrap();
    let err = ferret::cli::try_run(&["ferret", "init"]).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn config_path_prints_defaults_layer() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("XDG_CONFIG_HOME", tmp.path());
    let out = ferret::cli::try_run(&["ferret", "config", "path"]).unwrap();
    assert!(out.stdout.contains("defaults"));
}

#[test]
fn init_writes_roots_and_not_api_key() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("XDG_CONFIG_HOME", tmp.path());
    ferret::cli::try_run(&["ferret", "init", "--root", "/tmp/docs"]).unwrap();
    let body = std::fs::read_to_string(tmp.path().join("ferret/config.toml")).unwrap();
    assert!(body.contains("roots"));
    assert!(!body.contains("api_key"));
    assert!(!body.contains("multimodal"));
}
```

Do not call `set_var` from parallel tests without isolating: add `serial_test` **or** pass dirs into `try_run` via `FERRET_TEST_DIRS` later. Prefer `try_run` constructing `FerretDirs` from `XDG_*` already set in the test process, and mark these tests `#[serial]` (`serial_test = "3"` dev-dep).

`init` behavior: create `{config_dir}` (0700), write short TOML with `roots` from `--root` (repeatable), refuse if file exists unless `--force`. Do **not** write `api_key` or `multimodal`. Do not hit the network. If compiled/default backend is local, no upload warning.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test cli_init`
Expected: FAIL

- [ ] **Step 3: Implement clap + init + config show/path**

Other subcommands may return `Error::fatal("not implemented")` **only if** tests do not call them yet. Prefer empty stubs that return usage-ok no-ops for `gc`/`rm` so Task 13 can fill them. `index`/`search`/`status` wait for later tasks.

`config show`: merged TOML, comments stripped, `api_key` redacted.

`config path`: loaded file paths one per line, then `defaults` / `env` / `flags` if those layers contributed.

`--help` for search must include: `--under` / `--name` can starve the semantic arm on a large index; hybrid still has FTS. `--name` is case-sensitive GLOB.

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/cli.rs src/main.rs tests/cli_init.rs
git commit -m "$(cat <<'EOF'
feat: add ferret CLI scaffolding, init, and config show

EOF
)"
```

---

### Task 4: Appendix A config fingerprint

**Files:**
- Create: `src/fingerprint.rs`
- Test: unit tests in `src/fingerprint.rs` under `#[cfg(test)]`
- Modify: `Cargo.toml` (`blake3 = "1"`), `src/lib.rs`

**Interfaces:**
- Consumes: `Config`, `images_supported: bool`
- Produces:
  - `fingerprint::config_fingerprint(cfg: &Config, images_supported: bool) -> String` (hex blake3)
  - `fingerprint::config_fingerprint_json(cfg: &Config, images_supported: bool) -> String` (canonical JSON)
  - `fingerprint::file_fingerprint(canonical_path: &str, mtime: i64, size: u64, config_fingerprint: &str) -> String`
  - `fingerprint::text_pipeline_json` / `image_pipeline_json` helpers used by store meta
  - Constants `EMBED_RECIPE: &str = "v1";` `PREFIX_VERSION: &str = "v1";`

Canonical JSON: UTF-8, object keys sorted lexicographically, no insignificant whitespace. Implement by building `serde_json::Value::Object(Map)` from a `BTreeMap`, then `to_string()` (serde_json compact).

Field set **exactly** as Appendix A. `extension_kind` is the **full** merged map (no-dot, lower). `files.include_globs` / `include_hidden` / `respect_gitignore` / `exclude_globs` are **not** in this JSON.

`file_fingerprint = blake3(canonical_path || 0x1f || mtime ascii || 0x1f || size ascii || 0x1f || config_fingerprint)` hex.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn canonical_json_key_order_and_split_pipelines() {
    let cfg = compiled_defaults(&test_dirs());
    let json = config_fingerprint_json(&cfg, true);
    assert!(json.contains("\"image_pipeline\":{"));
    assert!(json.contains("\"text_pipeline\":{"));
    assert!(!json.contains("include_globs"));
    assert!(json.contains("\"prefix_profile\":\"embeddinggemma2\""));
    // image_pipeline.pdf has embed_page_image + render_dpi only
    assert!(json.contains("\"embed_page_image\":true"));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let img_pdf = &v["image_pipeline"]["pdf"];
    assert!(img_pdf.get("embed_text").is_none());
    assert!(img_pdf.get("max_pages").is_none());
    let txt_pdf = &v["text_pipeline"]["pdf"];
    assert!(txt_pdf.get("embed_page_image").is_none());
    assert_eq!(txt_pdf["max_pages"], 5000);
}

#[test]
fn fingerprint_is_stable_hex() {
    let cfg = compiled_defaults(&test_dirs());
    let a = config_fingerprint(&cfg, true);
    let b = config_fingerprint(&cfg, true);
    assert_eq!(a, b);
    assert_eq!(a.len(), 64);
    assert_ne!(a, config_fingerprint(&cfg, false));
}

#[test]
fn file_fingerprint_includes_unit_separator() {
    let fp = file_fingerprint("/a/b.txt", 1, 2, "abcd");
    let expected = blake3::hash(b"/a/b.txt\x1f1\x1f2\x1fabcd").to_hex().to_string();
    assert_eq!(fp, expected);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib fingerprint`
Expected: FAIL

- [ ] **Step 3: Implement `src/fingerprint.rs`**

Build the JSON object with keys: `backend`, `dimensions`, `embed_recipe`, `extension_kind`, `image_pipeline`, `model_id`, `prefix_profile`, `prefix_version`, `text_pipeline`.

`image_pipeline`: `{ heic, images_supported, max_edge_px, pdf: { embed_page_image, render_dpi } }`

`text_pipeline`: `{ max_chars, mode, numeric_heavy, overlap, pdf: { embed_text, first_pass_pages, max_pages }, skip_letterless }`

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/fingerprint.rs src/lib.rs
git commit -m "$(cat <<'EOF'
feat: hash split text/image pipeline fingerprints

EOF
)"
```

---

### Task 5: Store schema, meta, CRUD

**Files:**
- Create: `src/store/mod.rs`, `src/store/schema.rs`
- Modify: `Cargo.toml` — `rusqlite = { version = "0.40", features = ["bundled"] }`, `sqlite-vec = "=0.1.9"`
- Test: `tests/store_schema.rs`

**Interfaces:**
- Consumes: `Config`, fingerprints, `FileKind`, `ChunkKind`, `Error`
- Produces:
  - `store::open(path: &Path, cfg: &Config, images_supported: bool) -> Result<Store>`
  - `Store { conn: rusqlite::Connection, dimensions: u32 }` with methods:
    - `get_meta(&self, key: &str) -> Result<Option<String>>`
    - `upsert_file(&self, row: &NewFile) -> Result<i64>`
    - `get_file_by_path(&self, path: &str) -> Result<Option<FileRow>>`
    - `delete_file_by_path(&self, path: &str) -> Result<bool>`
    - `insert_chunk(&self, tx: &rusqlite::Transaction<'_>, chunk: NewChunk) -> Result<i64>`
    - `insert_vec(&self, tx: &rusqlite::Transaction<'_>, chunk_id: i64, embedding: &[f32], file_kind: FileKind, ext: &str, mtime: i64, collection: &str) -> Result<()>`
    - `delete_text_chunks_keep_images(&self, file_id: i64) -> Result<()>`
    - `delete_image_chunks_keep_text(&self, file_id: i64) -> Result<()>`
    - `status_counts(&self) -> Result<StatusCounts>`
    - `begin(&mut self) -> Result<rusqlite::Transaction<'_>>`

```rust
pub struct NewFile {
    pub path: String, pub collection: String, pub kind: FileKind, pub ext: String,
    pub size: u64, pub mtime: i64, pub fingerprint: String,
    pub complete: bool, pub next_page: Option<i64>,
    pub skip_reason: Option<String>, pub indexed_at: i64, pub error: Option<String>,
}
pub struct FileRow {
    pub id: i64, pub path: String, pub collection: String, pub kind: FileKind, pub ext: String,
    pub size: u64, pub mtime: i64, pub fingerprint: String,
    pub complete: bool, pub next_page: Option<i64>,
    pub skip_reason: Option<String>, pub indexed_at: i64, pub error: Option<String>,
}
pub struct NewChunk {
    pub file_id: i64, pub ordinal: i64, pub page: Option<i64>,
    pub kind: ChunkKind, pub text: Option<String>,
    pub start_byte: Option<i64>, pub end_byte: Option<i64>,
}
pub struct StatusCounts {
    pub files_text: u64, pub files_image: u64, pub files_pdf: u64,
    pub chunks: u64, pub incomplete: u64, pub errors: u64,
    pub indexed_at_max: Option<i64>,
}
```

Register sqlite-vec **before** opening connections. Prefer:

```rust
unsafe {
    rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
        sqlite_vec::sqlite3_vec_init as *const (),
    )));
}
```

If `rusqlite::ffi` is private in 0.40, use `libsqlite3-sys` at the same version rusqlite pins, or `rusqlite::auto_extension::register_auto_extension` if the `RawAutoExtension` signature matches. Registration failure → `Error::fatal` with the rustc/linker error. Verify with `select vec_version();` and `sqlite_version()` ≥ 3.38.0.

On open: `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;`

If the file is new: run `schema::ddl(dimensions)`, write meta: `schema_version=1`, `model_id`, `dimensions`, `backend`, `prefix_profile`, `images_supported`, `config_fingerprint`, `text_pipeline`, `image_pipeline` (pipeline values = canonical JSON objects from Task 4).

If the file exists: read meta. If `model_id`, `dimensions`, or `prefix_profile` differ from merged config → `Error::fatal` telling the user to pass a new `--index` or delete the sqlite file. If `backend` differs but model/dim/prefix match → `eprintln!` warning, not an error. `ferret rm` must not change meta (enforced by not offering a method that does).

**DDL** (`src/store/schema.rs`): tables `meta`, `files`, `chunks` as spec §5.1. FTS5 `chunks_fts(text, filename, path, tokenize='unicode61')` with `content='chunks'` **or** external-content off and triggers that also copy filename/path from `files`. Register scalar `ferret_basename(path)` via `conn.create_scalar_function` so triggers can fill `filename`. `chunks_vec` using `vec0(embedding FLOAT[{d}], file_kind TEXT, ext TEXT, mtime INTEGER, collection TEXT)`. Inserts use `rowid = chunks.id`. **`file_kind` is `files.kind`, never `chunks.kind`.**

Store empty string text as SQL NULL.

DDL in `schema::ddl(dimensions: u32) -> String` (interpolate only `{d}` into `FLOAT[{d}]`, never concat user paths):

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  collection TEXT NOT NULL,
  kind TEXT NOT NULL,
  ext TEXT NOT NULL,
  size INTEGER NOT NULL,
  mtime INTEGER NOT NULL,
  fingerprint TEXT NOT NULL,
  complete INTEGER NOT NULL,
  next_page INTEGER,
  skip_reason TEXT,
  indexed_at INTEGER NOT NULL,
  error TEXT
);
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  page INTEGER,
  kind TEXT NOT NULL,
  text TEXT,
  start_byte INTEGER,
  end_byte INTEGER
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  text, filename, path, tokenize='unicode61'
);
CREATE VIRTUAL TABLE chunks_vec USING vec0(
  embedding FLOAT[{d]},
  file_kind TEXT,
  ext TEXT,
  mtime INTEGER,
  collection TEXT
);
```

Triggers (after insert/update/delete on `chunks`; also rewrite FTS when `files.path` changes): `INSERT INTO chunks_fts(rowid, text, filename, path) SELECT new.id, new.text, ferret_basename(f.path), f.path FROM files f WHERE f.id = new.file_id`. Delete uses `INSERT INTO chunks_fts(chunks_fts, rowid) VALUES('delete', old.id)`. Register `ferret_basename` via `conn.create_scalar_function("ferret_basename", 1, …)`.

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn opens_and_registers_vec() {
    let store = open_temp(768);
    let v: String = store
        .conn()
        .query_row("select vec_version()", [], |r| r.get(0))
        .unwrap();
    assert!(v.starts_with('v'));
}

#[test]
fn model_mismatch_is_fatal() {
    let path = tmp_index();
    let cfg768 = cfg_with_dims(768);
    store::open(&path, &cfg768, true).unwrap();
    let cfg256 = cfg_with_dims(256);
    let err = store::open(&path, &cfg256, true).unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
fn prefix_profile_mismatch_is_fatal() {
    let path = tmp_index();
    let mut cfg = cfg_with_dims(768);
    store::open(&path, &cfg, true).unwrap();
    cfg.embedder.prefix_profile = PrefixProfile::LitertLm;
    let err = store::open(&path, &cfg, true).unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
fn file_kind_metadata_is_files_kind_not_chunk_kind() {
    let store = open_temp(8);
    let mut tx = store.begin().unwrap();
    let fid = store.upsert_file(&NewFile {
        path: "/a/lease.pdf".into(), collection: "Documents".into(),
        kind: FileKind::Pdf, ext: "pdf".into(), size: 1, mtime: 1,
        fingerprint: "x".into(), complete: true, next_page: Some(3),
        skip_reason: None, indexed_at: 1, error: None,
    }).unwrap();
    let cid = store.insert_chunk(&tx, NewChunk {
        file_id: fid, ordinal: 0, page: Some(1), kind: ChunkKind::PdfPage,
        text: Some("pets".into()), start_byte: None, end_byte: None,
    }).unwrap();
    let mut v = vec![0.0_f32; 8];
    v[0] = 1.0;
    store.insert_vec(&tx, cid, &v, FileKind::Pdf, "pdf", 1, "Documents").unwrap();
    tx.commit().unwrap();
    let kind: String = store.conn().query_row(
        "select file_kind from chunks_vec where rowid = ?",
        [cid],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(kind, "pdf");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test store_schema`
Expected: FAIL

- [ ] **Step 3: Implement store**

Keep `conn` private except a `#[cfg(test)]` accessor if tests need it. Do not load a loose `.so`.

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/store src/lib.rs tests/store_schema.rs
git commit -m "$(cat <<'EOF'
feat: add SQLite store with sqlite-vec and FTS5 schema

EOF
)"
```

---

### Task 6: Embedder trait, prefixes, fake embedder

**Files:**
- Create: `src/embedder/mod.rs`, `src/embedder/prefix.rs`, `src/embedder/fake.rs`
- Modify: `src/lib.rs`, `src/config.rs` (read `FERRET_EMBEDDER`)
- Test: `src/embedder/prefix.rs` unit tests + `tests/embedder_fake.rs`

**Interfaces:**
- Consumes: `Config`, `QueryTask`, `PrefixProfile`
- Produces:

```rust
pub enum EmbedInput {
    Query { text: String, task: QueryTask },
    Document { title: Option<String>, text: String },
    Image { bytes: Vec<u8>, mime: String },
    PdfPage { image_png: Vec<u8>, text: Option<String> },
}

pub trait Embedder: Send + Sync {
    fn embed(&self, input: EmbedInput) -> Result<Vec<f32>>;
    fn embed_batch(&self, inputs: &[EmbedInput]) -> Result<Vec<Vec<f32>>> {
        inputs.iter().cloned().map(|i| self.embed(i)).collect()
    }
    fn images_supported(&self) -> bool;
}

pub fn build_embedder(cfg: &Config, env: &dyn Fn(&str) -> Option<String>) -> Result<Box<dyn Embedder>>;
pub fn apply_prefix(profile: PrefixProfile, input: &EmbedInput) -> Option<String>;
pub fn l2_normalize(v: Vec<f32>) -> Result<Vec<f32>>;
pub fn truncate_or_check(v: Vec<f32>, backend: Backend, dimensions: u32) -> Result<Vec<f32>>;
```

`apply_prefix` returns `None` for image-only (no text prefix). Document prefix uses `"none"` when title is `None`.

**embeddinggemma2:**
- Query search: `task: search result | query: {text}`
- Query code: `task: code retrieval | query: {text}`
- Document: `title: {title or none} | text: {text}`

**litert_lm:**
- Query search: `task: search query | text: {text}`
- Query code: `task: code retrieval | query: {text}`
- Document: `task: search result | text: {text}`

Fake embedder: `images_supported = true`. Vector = first `dimensions` bytes of `blake3(prefix || 0x1f || payload)` interpreted as `u8 as f32`, then L2-normalize. Payload is UTF-8 text or raw image bytes. Always finite. `build_embedder` if `env("FERRET_EMBEDDER") == Some("fake")` returns fake **regardless of backend**.

HTTP embedder in this task: **not yet**. `build_embedder` without fake may return `Error::fatal("http embedder not implemented")` until Task 12.

`truncate_or_check`: openai/sglang must already be `dimensions` (caller sends `dimensions` in JSON later). llama_server/litert_lm: if `len > dimensions`, truncate then L2-renormalize; if `len < dimensions` or any non-finite → error.

- [ ] **Step 1: Write failing prefix tests**

```rust
#[test]
fn gemma2_search_query_prefix() {
    let s = apply_prefix(
        PrefixProfile::EmbeddingGemma2,
        &EmbedInput::Query { text: "pets".into(), task: QueryTask::Search },
    );
    assert_eq!(s.as_deref(), Some("task: search result | query: pets"));
}

#[test]
fn litert_search_query_prefix() {
    let s = apply_prefix(
        PrefixProfile::LitertLm,
        &EmbedInput::Query { text: "pets".into(), task: QueryTask::Search },
    );
    assert_eq!(s.as_deref(), Some("task: search query | text: pets"));
}

#[test]
fn image_only_has_no_text_prefix() {
    let s = apply_prefix(
        PrefixProfile::EmbeddingGemma2,
        &EmbedInput::Image { bytes: vec![1], mime: "image/png".into() },
    );
    assert!(s.is_none());
}

#[test]
fn fake_is_unit_length_and_deterministic() {
    let e = FakeEmbedder::new(8);
    let a = e.embed(EmbedInput::Document { title: Some("a.txt".into()), text: "hi".into() }).unwrap();
    let b = e.embed(EmbedInput::Document { title: Some("a.txt".into()), text: "hi".into() }).unwrap();
    assert_eq!(a, b);
    let n = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((n - 1.0).abs() < 1e-5);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib embedder`
Expected: FAIL

- [ ] **Step 3: Implement prefix + fake + normalize**

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedder tests/embedder_fake.rs src/lib.rs
git commit -m "$(cat <<'EOF'
feat: add embedder prefixes and fake unit-vector backend

EOF
)"
```

---

### Task 7: Search query parser

**Files:**
- Create: `src/search/mod.rs`, `src/search/parse.rs`
- Modify: `src/lib.rs`
- Test: `tests/query_parse.rs`

**Interfaces:**
- Consumes: `FileKind`, `QueryTask`, `SearchMode`, `Error`
- Produces:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Filters {
    pub kind: Vec<FileKind>,
    pub ext: Vec<String>,        // no dot, lower
    pub under: Option<PathBuf>,  // canonical; missing DIR => Error::usage (exit 2)
    pub collection: Option<String>,
    pub name: Option<String>,    // glob, as given
    pub after: Option<i64>,      // unix seconds local 00:00:00
    pub before: Option<i64>,     // next day 00:00:00 local
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedQuery {
    pub nl_query: String,         // remainder after prefix strip (JSON `query`)
    pub semantic_query: String,   // unquote phrases, drop standalone OR/AND/NOT
    pub keyword_query: String,    // remainder including quotes and operators
    pub task: QueryTask,
    pub filters: Filters,
}

pub fn parse_query(raw: &str, flag_overrides: &FilterFlags) -> Result<ParsedQuery>;
```

`FilterFlags` holds the clap-parsed flags. Flags win on conflict with prefixes. Repeatable flags union. `code:` at start of remainder or `--code` sets `task = Code`.

**Prefix tokenizer:** token at start or after whitespace, `name:` then value. `kind:pdf` and `kind: pdf` are the same. Values until next whitespace. Quoted values allowed (`path:"/home/nate/My Documents"`). Known prefixes: `kind`/`type`, `ext`, `path`/`under`, `collection`, `name`, `after`, `before`, `code`. Unknown prefixes stay in `nl_query`.

**Hyphens:** `+` / `-` are FTS operators only at the start of the remainder or after whitespace. `left-wing` and `COVID-19` are **not** subtraction. Do not rewrite them when building `keyword_query`.

**Dates:** naive local calendar `YYYY-MM-DD`. Invalid → `Error::usage`. `after` = that date 00:00:00 local; `before` = next day 00:00:00 local. Use `chrono::Local`.

`--under` / `path:`: tilde-expand, then `std::fs::canonicalize`. If DIR does not exist, `Error::usage`.

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn kind_pdf_and_spaced() {
    let a = parse_query("kind:pdf pets", &FilterFlags::default()).unwrap();
    let b = parse_query("kind: pdf pets", &FilterFlags::default()).unwrap();
    assert_eq!(a.filters.kind, vec![FileKind::Pdf]);
    assert_eq!(a.nl_query, "pets");
    assert_eq!(a, b);
}

#[test]
fn hyphenated_word_is_not_fts_minus() {
    let p = parse_query("left-wing", &FilterFlags::default()).unwrap();
    assert_eq!(p.keyword_query, "left-wing");
    assert_eq!(p.semantic_query, "left-wing");
}

#[test]
fn leading_minus_is_operator() {
    let p = parse_query("-wing birds", &FilterFlags::default()).unwrap();
    assert_eq!(p.keyword_query, "-wing birds");
    assert_eq!(p.semantic_query, "birds"); // standalone FTS op token dropped from semantic copy
}

#[test]
fn unknown_prefix_stays_in_query() {
    let p = parse_query("foo:bar pets", &FilterFlags::default()).unwrap();
    assert_eq!(p.nl_query, "foo:bar pets");
}

#[test]
fn flags_win_over_prefixes() {
    let flags = FilterFlags { kind: vec![FileKind::Text], ..Default::default() };
    let p = parse_query("kind:pdf pets", &flags).unwrap();
    assert_eq!(p.filters.kind, vec![FileKind::Text]);
}

#[test]
fn code_prefix_sets_task() {
    let p = parse_query("code: hashmap rust", &FilterFlags::default()).unwrap();
    assert_eq!(p.task, QueryTask::Code);
    assert_eq!(p.nl_query, "hashmap rust");
}

#[test]
fn invalid_date_is_usage() {
    let err = parse_query("after:2026-13-01 x", &FilterFlags::default()).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}
```

For `leading_minus`, semantic copy removes standalone FTS operators `OR`, `AND`, `NOT`, and `+`/`-` tokens (the operator plus its immediately-following token? Spec: "removes standalone FTS operators `OR`, `AND`, `NOT`" and `+`/`-` are operators at start/after whitespace. Semantic copy unquotes phrases (inner text kept) and removes those operators. For `-wing birds`, semantic is `birds` (operator token `-wing` removed as a `-` operator with value). Keep this explicit in `parse.rs` comments.

Quoted phrases: `"pets in the lease"` → semantic `pets in the lease`, keyword keeps quotes.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test query_parse`
Expected: FAIL

- [ ] **Step 3: Implement parser** (no regex-replace of interior hyphens)

- [ ] **Step 4: Run tests**

Run: `cargo test --test query_parse`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/search tests/query_parse.rs
git commit -m "$(cat <<'EOF'
feat: parse search prefixes without treating hyphens as FTS minus

EOF
)"
```

---

### Task 8: RRF, SQL filters, search JSON on fixture rows

**Files:**
- Create: `src/search/rrf.rs`, `src/search/sql.rs`, `src/search/json.rs`; fill `src/search/mod.rs`
- Test: `tests/search_rrf.rs`, `tests/search_json.rs`, `tests/search_filters.rs`
- Modify: `src/cli.rs` to call search; `Cargo.toml` if needed

**Interfaces:**
- Consumes: `Store`, `Embedder`, `ParsedQuery`, `Config.search`
- Produces:

```rust
pub struct SearchRequest {
    pub parsed: ParsedQuery,
    pub mode: SearchMode,
    pub limit: usize,
    pub min_score: f64,
    pub json: bool,
}

pub struct Hit {
    pub path: String,
    pub kind: FileKind,
    pub chunk_kind: ChunkKind,
    pub collection: String,
    pub ext: String,
    pub page: Option<i64>,
    pub ordinal: i64,
    pub mtime: i64,
    pub score: f64, // fused RRF
    pub scores: HitScores,
    pub snippet: String,
}

pub struct HitScores {
    pub vec: Option<f64>,      // cosine
    pub fts: Option<f64>,      // raw FTS5 rank
    pub filename: Option<f64>, // raw FTS5 rank
}

pub fn rrf(ranks: &[(i64, Arm, u32 /*1-based rank*/)], k: f64, w_vec: f64, w_fts: f64, w_fn: f64) -> Vec<(i64, f64)>;
pub fn run_search(store: &Store, embedder: &dyn Embedder, cfg: &Config, req: SearchRequest) -> Result<SearchOutput>;
pub fn render_json(out: &SearchOutput) -> String; // one object, nothing else
pub fn render_human(out: &SearchOutput) -> String; // path[:page] then snippet; no percent
```

Pipeline (implement exactly):
1. If semantic copy empty: skip vec arm unless `mode=semantic` → `Error::usage`.
2. Vec arm: KNN on `chunks_vec`. Metadata WHERE for kind/ext/mtime/collection using `file_kind IN (...)`. If `--under` or `--name` set, `k = min(candidates * 8, 400)`, JOIN `files`, drop mismatches, take up to `candidates`. Do not loop a second KNN.
3. Keyword arm: FTS5 MATCH on `chunks_fts` (body), same filters, cap `candidates`.
4. Filename arm: FTS5 MATCH on `filename` and `path` only.
5. RRF unless single-arm mode. Drop fused `< min_score`. Return `limit` hits.
6. `scores.vec = clamp(1 - (l2_distance^2)/2, 0, 1)` because vectors are L2-normalized. `scores.*` is JSON `null` when that arm did not list the hit.
7. Snippet: ≤200 chars from `chunks.text` first match window; if text null/empty, basename.

JSON object fields frozen as spec §6.4. `query` = `nl_query`. `task` = `"search"` or `"code"`. Zero hits: exit 0, `"hits": []`.

Vec SQL sketch:

```sql
SELECT rowid, distance
FROM chunks_vec
WHERE file_kind IN ('pdf')
  AND embedding MATCH ?
  AND k = ?;
```

sqlite-vec match blob: little-endian `f32` bytes of the query vector. After JOIN:

```sql
SELECT cv.rowid, cv.distance
FROM chunks_vec cv
JOIN chunks c ON c.id = cv.rowid
JOIN files f ON f.id = c.file_id
WHERE cv.embedding MATCH ?
  AND k = ?
  AND f.path GLOB ?   -- under: prefix + '*' after canonicalize
  AND f.path GLOB '*' || ?; -- do NOT use this for --name
```

`--name` is basename only: `ferret_basename(f.path) GLOB :name` (SQL GLOB, case-sensitive). `--under`: `f.path = :under OR f.path > :under || '/' AND f.path < :under || '0'` **or** `instr(f.path, :under) = 1` after ensuring `:under` has no trailing slash. Prefer `f.path = ? OR f.path LIKE ?` with `dir/%` only if LIKE is case-sensitive (it is not on ASCII by default with NOCASE). **Use `GLOB` with a escaped prefix + `*` or a `>=` / `<` range on the canonical prefix.** Spec: path prefix. Implement `path_is_under(path, dir)` in SQL as `f.path = :u OR f.path GLOB :u || '/*'` — GLOB `/` is not special. Escape GLOB metacharacters in the directory string (`*`, `?`, `[`).

- [ ] **Step 1: Write failing RRF + JSON + filter tests**

`tests/search_rrf.rs`:

```rust
#[test]
fn rrf_prefers_agreement() {
    // chunk 1 rank 1 vec + rank 1 fts; chunk 2 rank 2 vec only
    let fused = rrf(
        &[(1, Arm::Vec, 1), (1, Arm::Fts, 1), (2, Arm::Vec, 2)],
        60.0, 1.0, 1.0, 0.4,
    );
    assert_eq!(fused[0].0, 1);
    let s1 = 1.0 / 61.0 + 1.0 / 61.0;
    assert!((fused[0].1 - s1).abs() < 1e-12);
}
```

`tests/search_filters.rs`: in-memory/temp store, insert text file under `/tmp/a/in.md` and `/tmp/b/out.md` with distinct fake vectors/text. `kind:text` must not return an image row. `--under /tmp/a` hybrid must return the in-dir hit (FTS if vec starved). `--kind pdf` uses `file_kind = 'pdf'` (file kind, not chunk kind).

`tests/search_json.rs`:

```rust
#[test]
fn json_object_keys_and_null_page() {
    let out = run_search_json_on_fixture("pets");
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    for k in ["query", "task", "mode", "limit", "min_score", "filters", "hits"] {
        assert!(v.get(k).is_some(), "{k}");
    }
    let f = &v["filters"];
    for k in ["kind", "ext", "under", "collection", "name", "after", "before"] {
        assert!(f.get(k).is_some(), "{k}");
    }
    let h = &v["hits"][0];
    assert!(h.get("start_byte").is_none());
    assert!(h.get("end_byte").is_none());
    assert!(h["scores"]["vec"].as_f64().unwrap() >= 0.0);
    assert!(h["scores"]["vec"].as_f64().unwrap() <= 1.0);
    assert_eq!(h["chunk_kind"], "text");
    assert!(h["page"].is_null());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test search_rrf --test search_json --test search_filters`
Expected: FAIL

- [ ] **Step 3: Implement search + wire `ferret search`**

`mode=semantic` with empty semantic copy → exit 2. Fatal embedder/DB → stderr, empty stdout, exit 1. JSON to stdout only; warnings stderr.

Human line: `{path}` or `{path}:{page}` then a short snippet. Not a client contract.

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/search src/cli.rs tests/search_rrf.rs tests/search_json.rs tests/search_filters.rs
git commit -m "$(cat <<'EOF'
feat: hybrid search with SQL filters, RRF, and frozen JSON

EOF
)"
```

---

### Task 9: Text extraction and chunking

**Files:**
- Create: `src/extract/mod.rs`, `src/extract/text.rs`
- Test: `tests/extract_text.rs`, `testdata/two_sentences.txt`, `testdata/numeric.txt`, `testdata/letterless.txt`, `testdata/sample.html`
- Modify: `Cargo.toml` (`unicode-segmentation`, `html-escape`)

**Interfaces:**
- Consumes: `Config.chunk.text`, `FileKind`
- Produces:

```rust
pub struct Chunk {
    pub kind: ChunkKind,
    pub text: Option<String>,
    pub start_byte: Option<i64>,
    pub end_byte: Option<i64>,
    pub page: Option<i64>,
    pub embed: EmbedDecision,
}

pub enum EmbedDecision {
    Vec,      // write chunks + fts + vec
    FtsOnly,  // numeric_heavy: chunks + fts, no vec
    Drop,     // skip_letterless: do not emit
}

pub fn extract_text(path: &Path, bytes: &[u8], cfg: &TextChunkConfig) -> Result<Vec<Chunk>>;
```

HTML (`ext` html/htm): strip tags, decode entities, then same windows. JSON/YAML/TOML: raw UTF-8, no pretty-print. `svg` is text.

`mode=chars`: sliding windows of `max_chars` Unicode scalars, overlap `overlap`, prefer break on newline then space. Store UTF-8 byte offsets.

`mode=sentences`: `unicode_sentences()`, pack until `max_chars`; one over-long sentence falls back to char windows (with overlap) for that sentence only.

`skip_letterless`: if no `char::is_alphabetic`, **Drop**.

`numeric_heavy`: let `n` = non-whitespace scalars, `d` = ASCII digits among those; if `n>0` and `d as f64 / n as f64 >= numeric_heavy` (and numeric_heavy != 0) → `FtsOnly`.

If `overlap >= max_chars`, config already rejected.

Empty/whitespace file → empty vec (caller still writes `files` complete=1).

Title for document prefix is basename — that is index's job, not extract.

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn sentences_split_when_chars_would_not() {
    let cfg = TextChunkConfig { mode: TextChunkMode::Sentences, max_chars: 2000, overlap: 200, skip_letterless: true, numeric_heavy: 0.7 };
    let chunks = extract_text(Path::new("x.txt"), b"Hello world. Goodbye moon.", &cfg).unwrap();
    assert_eq!(chunks.len(), 2);
}

#[test]
fn numeric_heavy_is_fts_only() {
    let cfg = TextChunkConfig { mode: TextChunkMode::Chars, max_chars: 2000, overlap: 200, skip_letterless: true, numeric_heavy: 0.7 };
    let chunks = extract_text(Path::new("n.txt"), b"1234567890 12345", &cfg).unwrap();
    assert!(matches!(chunks[0].embed, EmbedDecision::FtsOnly));
}

#[test]
fn letterless_dropped() {
    let cfg = TextChunkConfig { mode: TextChunkMode::Chars, max_chars: 2000, overlap: 200, skip_letterless: true, numeric_heavy: 0.7 };
    let chunks = extract_text(Path::new("x.txt"), b"***\n---\n", &cfg).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn html_strips_tags() {
    let cfg = TextChunkConfig { mode: TextChunkMode::Chars, max_chars: 2000, overlap: 200, skip_letterless: true, numeric_heavy: 0.7 };
    let chunks = extract_text(Path::new("a.html"), b"<p>Hi&amp;bye</p>", &cfg).unwrap();
    assert_eq!(chunks[0].text.as_deref(), Some("Hi&bye"));
}
```

Wait: `&amp;` decodes to `&`, so text is `Hi&bye`. Offsets are into the **original file** bytes, not the stripped string — spec: `start_byte` / `end_byte` in the original file (UTF-8 byte offsets). For HTML, offsets into the original file covering the extracted region are best-effort: use the span of the remaining text mapped back if cheap; if mapping is messy, store offsets into the stripped buffer **only if** we document that — spec says original file. Implement a simple tag stripper that records byte ranges of kept scalars in the original buffer.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test extract_text`
Expected: FAIL

- [ ] **Step 3: Implement**

Do not use a full browser HTML parser. Sequential scan: drop `<...>` including comments/`<script>`/`<style>` inner text; `html_escape::decode_html_entities`.

- [ ] **Step 4: Run tests**

Run: `cargo test --test extract_text`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/extract testdata tests/extract_text.rs Cargo.toml
git commit -m "$(cat <<'EOF'
feat: extract and chunk text with sentence and numeric-heavy modes

EOF
)"
```

---

### Task 10: Walk, skip rules, gitignore, globs

**Files:**
- Create: `src/index/skip.rs`, `src/index/walk.rs`, `src/index/mod.rs` (stubs)
- Modify: `Cargo.toml` (`ignore = "0.4"`, `globset = "0.4"`)
- Test: `tests/walk_skip.rs`

**Interfaces:**
- Consumes: `Config.files`, `FileKind`, `extension_kind`
- Produces:

```rust
pub enum SkipReason {
    UnsupportedExt,
    SkipRule,
    NoImageBackend,
    HeicSkip,
    NoMagick,
    Gitignore,
    IncludeGlobs,
}

impl SkipReason {
    pub fn as_str(&self) -> &'static str; // spec NDJSON reasons
}

pub enum WalkEvent {
    File { path: PathBuf, collection: String, kind: FileKind, ext: String },
    Skip { path: PathBuf, reason: SkipReason },
}

pub fn walk(roots: &[RootEntry], extra: &[PathBuf], files_cfg: &FilesConfig, follow_symlinks: bool) -> Result<Vec<WalkEvent>>;
pub fn kind_for_ext(ext: &str, map: &BTreeMap<String, FileKind>) -> Option<FileKind>;
```

Walk order after extension typing: skip-rule list → include_globs (if non-empty, relative path from collection root must match ≥1 glob) → respect_gitignore via `ignore` crate when a `.git` exists at or above the file (at or below a configured root). A configured root path is never gitignored away.

Skip-rule list (walk-skip), names apply even when the root is hidden:
- path component in `{.git, node_modules, .cache, target, dist, __pycache__, .venv, venv}`
- hidden components (name starts with `.`) except the configured root itself, unless `include_hidden`
- basename globs: `*.pem` `*.key` `*.p8` `*.p12` `*.pfx` `.netrc` `*.netrc` `.env` `.env.*` `credentials.json` `*.sqlite-journal`
- basename `id_rsa` `id_dsa` `id_ecdsa` `id_ed25519` or `id_*` not ending in `.pub`
- size > `max_size`
- user `exclude_globs`
- include_globs miss
- gitignore hit

Never follow symlinks unless configured. Unknown extensions: walk-skip (no new files row). `.heic`/`.heif` with `heic=skip` → `HeicSkip`. Do not run magick here; missing magick is Task 14.

If a walk-skipped path **already exists in DB**, Task 11 deletes it. Walk only reports the skip.

- [ ] **Step 1: Write failing tests**

```rust
fn fixture_tree() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("corpus");
    std::fs::create_dir_all(root.join(".hidden")).unwrap();
    std::fs::write(root.join("ok.md"), "hi").unwrap();
    std::fs::write(root.join("secret.pem"), "k").unwrap();
    std::fs::write(root.join(".hidden/a.txt"), "x").unwrap();
    std::fs::write(root.join("skipme.rs"), "fn x(){}").unwrap();
    (tmp, root)
}

#[test]
fn pem_is_skip_rule() {
    let (_tmp, root) = fixture_tree();
    let ev = walk_one(&root);
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::Skip { reason: SkipReason::SkipRule, .. })));
}

#[test]
fn hidden_skipped_unless_include_hidden() {
    let (_tmp, root) = fixture_tree();
    let ev = walk_one(&root);
    assert!(ev.iter().any(|e| match e {
        WalkEvent::Skip { path, reason: SkipReason::SkipRule } => path.ends_with("a.txt"),
        _ => false,
    }));
}

#[test]
fn include_globs_skips_non_matching() {
    let (_tmp, root) = fixture_tree();
    let mut cfg = files_cfg();
    cfg.include_globs = vec!["*.md".into()];
    let ev = walk(&[root], &cfg);
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::Skip { reason: SkipReason::IncludeGlobs, .. })));
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::File { kind: FileKind::Text, ext, .. } if ext == "md")));
}

#[test]
fn gitignore_skip() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("repo");
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::write(root.join(".gitignore"), "skipme.txt\n").unwrap();
    std::fs::write(root.join("skipme.txt"), "x").unwrap();
    std::fs::write(root.join("keep.md"), "y").unwrap();
    let ev = walk(&[root], &files_cfg());
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::Skip { reason: SkipReason::Gitignore, .. })));
}
```

Helper `walk_one` in the test file wraps `walk` with compiled `FilesConfig` defaults. Extra path under a root inherits that root's collection — assert in a fifth test: `walk(&[root], extra: &[root.join("ok.md")])` yields `collection == root.file_name()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test walk_skip`
Expected: FAIL

- [ ] **Step 3: Implement walk using `ignore::WalkBuilder`** only when `respect_gitignore`; otherwise `walkdir` **or** `ignore` with git disabled. Prefer `ignore::WalkBuilder::new.hidden(false)` and apply hidden rules ourselves so the configured root may be `~/.ssh`. Do **not** add `walkdir` if `ignore` can do both (`git_ignore(false)`).

- [ ] **Step 4: Run tests**

Run: `cargo test --test walk_skip`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/index tests/walk_skip.rs Cargo.toml
git commit -m "$(cat <<'EOF'
feat: walk corpus with skip rules, gitignore, and include globs

EOF
)"
```

---

### Task 11: Index text files (fake embedder)

**Files:**
- Fill: `src/index/mod.rs`
- Modify: `src/cli.rs` (`index`, `status` minimal), `tests/common.rs` (`test_env`)
- Test: `tests/index_pipeline.rs`
- Create stubs: `src/extract/image.rs`, `src/extract/pdf.rs` (replaced in Tasks 14–15)

**Test helper** (`tests/common.rs`):

```rust
pub struct TestEnv {
    pub tmp: tempfile::TempDir,
    pub root: PathBuf,
    pub cfg: PathBuf,
    pub index: PathBuf,
}

pub fn test_env() -> TestEnv {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("XDG_CONFIG_HOME", tmp.path().join("xdg-cfg"));
    std::env::set_var("XDG_DATA_HOME", tmp.path().join("xdg-data"));
    std::env::set_var("XDG_CACHE_HOME", tmp.path().join("xdg-cache"));
    std::env::set_var("FERRET_EMBEDDER", "fake");
    let root = tmp.path().join("corpus");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("pets.md"), "pets in the lease").unwrap();
    ferret::cli::try_run(&[
        "ferret", "init", "--root", root.to_str().unwrap(),
    ]).unwrap();
    TestEnv {
        cfg: tmp.path().join("xdg-cfg/ferret/config.toml"),
        index: tmp.path().join("xdg-data/ferret/index.sqlite"),
        root,
        tmp,
    }
}
```

Mark CLI tests that call `test_env` / `set_var` with `#[serial]`.

**Interfaces:**
- Consumes: walk, extract_text, embedder, store, fingerprint
- Produces: `index::run(cfg, embedder, extra_paths, json_ndjson: bool, strict: bool) -> Result<IndexSummary>`

```rust
pub struct IndexSummary {
    pub seen: u64,
    pub skipped: u64,
    pub embedded: u64,
    pub errors: u64,
}
```

Stderr summary: files seen / skipped / embedded / errors.

`--json` on index: NDJSON stdout events `{"event":"file","path":"...","status":"ok|skip|error","reason":null or str,"error":null or str}`. Walk-skip has `path` and no DB id.

**Per text file:** one SQLite transaction (extract + embed + insert). `numeric_heavy` chunks: insert chunk+FTS, skip vec. Letterless dropped before insert.

Fingerprint skip: unchanged fingerprint **and** `complete=1` → skip extract/embed. `complete=0` → retry whole text file.

Pipeline granularity vs stored meta:
- `text_pipeline` only changed: delete text chunks (and PDF text per Task 15); keep image vecs
- `image_pipeline` only: Task 14
- both / `extension_kind` / `embed_recipe` / `prefix_version`: full reindex `next_page=1`

Unchanged `config_fingerprint` is the skip fast-path.

**Collections:** unique. Default last component of canonical root. Explicit `collection` overrides. Collision at start of `index`/`init` → exit 1. Extra paths under a root inherit; extra paths not under a root use last component and error if that name exists.

`index` does not delete missing-from-disk rows (`gc` does). It **does** delete walked paths that now match walk-skip.

Per-file extract errors: set `files.error`, `complete=0`, continue. `--strict` → exit 1 if any. Fatal: DB, embedder unreachable, schema mismatch.

Upload warning once on stderr if backend is `openai` or `sglang`.

Probe `images_supported` once per process (Task 14). For text-only this task: if fake, true; if HTTP not implemented, still index text.

- [ ] **Step 1: Write failing tests** using `FERRET_EMBEDDER=fake` and temp index/roots:

```rust
#[test]
fn index_markdown_then_search() {
    let env = test_env(); // sets XDG_*, FERRET_EMBEDDER=fake, writes a root with pets.md
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out = ferret::cli::try_run(&["ferret", "search", "--json", "pets"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["hits"][0]["kind"], "text");
}

#[test]
fn bump_max_chars_reembeds_text() {
    let env = test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let fp1 = meta_fingerprint(&env.index);
    std::fs::write(env.cfg, "[chunk.text]\nmax_chars = 100\noverlap = 10\n").unwrap();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let fp2 = meta_fingerprint(&env.index);
    assert_ne!(fp1, fp2);
}

#[test]
fn collection_collision_is_fatal() {
    let err = index_two_roots_named_docs().unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
fn walk_skip_deletes_existing_row() {
    let env = test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    std::fs::write(env.root.join("gone.md"), "x").unwrap();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    std::fs::remove_file(env.root.join("gone.md")).unwrap();
    // still in DB until gc — this test instead adds exclude_globs = ["gone.md"] and reindexes
    assert!(file_row(&env.index, "gone.md").is_none());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test index_pipeline`
Expected: FAIL

- [ ] **Step 3: Implement text indexing**

Put this match in `index::run`. Do **not** invent a skip reason. Image/Pdf call extract functions that Task 14/15 replace; until then they return `Err(Error::fatal("not implemented"))` which `index::run` **catches per file** (does not abort the walk): writes `files.error`, `complete=0`, NDJSON `status=error`, continues. Embedder-down (`endpoint` unreachable) is still fatal and aborts.

```rust
match kind {
    FileKind::Text => index_text_file(/* one SQLite transaction */)?,
    FileKind::Image => match extract::image::extract_image(&path, &cfg.chunk.image, &cache, cfg.files.heic) {
        Ok(chunks) => commit_file(chunks)?,
        Err(e) => record_extract_error(&path, e),
    },
    FileKind::Pdf => match extract::pdf::extract_pdf(/* ... */) {
        Ok(_) => { /* Task 15 uses per-page txs; this arm becomes that loop */ }
        Err(e) => record_extract_error(&path, e),
    },
}

pub fn extract_image(...) -> Result<Vec<Chunk>> {
    Err(Error::fatal("image extract not implemented"))
}
pub fn extract_pdf(...) -> Result<Vec<Chunk>> {
    Err(Error::fatal("pdf extract not implemented"))
}
```

`record_extract_error` is not `Error::fatal` to the CLI: increment `IndexSummary.errors`, set the row, continue. `--strict` → after the walk, if `errors > 0` return `Error::fatal`. Create `src/extract/image.rs` and `src/extract/pdf.rs` with those stubs in this task so the crate compiles.

- [ ] **Step 4: Run tests**

Run: `cargo test --test index_pipeline`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/index src/cli.rs tests/index_pipeline.rs
git commit -m "$(cat <<'EOF'
feat: index text files with fake embeddings and fingerprints

EOF
)"
```

---

### Task 12: HTTP llama_server / litert_lm text embeddings, health, retries, managed spawn

**Files:**
- Create: `src/embedder/http.rs`, `src/embedder/probe.rs`
- Modify: `src/embedder/mod.rs` `build_embedder`
- Test: `tests/embedder_http.rs` with `httpmock` 0.8
- Modify: `Cargo.toml` (`reqwest` blocking+json+rustls, `base64` unused until images, `httpmock` dev)

**Interfaces:**
- Consumes: `EmbedderConfig`, `EmbedInput`, prefixes
- Produces: `HttpEmbedder` implementing `Embedder`

Health: `GET {endpoint}/health`. HTTP 404 → fall back `POST {endpoint}/v1/embeddings` with input `"ok"`. Other 4xx/5xx or connect fail → embedder down (`Error::fatal` mentioning `embedder.endpoint`).

Text POST (both llama_server and litert_lm): Appendix B.1. `Content-Type: application/json`. `encoding_format: "float"`. Do **not** send `dimensions` for llama_server/litert_lm. Auth: `Authorization: Bearer {api_key}` when non-empty.

Retries: 3 attempts, sleep 200ms, 400ms, 800ms on 429/503/connection reset. Other 4xx: no retry; per-file error if during index, fatal if health/query.

`managed = true`: if health fails, spawn `managed_command` if non-empty, else:

```text
llama-server -m {gguf} --mmproj {mmproj} --embeddings --pooling mean
  --host 127.0.0.1 --port {from endpoint} -c 4096 -b 4096 -ub 4096
```

Omit `--mmproj` if `mmproj` empty. Do **not** shut the child down. Then re-health.

Images_supported this task: `backend=openai` → false (openai client still Task 16). `multimodal` Some(true/false) → that value. Else probe 1×1 PNG (Task 14 can share probe); if HTTP 4xx/transport on probe → false + one stderr warning. Fake skips HTTP.

- [ ] **Step 1: Write httpmock tests**

```rust
#[test]
fn llama_server_query_body_is_appendix_b1() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings").json_body(serde_json::json!({
            "model": "embeddinggemma-2",
            "encoding_format": "float",
            "input": "task: search result | query: pets in the lease"
        }));
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": [1.0, 0.0, 0.0, 0.0] }]
        }));
    });
    let mut cfg = compiled_defaults(&test_dirs());
    cfg.embedder.endpoint = server.base_url();
    cfg.embedder.dimensions = 4;
    let e = HttpEmbedder::new(&cfg).unwrap();
    let _ = e.embed(EmbedInput::Query { text: "pets in the lease".into(), task: QueryTask::Search }).unwrap();
    mock.assert();
}

#[test]
fn llama_server_omits_dimensions_key() {
    // mock when.json_body_includes without "dimensions"; fail if body contains the key
}

#[test]
fn retries_503_then_succeeds() {
    // first POST 503, second 200; call count 2
}

#[test]
fn does_not_retry_400() {
    // POST 400 once; embed Err; call count 1
}

#[test]
fn bearer_header_when_api_key_set() {
    // when.header("Authorization", "Bearer secret")
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test embedder_http`
Expected: FAIL

- [ ] **Step 3: Implement frozen bodies** using `serde_json::json!`. Parse response `data[0].embedding` as `Vec<f32>`. Run `truncate_or_check` + already normalized? Server may not L2-normalize — **always** L2-normalize what ferret stores.

- [ ] **Step 4: Run tests**

Run: `cargo test --test embedder_http`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedder tests/embedder_http.rs Cargo.toml
git commit -m "$(cat <<'EOF'
feat: POST OpenAI-shaped text embeddings to llama-server

EOF
)"
```

---

### Task 13: rm, gc, status, index NDJSON polish, init upload warning

**Files:**
- Modify: `src/cli.rs`, `src/store/mod.rs`, `src/index/mod.rs`
- Test: `tests/cli_maintenance.rs`

**Interfaces:**
- `rm PATH...`: canonicalize each path; delete file row if present (CASCADE chunks); if not present, stderr warn, exit 0. Does not change meta.
- `gc`: delete DB rows whose files are gone from disk (and their chunks/vec/fts via FK).
- `status` / `status --json`: fields exactly spec §8. No embedder call. `files: {text,image,pdf}` counts, `chunks`, `incomplete` (`complete=0`), `errors` (non-null `error`), `indexed_at_max`.
- `index --json` NDJSON reasons include `unsupported_ext`, `skip_rule`, `no_image_backend`, `heic_skip`, `no_magick`, `gitignore`, `include_globs`.
- `init` stderr upload warning only if merged backend is openai/sglang (not at init if still llama_server). Index prints it when backend is openai/sglang.

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn rm_missing_path_is_success() {
    let env = test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out = ferret::cli::try_run(&["ferret", "rm", "/no/such/file.md"]).unwrap();
    assert!(out.stderr.contains("not in index") || out.stderr.to_lowercase().contains("missing"));
}

#[test]
fn gc_deletes_gone_files() {
    let env = test_env();
    std::fs::write(env.root.join("gone.md"), "x").unwrap();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    std::fs::remove_file(env.root.join("gone.md")).unwrap();
    ferret::cli::try_run(&["ferret", "gc"]).unwrap();
    assert!(file_row(&env.index, "gone.md").is_none());
}

#[test]
fn status_json_shape_and_no_embed() {
    let env = test_env();
    let out = ferret::cli::try_run(&["ferret", "status", "--json"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    for k in ["index","model_id","dimensions","backend","images_supported","files","chunks","incomplete","errors","indexed_at_max"] {
        assert!(v.get(k).is_some(), "{k}");
    }
    assert!(v["files"].get("text").is_some());
}

#[test]
fn init_openai_does_not_warn_until_index() {
    // init with default llama_server: stderr has no "upload"
    // writing backend=openai then `ferret index` stderr contains "upload"
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test cli_maintenance`
Expected: FAIL

- [ ] **Step 3: Implement rm/gc/status**

`rm`: canonicalize each CLI path (tilde-expand first); `delete_file_by_path`; if false, `eprintln!` one line, continue; always exit 0 unless canonicalize of a still-existing path fails fatally. Missing path: warn, continue.

`gc`: `SELECT path FROM files`; if `!Path::new(path).exists()`, delete that row (CASCADE).

`status --json`: object with keys exactly spec §8. `files.text|image|pdf` from `SELECT kind, count(*)`. `incomplete` = `complete=0`. `errors` = `error IS NOT NULL`. `indexed_at_max` = `max(indexed_at)` or JSON null. No `build_embedder` call.

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cli.rs src/store src/index tests/cli_maintenance.rs
git commit -m "$(cat <<'EOF'
feat: add ferret rm, gc, and status

EOF
)"
```

---

### Task 14: Images — extract, capability skip, fingerprint vs probe, HEIC

**Files:**
- Create: `src/extract/image.rs`
- Modify: `src/index/mod.rs`, `src/embedder/http.rs` (Appendix B.2), `src/embedder/probe.rs`
- Test: `tests/extract_image.rs`, `tests/index_images.rs`
- Testdata: 1×1 PNG, 2-frame GIF
- Modify: `Cargo.toml` (`image = { version = "0.25", default-features = false, features = ["png","jpeg","gif","webp","bmp"] }`, `base64`)

**Interfaces:**
- `extract_image(path, cfg, cache_dir, heic: HeicMode) -> Result<ExtractOutcome>`
- Animated GIF/WebP: **first frame only**. Downscale long edge to `max_edge_px` (default 1024), PNG-encode. One chunk, `text` empty, `start_byte`/`end_byte` null, `kind=image`. FTS still gets filename/path via triggers.
- Corrupt image: extract-error (`error` set, `complete=0`, `skip_reason` null).
- `images_supported=false`: capability-skip row `skip_reason=no_image_backend`, `complete=1`, zero chunks. Count skipped, not error. Fingerprint includes `images_supported`; later true probe **must** re-embed.
- `heic=skip`: walk-skip `heic_skip`. `heic=magick`: run `magick` or `convert` → PNG in `{cache_dir}`; missing binary walk-skip `no_magick` (not extract-error). Conversion fail: extract-error.
- HTTP image body: Appendix B.2 (`input: [{ content: [{type:image_url, image_url:{url: data:image/png;base64,...}}]}]`). No text prefix.
- Text-pipeline-only bump: **keep** image vectors. Image-pipeline bump: re-embed images.

Index: one transaction per image file.

- [ ] **Step 1: Write failing tests**
  - Tiny PNG extracts one image chunk
  - GIF first frame (dimensions of frame 0)
  - Index with `multimodal=false` / probe fail → skip reason `no_image_backend`; flip to true (or fake) → image rows get vecs
  - Bump `chunk.text.max_chars` on mixed corpus: PNG vec payload unchanged (compare blob bytes or fake vector); text files new fingerprint
  - `heic=magick` without binary → `no_magick` skip (don't require a HEIC fixture)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test extract_image --test index_images`
Expected: FAIL

- [ ] **Step 3: Implement extract + index arm + HTTP image body + probe PNG**

1×1 PNG bytes (valid file), constant in `probe.rs`. Probe uses `EmbedInput::Image`.

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/extract/image.rs src/embedder src/index testdata tests/extract_image.rs tests/index_images.rs Cargo.toml
git commit -m "$(cat <<'EOF'
feat: index images with capability skips and split fingerprints

EOF
)"
```

---

### Task 15: PDF pages — poppler, per-page transactions, resume cursor

**Files:**
- Create: `src/extract/pdf.rs`
- Modify: `src/index/mod.rs`, `src/embedder/http.rs` (Appendix B.3)
- Test: `tests/extract_pdf.rs`, `tests/index_pdf.rs`
- Testdata: `testdata/two_pages.pdf` (committed minimal 2-page PDF)

**Interfaces:**
- Resolve binaries: `extract.pdfinfo` / `pdftotext` / `pdftoppm` if set, else `{poppler_dir}/name`, else `PATH`.
- `pdfinfo` → page count `n`. `pdftotext -f N -l N -layout`. `pdftoppm -f N -l N -png -r {render_dpi} -singlefile` writing `{cache_dir}/pages/{files.id}-{page}.png`. Delete raster after the page transaction commits. Never write next to the corpus PDF.
- Pages 1-based. `start_byte`/`end_byte` null for `pdf_page` chunks.
- `files.next_page` = next page to attempt (1 on first sight). After embed **or** empty skip, `next_page = page+1`.
- Pass order: in one `index` run, first bring every PDF through `min(first_pass_pages, n)` from `next_page`, then continue through `min(max_pages, n)`. Scheduling only.
- Resume: never jump over a failed page. If a page `< next_page` is missing from `chunks`, backfill it first.
- Embed: if `embed_page_image && images_supported` → `EmbedInput::PdfPage`. Else if `embed_text` and page has text → `Document`. Else skip page (advance cursor).
- `embed_text=false` with images: `EmbedInput::Image` per page; still store pdftotext in `chunks.text` when non-whitespace.
- Always store page text in `chunks.text` when pdftotext returns non-whitespace.
- Missing poppler: extract-error `pdftotext not found` (or whichever missing), `complete=0`.
- `complete=1` iff `error` null **and** `next_page > min(max_pages, n)`. If `n > max_pages`, complete after `max_pages` and stderr warn once per file.
- **One transaction per page.** Ctrl-C / embedder down: roll back open tx, abort command; committed pages stay; `complete=0`.
- HTTP PdfPage (llama_server/litert): image_url part **first**, then text part with Document-prefixed page text (or empty if `embed_text=false`). **No** `<|image|>` in the string.
- Text-pipeline bump for PdfPage vectors: **do not drop vec rows**; rewrite `chunks.text` from pdftotext for FTS. Image-pipeline bump: delete pdf_page vec rows, `next_page=1`.

- [ ] **Step 1: Write failing tests**
  - Skip entire test file if poppler missing (`which pdftotext`); otherwise 2-page fixture → 2 chunks, pages 1 and 2, ordinals 0 and 1
  - Simulate fail after page 1 (`complete=0`, `next_page=2`); resume embeds page 2 next, not a jump
  - httpmock: PDF body has image_url then text, and text does not contain `<|image|>`

Include a **minimal two-page PDF** as `testdata/two_pages.pdf`. If binary is awkward in git, generate with a checked-in ASCII PDF:

```
%PDF-1.1
1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj
2 0 obj<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>endobj
3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 5 0 R >>endobj
4 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 6 0 R >>endobj
5 0 obj<< /Length 44 >>stream
BT /F1 12 Tf 10 100 Td (page one pets) Tj ET
endstream
endobj
...
```

Valid PDF generation: write a known-good tiny file during this task using Python `pypdf` **or** commit a real file created once with `printf`. Verify `pdfinfo` shows 2 pages before checking in.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test extract_pdf --test index_pdf`
Expected: FAIL (or skip if no poppler — still compile)

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS (PDF tests skipped only when poppler absent)

- [ ] **Step 5: Commit**

```bash
git add src/extract/pdf.rs src/index src/embedder testdata tests/extract_pdf.rs tests/index_pdf.rs
git commit -m "$(cat <<'EOF'
feat: index PDF pages with per-page resume cursor

EOF
)"
```

---

### Task 16: openai + sglang HTTP profiles and capability flags

**Files:**
- Modify: `src/embedder/http.rs`
- Test: `tests/embedder_http.rs` (new cases)

**Interfaces:**
- `openai`: strings only. Send `"dimensions": n` when `n != 768`; omit when 768. No image parts. Image files capability-skip. PDF pages as `Document` (text only) even if `embed_page_image` true.
- `sglang`: items `{text, image}`. One image per item. PDF: `text` is `<|image|>\n` + Document-prefixed page text (or just `<|image|>` if no text), plus `image` data URI. Send `dimensions` when `n != 768`.
- `litert_lm`: same HTTP as llama_server; prefixes already Task 6. Default prefix_profile already Task 2.
- Mixing prefix profiles in one index: already a hard error via meta (Task 5).
- Ollama is **not** a backend enum value. Do not send `/api/embed`.

If openai/sglang server rejects `dimensions`, fail the call (do not store 768-d in a 256-d index).

- [ ] **Step 1: Write httpmock tests** for B.4 and B.5

```rust
#[test]
fn openai_query_sends_dimensions_when_not_768() {
    // body has "dimensions": 256 and input is a string, never an array of content parts
}

#[test]
fn openai_omits_dimensions_at_768() {
    // body has no "dimensions" key
}

#[test]
fn openai_image_file_never_posts() {
    // index a PNG with backend=openai; mock expects 0 POSTs that contain image_url
}

#[test]
fn sglang_pdf_uses_image_marker() {
    // POST body input[0].text starts with "<|image|>\n" and has "image": "data:image/png;base64,"
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test embedder_http`
Expected: FAIL on new cases

- [ ] **Step 3: Implement dialect mapping from `EmbedInput` + backend**

- [ ] **Step 4: Run tests**

Run: `cargo test --test embedder_http`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedder/http.rs tests/embedder_http.rs
git commit -m "$(cat <<'EOF'
feat: add openai and sglang embedding HTTP dialects

EOF
)"
```

---

### Task 17: Nix flake, wrap poppler, operator README

**Files:**
- Create: `flake.nix`, `nix/module.nix`, `README.md`
- Generate: `flake.lock` via `nix flake lock`
- Modify: none of the Rust JSON contract (already frozen in Task 8)

**Interfaces:**
- `packages.ferret` default; `meta.mainProgram = "ferret"`
- Wrap PATH with `poppler-utils` (`pdfinfo`, `pdftotext`, `pdftoppm`)
- Default package does **not** pull ImageMagick; module option `heicMagick` may wrap it
- `devShells.default`: rustc, cargo, clippy, rustfmt, pkg-config, poppler-utils
- `overlays.default`
- `nixosModules.ferret` writes `/etc/ferret/config.toml` from options; **no daemon**
- Systems: `x86_64-linux`, `aarch64-linux`, `aarch64-darwin`, `x86_64-darwin`
- Do not vendor GGUFs, llama.cpp, LiteRT, or PyTorch

README must document:
- llama-server argv from spec §3.4 (`--embeddings --pooling mean`, `-c/-b/-ub`, `--mmproj`)
- `nix build` / `nix develop`
- `FERRET_EMBEDDER=fake`
- Quickshell/fzf as **external** clients of `ferret search --json` (no QML in repo)
- `--name` case-sensitive GLOB
- `--under`/`--name` can starve the semantic arm
- JSON `score` is not a percent; use `scores.vec` for a 0–1 bar
- macOS paths + XDG override
- poppler via Nix or Homebrew; HEIC via magick optional

Flake sketch (adjust nixpkgs pin when locking):

```nix
{
  description = "ferret: local-first semantic file search";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAll = nixpkgs.lib.genAttrs systems;
    in {
      packages = forAll (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          ferret = pkgs.rustPlatform.buildRustPackage {
            pname = "ferret";
            version = "0.1.0";
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;
            nativeBuildInputs = [ pkgs.pkg-config pkgs.makeWrapper ];
            postInstall = ''
              wrapProgram $out/bin/ferret --prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.poppler-utils ]}
            '';
            meta.mainProgram = "ferret";
          };
        in { inherit ferret; default = ferret; });
      devShells = forAll (system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in {
          default = pkgs.mkShell {
            packages = [
              pkgs.rustc pkgs.cargo pkgs.clippy pkgs.rustfmt
              pkgs.pkg-config pkgs.poppler-utils
            ];
          };
        });
      overlays.default = final: prev: {
        ferret = self.packages.${final.system}.ferret;
      };
      nixosModules.ferret = import ./nix/module.nix;
    };
}
```

`nix/module.nix`: `options.services.ferret` or `options.programs.ferret` with config attrs matching TOML sections; `environment.etc."ferret/config.toml".text = ...`. No systemd unit.

- [ ] **Step 1: Write README + flake + module** (no test binary required for README). Add a tiny `tests/readme_contract.rs` **or** a comment in README only — prefer running `cargo test --test search_json` as the freeze gate rather than snapshotting README.

- [ ] **Step 2: Verify flake evaluates**

Run: `nix flake check` (or `nix build .#ferret` if check is heavy). Expected: build succeeds on this system. Darwin systems need not be built from Linux; listing them in `systems` is enough.

If Nix is unavailable in the environment, run `cargo test` and note flake check as a follow-up — still commit `flake.nix`.

- [ ] **Step 3: Confirm search JSON tests still pass (freeze)**

Run: `cargo test --test search_json`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add flake.nix flake.lock nix/module.nix README.md
git commit -m "$(cat <<'EOF'
feat: package ferret with Nix and document the JSON client contract

EOF
)"
```

---

## Self-review (plan vs spec)

**Spec coverage**

| Spec | Task |
|---|---|
| §1 goals / non-goals | Global constraints |
| §2 modules | File structure |
| §3.1 EmbedInput, dims, normalize | 6, 12, 16 |
| §3.2 prefix profiles | 6 |
| §3.3 backends, probe, meta mismatch | 5, 6, 12, 14, 16 |
| §3.4 managed llama-server + health | 12 |
| §4.1–4.2 text kinds/chunking | 9, 10 |
| §4.3 images / HEIC | 14 |
| §4.4 PDFs / next_page / per-page tx | 15 |
| §4.5 skip taxonomy + NDJSON reasons | 10, 11, 13 |
| §5 schema, fingerprint split, collections | 4, 5, 11 |
| §6 search pipeline, RRF, JSON | 7, 8 |
| §7 config layering, init short file | 2, 3 |
| §8 CLI + exit codes + status JSON | 3, 8, 13 |
| §9 retries, transactions, Ctrl-C | 11, 12, 15 |
| §10 test table | tests named in tasks 1–16 |
| §11 build order | Task order 1→17 |
| §12 Quickshell out of tree | Task 17 README |
| §15 Nix | 17 |
| §16 macOS paths / case-sensitive GLOB | 1, 8, 17 |
| Appendix A | 4 |
| Appendix B | 12, 14, 15, 16 |

**Not in this plan (spec forbids for v1):** watcher, GUI/QML, audio/video, Ollama, AUR/deb/Homebrew, vendored GGUF, in-process model, OCR, docx.

**Placeholder scan:** no TBD/TODO remaining. Image/pdf extract functions are created in Task 11 as per-file extract-errors and replaced in Tasks 14–15 before the Nix/README freeze. HTTP mock cases in Tasks 12 and 16 include one fully-specified B.1 body; remaining dialect tests follow that same `MockServer` pattern with Appendix B JSON.

**Type consistency:** `FileKind` / `ChunkKind` / `QueryTask` / `Backend` / `PrefixProfile` live in `src/types.rs` from Task 2. `EmbedInput` Task 6. `Filters`/`ParsedQuery` Task 7. `Hit`/`HitScores` Task 8. `Chunk`/`EmbedDecision` Task 9. `SkipReason` Task 10. `Store::open(path, cfg, images_supported)` Task 5.

---

## Execution notes for agents

- TDD each task: failing test → implement → pass → commit.
- Do not add `watch`, QML, or a second crate.
- When a crate version on crates.io disagrees with the pins above, keep the **API** in this plan and adjust the version in `Cargo.toml`.
- sqlite-vec metadata `IN (...)` requires SQLite ≥ 3.38; rusqlite `bundled` must meet that (0.40 does).
- Do not commit secrets. `ferret init` never writes `api_key`.

---

Plan complete and saved to `docs/superpowers/plans/2026-10-12-ferret-v1.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch with checkpoints.

Which approach?
