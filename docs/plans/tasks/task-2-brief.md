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

