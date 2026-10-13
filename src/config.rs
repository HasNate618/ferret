//! Layered TOML config: compiled defaults → `/etc/ferret/config.toml` →
//! `{config_dir}/config.toml` (or `--config FILE`, which replaces that layer,
//! never stacks) → `./.ferret.toml` → `FERRET__SECTION__KEY` env → `--index`.
//!
//! All config errors are `Error::fatal` (exit 1). `FERRET_EMBEDDER` is a
//! special test hook read by `embedder::build_embedder` via its `env`
//! parameter, never here.

/// System config layer path (also referenced by `config path` output).
pub const SYSTEM_CONFIG_PATH: &str = "/etc/ferret/config.toml";

use crate::error::{Error, Result};
use crate::paths::{FerretDirs, INDEX_FILENAME, ferret_dirs};
use crate::types::{Backend, FileKind, HeicMode, PrefixProfile, SearchMode, TextChunkMode};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use toml::{Table, Value};

/// Input to config loading: the `--config` and `--index` CLI flags.
#[derive(Debug, Clone, Default)]
pub struct LoadOpts {
    /// `--config FILE`; replaces the `{config_dir}/config.toml` layer.
    /// A read error for this file is fatal (exit 1), never a skip.
    pub config_flag: Option<PathBuf>,
    /// `--index PATH`; overrides `index.path`.
    pub index_flag: Option<PathBuf>,
}

/// Layers that contributed values, in merge order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerSource {
    Defaults,
    Etc,
    UserFile(PathBuf),
    CwdFile(PathBuf),
    Env,
    Flags,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub index: IndexConfig,
    pub embedder: EmbedderConfig,
    pub search: SearchConfig,
    pub chunk: ChunkConfig,
    pub extract: ExtractConfig,
    pub files: FilesConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexConfig {
    pub path: PathBuf,
    pub roots: Vec<RootEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootEntry {
    pub path: PathBuf,
    pub collection: String,
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkConfig {
    pub text: TextChunkConfig,
    pub pdf: PdfChunkConfig,
    pub image: ImageChunkConfig,
    pub audio: AudioChunkConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextChunkConfig {
    pub mode: TextChunkMode,
    pub max_chars: usize,
    pub overlap: usize,
    pub skip_letterless: bool,
    pub numeric_heavy: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PdfChunkConfig {
    pub max_pages: u32,
    pub first_pass_pages: u32,
    pub render_dpi: u32,
    pub embed_text: bool,
    pub embed_page_image: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageChunkConfig {
    pub max_edge_px: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunkConfig {
    /// Clip length in seconds; each segment embeds as one vector (~25
    /// tokens/sec of audio, so 30s ≈ 750 tokens).
    pub segment_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractConfig {
    pub poppler_dir: String,
    pub pdfinfo: String,
    pub pdftotext: String,
    pub pdftoppm: String,
    /// Directory containing `ffmpeg`/`ffprobe`, mirroring `poppler_dir`:
    /// empty means search PATH.
    pub ffmpeg_dir: String,
}

#[derive(Debug, Clone, PartialEq)]
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

/// Load config with the real environment, filesystem, and cwd.
pub fn load(opts: &LoadOpts) -> Result<Config> {
    let dirs = ferret_dirs(&|k| std::env::var(k).ok());
    let cwd = std::env::current_dir()?;
    let (cfg, _) = load_from(opts, &dirs, &|k| std::env::var(k).ok(), &cwd, &|p| {
        std::fs::read_to_string(p)
            .map_err(|e| Error::fatal(format!("config `{}`: {e}", p.display())))
    })?;
    Ok(cfg)
}

/// Load config with injectable dirs, env, cwd, and file reader. Returns the
/// merged config plus the layers that contributed (in merge order).
pub fn load_from(
    opts: &LoadOpts,
    dirs: &FerretDirs,
    env: &dyn Fn(&str) -> Option<String>,
    cwd: &Path,
    read: &dyn Fn(&Path) -> Result<String>,
) -> Result<(Config, Vec<LayerSource>)> {
    let mut layers = vec![LayerSource::Defaults];
    let mut merged = Table::new();

    // Skippable: read error means the layer is absent.
    let etc = Path::new(SYSTEM_CONFIG_PATH);
    if let Some(table) = read_layer(etc, read, false)? {
        deep_merge(&mut merged, table);
        layers.push(LayerSource::Etc);
    }

    // User layer: `--config FILE` replaces `{config_dir}/config.toml` and a
    // read error is fatal; the implicit path stays skippable.
    let (user_path, explicit) = match &opts.config_flag {
        Some(p) => (p.clone(), true),
        None => (dirs.config_dir.join("config.toml"), false),
    };
    if let Some(table) = read_layer(&user_path, read, explicit)? {
        deep_merge(&mut merged, table);
        layers.push(LayerSource::UserFile(user_path));
    }

    // Skippable: `./.ferret.toml`.
    let cwd_file = cwd.join(".ferret.toml");
    if let Some(table) = read_layer(&cwd_file, read, false)? {
        deep_merge(&mut merged, table);
        layers.push(LayerSource::CwdFile(cwd_file));
    }

    // `FERRET__SECTION__KEY` (nested sections use more `__`). Only the
    // registry below is probed; `FERRET_EMBEDDER` is not consumed here.
    if apply_env(&mut merged, env)? {
        layers.push(LayerSource::Env);
    }

    let raw: RawConfig = table_into_raw(merged)?;
    let mut cfg = build_config(raw, dirs)?;

    if let Some(index_path) = &opts.index_flag {
        cfg.index.path = index_path.clone();
        layers.push(LayerSource::Flags);
    }

    validate(&cfg)?;
    Ok((cfg, layers))
}

/// Compiled defaults (spec §7.2). `prefix_profile` default depends on
/// `backend`: `litert_lm` when `backend = "litert_lm"`, else `embeddinggemma2`.
pub fn compiled_defaults(dirs: &FerretDirs) -> Config {
    Config {
        index: IndexConfig {
            path: dirs.data_dir.join(INDEX_FILENAME),
            roots: Vec::new(),
        },
        embedder: EmbedderConfig {
            backend: Backend::LlamaServer,
            endpoint: "http://127.0.0.1:8080".to_string(),
            model: "embeddinggemma-2".to_string(),
            prefix_profile: PrefixProfile::EmbeddingGemma2,
            dimensions: 768,
            batch_size: 8,
            timeout_ms: 120_000,
            multimodal: None,
            managed: false,
            managed_command: Vec::new(),
            gguf: String::new(),
            mmproj: String::new(),
            api_key: String::new(),
        },
        search: SearchConfig {
            mode: SearchMode::Hybrid,
            rrf_k: 60.0,
            weight_vec: 1.0,
            weight_fts: 1.0,
            weight_filename: 0.4,
            default_limit: 20,
            candidates: 50,
            min_score: 0.0,
        },
        chunk: ChunkConfig {
            text: TextChunkConfig {
                mode: TextChunkMode::Chars,
                max_chars: 2000,
                overlap: 200,
                skip_letterless: true,
                numeric_heavy: 0.7,
            },
            pdf: PdfChunkConfig {
                max_pages: 5000,
                first_pass_pages: 30,
                render_dpi: 120,
                embed_text: true,
                embed_page_image: true,
            },
            image: ImageChunkConfig { max_edge_px: 1024 },
            audio: AudioChunkConfig { segment_secs: 30 },
        },
        extract: ExtractConfig {
            poppler_dir: String::new(),
            pdfinfo: String::new(),
            pdftotext: String::new(),
            pdftoppm: String::new(),
            ffmpeg_dir: String::new(),
        },
        files: FilesConfig {
            follow_symlinks: false,
            include_hidden: false,
            respect_gitignore: true,
            heic: HeicMode::Skip,
            max_size: 40 * 1024 * 1024,
            include_globs: Vec::new(),
            exclude_globs: DEFAULT_EXCLUDE_GLOBS
                .iter()
                .map(|g| g.to_string())
                .collect(),
            extension_kind: default_extension_kind(),
        },
    }
}

const DEFAULT_EXCLUDE_GLOBS: &[&str] = &[
    "**/.git/**",
    "**/node_modules/**",
    "**/.cache/**",
    "**/target/**",
];

/// Default extension → kind map (spec §4.1). `.heic`/`.heif` are not listed:
/// they are walk-skipped unless `files.heic = "magick"`.
fn default_extension_kind() -> BTreeMap<String, FileKind> {
    fn put(map: &mut BTreeMap<String, FileKind>, kind: FileKind, exts: &[&str]) {
        for ext in exts {
            map.insert((*ext).to_string(), kind);
        }
    }
    let mut map = BTreeMap::new();
    put(
        &mut map,
        FileKind::Text,
        &[
            "txt", "md", "markdown", "rst", "org", "html", "htm", "csv", "rs", "go", "py", "js",
            "ts", "tsx", "jsx", "c", "h", "cpp", "java", "sh", "toml", "yaml", "yml", "json",
            "nix", "lua", "css", "svg",
        ],
    );
    put(
        &mut map,
        FileKind::Image,
        &["jpg", "jpeg", "png", "webp", "gif", "bmp"],
    );
    put(&mut map, FileKind::Pdf, &["pdf"]);
    put(
        &mut map,
        FileKind::Audio,
        &["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac"],
    );
    map
}

/// The merged TOML for `ferret config show` — comments stripped, `api_key`
/// shown as `"***"` when non-empty.
pub fn to_display_toml(cfg: &Config) -> String {
    let mut root = Table::new();

    let mut index = Table::new();
    index.insert(
        "path".into(),
        Value::from(cfg.index.path.to_string_lossy().into_owned()),
    );
    let roots: Vec<Value> = cfg
        .index
        .roots
        .iter()
        .map(|r| {
            let mut t = Table::new();
            t.insert(
                "path".into(),
                Value::from(r.path.to_string_lossy().into_owned()),
            );
            t.insert("collection".into(), Value::from(r.collection.clone()));
            Value::Table(t)
        })
        .collect();
    index.insert("roots".into(), Value::Array(roots));
    root.insert("index".into(), Value::Table(index));

    let mut embedder = Table::new();
    embedder.insert(
        "backend".into(),
        Value::from(cfg.embedder.backend.to_string()),
    );
    embedder.insert(
        "endpoint".into(),
        Value::from(cfg.embedder.endpoint.clone()),
    );
    embedder.insert("model".into(), Value::from(cfg.embedder.model.clone()));
    embedder.insert(
        "prefix_profile".into(),
        Value::from(cfg.embedder.prefix_profile.to_string()),
    );
    embedder.insert("dimensions".into(), Value::from(cfg.embedder.dimensions));
    embedder.insert("batch_size".into(), Value::from(cfg.embedder.batch_size));
    embedder.insert(
        "timeout_ms".into(),
        Value::from(cfg.embedder.timeout_ms as i64),
    );
    if let Some(multimodal) = cfg.embedder.multimodal {
        embedder.insert("multimodal".into(), Value::from(multimodal));
    }
    embedder.insert("managed".into(), Value::from(cfg.embedder.managed));
    embedder.insert(
        "managed_command".into(),
        Value::Array(
            cfg.embedder
                .managed_command
                .iter()
                .map(|s| Value::from(s.clone()))
                .collect(),
        ),
    );
    embedder.insert("gguf".into(), Value::from(cfg.embedder.gguf.clone()));
    embedder.insert("mmproj".into(), Value::from(cfg.embedder.mmproj.clone()));
    embedder.insert(
        "api_key".into(),
        Value::from(if cfg.embedder.api_key.is_empty() {
            ""
        } else {
            "***"
        }),
    );
    root.insert("embedder".into(), Value::Table(embedder));

    let mut search = Table::new();
    search.insert("mode".into(), Value::from(cfg.search.mode.to_string()));
    search.insert("rrf_k".into(), Value::from(cfg.search.rrf_k));
    search.insert("weight_vec".into(), Value::from(cfg.search.weight_vec));
    search.insert("weight_fts".into(), Value::from(cfg.search.weight_fts));
    search.insert(
        "weight_filename".into(),
        Value::from(cfg.search.weight_filename),
    );
    search.insert(
        "default_limit".into(),
        Value::from(cfg.search.default_limit as i64),
    );
    search.insert(
        "candidates".into(),
        Value::from(cfg.search.candidates as i64),
    );
    search.insert("min_score".into(), Value::from(cfg.search.min_score));
    root.insert("search".into(), Value::Table(search));

    let mut text = Table::new();
    text.insert("mode".into(), Value::from(cfg.chunk.text.mode.to_string()));
    text.insert(
        "max_chars".into(),
        Value::from(cfg.chunk.text.max_chars as i64),
    );
    text.insert("overlap".into(), Value::from(cfg.chunk.text.overlap as i64));
    text.insert(
        "skip_letterless".into(),
        Value::from(cfg.chunk.text.skip_letterless),
    );
    text.insert(
        "numeric_heavy".into(),
        Value::from(cfg.chunk.text.numeric_heavy),
    );
    let mut pdf = Table::new();
    pdf.insert("max_pages".into(), Value::from(cfg.chunk.pdf.max_pages));
    pdf.insert(
        "first_pass_pages".into(),
        Value::from(cfg.chunk.pdf.first_pass_pages),
    );
    pdf.insert("render_dpi".into(), Value::from(cfg.chunk.pdf.render_dpi));
    pdf.insert("embed_text".into(), Value::from(cfg.chunk.pdf.embed_text));
    pdf.insert(
        "embed_page_image".into(),
        Value::from(cfg.chunk.pdf.embed_page_image),
    );
    let mut image = Table::new();
    image.insert(
        "max_edge_px".into(),
        Value::from(cfg.chunk.image.max_edge_px),
    );
    let mut audio = Table::new();
    audio.insert(
        "segment_secs".into(),
        Value::from(cfg.chunk.audio.segment_secs as i64),
    );
    let mut chunk = Table::new();
    chunk.insert("text".into(), Value::Table(text));
    chunk.insert("pdf".into(), Value::Table(pdf));
    chunk.insert("image".into(), Value::Table(image));
    chunk.insert("audio".into(), Value::Table(audio));
    root.insert("chunk".into(), Value::Table(chunk));

    let mut extract = Table::new();
    extract.insert(
        "poppler_dir".into(),
        Value::from(cfg.extract.poppler_dir.clone()),
    );
    extract.insert("pdfinfo".into(), Value::from(cfg.extract.pdfinfo.clone()));
    extract.insert(
        "pdftotext".into(),
        Value::from(cfg.extract.pdftotext.clone()),
    );
    extract.insert("pdftoppm".into(), Value::from(cfg.extract.pdftoppm.clone()));
    extract.insert(
        "ffmpeg_dir".into(),
        Value::from(cfg.extract.ffmpeg_dir.clone()),
    );
    root.insert("extract".into(), Value::Table(extract));

    let mut files = Table::new();
    files.insert(
        "follow_symlinks".into(),
        Value::from(cfg.files.follow_symlinks),
    );
    files.insert(
        "include_hidden".into(),
        Value::from(cfg.files.include_hidden),
    );
    files.insert(
        "respect_gitignore".into(),
        Value::from(cfg.files.respect_gitignore),
    );
    files.insert("heic".into(), Value::from(cfg.files.heic.to_string()));
    files.insert("max_size".into(), Value::from(cfg.files.max_size as i64));
    files.insert(
        "include_globs".into(),
        Value::Array(
            cfg.files
                .include_globs
                .iter()
                .map(|s| Value::from(s.clone()))
                .collect(),
        ),
    );
    files.insert(
        "exclude_globs".into(),
        Value::Array(
            cfg.files
                .exclude_globs
                .iter()
                .map(|s| Value::from(s.clone()))
                .collect(),
        ),
    );
    files.insert(
        "extension_kind".into(),
        Value::Table(
            cfg.files
                .extension_kind
                .iter()
                .map(|(k, v)| (k.clone(), Value::from(v.to_string())))
                .collect::<Table>(),
        ),
    );
    root.insert("files".into(), Value::Table(files));

    toml::to_string(&Value::Table(root)).expect("display TOML serialization is infallible")
}

// ---------------------------------------------------------------------------
// Layer reading and merging
// ---------------------------------------------------------------------------

/// Read and parse one file layer. `read` returning `Err` means the layer is
/// absent (skip), except when `explicit` is set (`--config FILE` → fatal).
fn read_layer(
    path: &Path,
    read: &dyn Fn(&Path) -> Result<String>,
    explicit: bool,
) -> Result<Option<Table>> {
    let text = match read(path) {
        Ok(text) => text,
        Err(e) => {
            if explicit {
                return Err(e);
            }
            return Ok(None);
        }
    };
    let table = toml::from_str::<Table>(&text)
        .map_err(|e| Error::fatal(format!("config `{}`: {e}", path.display())))?;
    check_api_key_mode(path, &table)?;
    Ok(Some(table))
}

/// A loaded TOML containing a non-empty `embedder.api_key` must not be more
/// permissive than 0600 (unix only).
fn check_api_key_mode(path: &Path, table: &Table) -> Result<()> {
    #[cfg(unix)]
    {
        let has_key = table
            .get("embedder")
            .and_then(|v| v.get("api_key"))
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.is_empty());
        if has_key {
            use std::os::unix::fs::PermissionsExt;
            let md = std::fs::metadata(path).map_err(|e| {
                Error::fatal(format!(
                    "config `{}` holds api_key but is unreadable: {e}",
                    path.display()
                ))
            })?;
            let mode = md.permissions().mode();
            if mode & 0o077 != 0 {
                return Err(Error::fatal(format!(
                    "config `{}` contains api_key but is mode {:o}; chmod 600 it",
                    path.display(),
                    mode & 0o777
                )));
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (path, table);
    }
    Ok(())
}

/// Last-wins deep merge of TOML tables.
fn deep_merge(dst: &mut Table, src: Table) {
    for (key, value) in src {
        match (dst.get_mut(&key), value) {
            (Some(Value::Table(existing)), Value::Table(incoming)) => {
                deep_merge(existing, incoming)
            }
            (_, value) => {
                dst.insert(key, value);
            }
        }
    }
}

/// Every scalar config path settable via env, used to probe
/// `FERRET__SECTION__KEY` (nested sections via more `__`).
const ENV_PATHS: &[&str] = &[
    "index.path",
    "embedder.backend",
    "embedder.endpoint",
    "embedder.model",
    "embedder.prefix_profile",
    "embedder.dimensions",
    "embedder.batch_size",
    "embedder.timeout_ms",
    "embedder.multimodal",
    "embedder.managed",
    "embedder.gguf",
    "embedder.mmproj",
    "embedder.api_key",
    "search.mode",
    "search.rrf_k",
    "search.weight_vec",
    "search.weight_fts",
    "search.weight_filename",
    "search.default_limit",
    "search.candidates",
    "search.min_score",
    "chunk.text.mode",
    "chunk.text.max_chars",
    "chunk.text.overlap",
    "chunk.text.skip_letterless",
    "chunk.text.numeric_heavy",
    "chunk.pdf.max_pages",
    "chunk.pdf.first_pass_pages",
    "chunk.pdf.render_dpi",
    "chunk.pdf.embed_text",
    "chunk.pdf.embed_page_image",
    "chunk.image.max_edge_px",
    "chunk.audio.segment_secs",
    "extract.poppler_dir",
    "extract.pdfinfo",
    "extract.pdftotext",
    "extract.pdftoppm",
    "files.follow_symlinks",
    "files.include_hidden",
    "files.respect_gitignore",
    "files.heic",
    "files.max_size",
];

fn apply_env(merged: &mut Table, env: &dyn Fn(&str) -> Option<String>) -> Result<bool> {
    let mut applied = false;
    for path in ENV_PATHS {
        let key = format!("FERRET__{}", path.replace('.', "__").to_uppercase());
        if let Some(raw) = env(&key) {
            let value = coerce_env_value(path, &raw)?;
            set_path(merged, path, value);
            applied = true;
        }
    }
    Ok(applied)
}

/// Coerce an env string into the TOML value type its config field expects.
fn coerce_env_value(path: &str, raw: &str) -> Result<Value> {
    Ok(match path {
        "index.path"
        | "embedder.backend"
        | "embedder.endpoint"
        | "embedder.model"
        | "embedder.prefix_profile"
        | "embedder.gguf"
        | "embedder.mmproj"
        | "embedder.api_key"
        | "extract.poppler_dir"
        | "extract.pdfinfo"
        | "extract.pdftotext"
        | "extract.pdftoppm"
        | "files.max_size"
        | "search.mode"
        | "chunk.text.mode"
        | "files.heic" => Value::from(raw),
        // TOML integers are i64; the target field types are enforced when
        // the merged table is deserialized into RawConfig.
        "embedder.dimensions"
        | "embedder.batch_size"
        | "chunk.pdf.max_pages"
        | "chunk.pdf.first_pass_pages"
        | "chunk.pdf.render_dpi"
        | "chunk.image.max_edge_px" => Value::from(i64::from(parse_env::<u32>(path, raw)?)),
        "chunk.audio.segment_secs" => Value::from(parse_env::<u64>(path, raw)? as i64),
        "embedder.timeout_ms" => Value::from(parse_env::<u64>(path, raw)? as i64),
        "search.default_limit"
        | "search.candidates"
        | "chunk.text.max_chars"
        | "chunk.text.overlap" => Value::from(parse_env::<usize>(path, raw)? as i64),
        "search.rrf_k"
        | "search.weight_vec"
        | "search.weight_fts"
        | "search.weight_filename"
        | "search.min_score"
        | "chunk.text.numeric_heavy" => Value::from(parse_env::<f64>(path, raw)?),
        "embedder.multimodal"
        | "embedder.managed"
        | "chunk.text.skip_letterless"
        | "chunk.pdf.embed_text"
        | "chunk.pdf.embed_page_image"
        | "files.follow_symlinks"
        | "files.include_hidden"
        | "files.respect_gitignore" => Value::from(parse_env::<bool>(path, raw)?),
        _ => Value::from(raw),
    })
}

fn parse_env<T: std::str::FromStr>(path: &str, raw: &str) -> Result<T> {
    raw.parse::<T>()
        .map_err(|_| Error::fatal(format!("config: invalid value `{raw}` for `{path}`")))
}

/// Insert a value at a dotted path, creating intermediate tables as needed.
fn set_path(table: &mut Table, dotted: &str, value: Value) {
    let mut segments: Vec<&str> = dotted.split('.').collect();
    let last = segments.pop().expect("non-empty path");
    let mut current = table;
    for segment in segments {
        let entry = current
            .entry(segment.to_string())
            .or_insert_with(|| Value::Table(Table::new()));
        if !entry.is_table() {
            *entry = Value::Table(Table::new());
        }
        current = entry.as_table_mut().expect("just-created table");
    }
    current.insert(last.to_string(), value);
}

// ---------------------------------------------------------------------------
// Raw (partial) TOML shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawConfig {
    index: RawIndex,
    embedder: RawEmbedder,
    search: RawSearch,
    chunk: RawChunk,
    extract: RawExtract,
    files: RawFiles,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawIndex {
    path: Option<PathBuf>,
    roots: Option<RootSpec>,
}

/// Both root TOML shapes. Mixing `roots = ["…"]` and `[[index.roots]]` in one
/// file fails to match either variant — a config error (exit 1).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RootSpec {
    Paths(Vec<String>),
    Tables(Vec<RawRoot>),
}

#[derive(Debug, Deserialize)]
struct RawRoot {
    path: String,
    collection: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawEmbedder {
    backend: Option<Backend>,
    endpoint: Option<String>,
    model: Option<String>,
    prefix_profile: Option<PrefixProfile>,
    dimensions: Option<u32>,
    batch_size: Option<u32>,
    timeout_ms: Option<u64>,
    multimodal: Option<bool>,
    managed: Option<bool>,
    managed_command: Option<Vec<String>>,
    gguf: Option<String>,
    mmproj: Option<String>,
    api_key: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawSearch {
    mode: Option<SearchMode>,
    rrf_k: Option<f64>,
    weight_vec: Option<f64>,
    weight_fts: Option<f64>,
    weight_filename: Option<f64>,
    default_limit: Option<usize>,
    candidates: Option<usize>,
    min_score: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawChunk {
    text: RawTextChunk,
    pdf: RawPdfChunk,
    image: RawImageChunk,
    audio: RawAudioChunk,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawTextChunk {
    mode: Option<TextChunkMode>,
    max_chars: Option<usize>,
    overlap: Option<usize>,
    skip_letterless: Option<bool>,
    numeric_heavy: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawPdfChunk {
    max_pages: Option<u32>,
    first_pass_pages: Option<u32>,
    render_dpi: Option<u32>,
    embed_text: Option<bool>,
    embed_page_image: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawImageChunk {
    max_edge_px: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawAudioChunk {
    segment_secs: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawExtract {
    poppler_dir: Option<String>,
    pdfinfo: Option<String>,
    pdftotext: Option<String>,
    pdftoppm: Option<String>,
    ffmpeg_dir: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawFiles {
    follow_symlinks: Option<bool>,
    include_hidden: Option<bool>,
    respect_gitignore: Option<bool>,
    heic: Option<HeicMode>,
    max_size: Option<SizeSpec>,
    include_globs: Option<Vec<String>>,
    exclude_globs: Option<Vec<String>>,
    extension_kind: Option<BTreeMap<String, FileKind>>,
}

/// `max_size` accepts integer bytes or `KiB` / `MiB` / `GiB` (1024-based).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SizeSpec {
    Bytes(u64),
    Human(String),
}

fn parse_size(spec: SizeSpec) -> Result<u64> {
    match spec {
        SizeSpec::Bytes(n) => Ok(n),
        SizeSpec::Human(s) => {
            let s = s.trim();
            // Bare digits are integer bytes (the error below promises it).
            if let Ok(n) = s.parse::<u64>() {
                return Ok(n);
            }
            for (suffix, mult) in [
                ("KiB", 1024u64),
                ("MiB", 1024 * 1024),
                ("GiB", 1024 * 1024 * 1024),
            ] {
                if let Some(num) = s.strip_suffix(suffix) {
                    let n: u64 = num.trim().parse().map_err(|_| {
                        Error::fatal(format!("config: invalid files.max_size `{s}`"))
                    })?;
                    return Ok(n * mult);
                }
            }
            Err(Error::fatal(format!(
                "config: invalid files.max_size `{s}` (integer bytes or KiB/MiB/GiB)"
            )))
        }
    }
}

fn table_into_raw(merged: Table) -> Result<RawConfig> {
    let text =
        toml::to_string(&Value::Table(merged)).map_err(|e| Error::fatal(format!("config: {e}")))?;
    toml::from_str::<RawConfig>(&text).map_err(|e| Error::fatal(format!("config: {e}")))
}

// ---------------------------------------------------------------------------
// Merging raw values into compiled defaults
// ---------------------------------------------------------------------------

fn build_config(raw: RawConfig, dirs: &FerretDirs) -> Result<Config> {
    let mut cfg = compiled_defaults(dirs);

    if let Some(path) = raw.index.path {
        cfg.index.path = path;
    }
    if let Some(roots) = raw.index.roots {
        cfg.index.roots = build_roots(roots)?;
    }

    let e = raw.embedder;
    if let Some(v) = e.backend {
        cfg.embedder.backend = v;
    }
    if let Some(v) = e.endpoint {
        cfg.embedder.endpoint = v;
    }
    if let Some(v) = e.model {
        cfg.embedder.model = v;
    }
    if let Some(v) = e.dimensions {
        cfg.embedder.dimensions = v;
    }
    if let Some(v) = e.batch_size {
        cfg.embedder.batch_size = v;
    }
    if let Some(v) = e.timeout_ms {
        cfg.embedder.timeout_ms = v;
    }
    if let Some(v) = e.multimodal {
        cfg.embedder.multimodal = Some(v);
    }
    if let Some(v) = e.managed {
        cfg.embedder.managed = v;
    }
    if let Some(v) = e.managed_command {
        cfg.embedder.managed_command = v;
    }
    if let Some(v) = e.gguf {
        cfg.embedder.gguf = v;
    }
    if let Some(v) = e.mmproj {
        cfg.embedder.mmproj = v;
    }
    if let Some(v) = e.api_key {
        cfg.embedder.api_key = v;
    }
    // prefix_profile default depends on the *merged* backend.
    cfg.embedder.prefix_profile = e.prefix_profile.unwrap_or(match cfg.embedder.backend {
        Backend::LitertLm => PrefixProfile::LitertLm,
        _ => PrefixProfile::EmbeddingGemma2,
    });

    let s = raw.search;
    if let Some(v) = s.mode {
        cfg.search.mode = v;
    }
    if let Some(v) = s.rrf_k {
        cfg.search.rrf_k = v;
    }
    if let Some(v) = s.weight_vec {
        cfg.search.weight_vec = v;
    }
    if let Some(v) = s.weight_fts {
        cfg.search.weight_fts = v;
    }
    if let Some(v) = s.weight_filename {
        cfg.search.weight_filename = v;
    }
    if let Some(v) = s.default_limit {
        cfg.search.default_limit = v;
    }
    if let Some(v) = s.candidates {
        cfg.search.candidates = v;
    }
    if let Some(v) = s.min_score {
        cfg.search.min_score = v;
    }

    let t = raw.chunk.text;
    if let Some(v) = t.mode {
        cfg.chunk.text.mode = v;
    }
    if let Some(v) = t.max_chars {
        cfg.chunk.text.max_chars = v;
    }
    if let Some(v) = t.overlap {
        cfg.chunk.text.overlap = v;
    }
    if let Some(v) = t.skip_letterless {
        cfg.chunk.text.skip_letterless = v;
    }
    if let Some(v) = t.numeric_heavy {
        cfg.chunk.text.numeric_heavy = v;
    }

    let p = raw.chunk.pdf;
    if let Some(v) = p.max_pages {
        cfg.chunk.pdf.max_pages = v;
    }
    if let Some(v) = p.first_pass_pages {
        cfg.chunk.pdf.first_pass_pages = v;
    }
    if let Some(v) = p.render_dpi {
        cfg.chunk.pdf.render_dpi = v;
    }
    if let Some(v) = p.embed_text {
        cfg.chunk.pdf.embed_text = v;
    }
    if let Some(v) = p.embed_page_image {
        cfg.chunk.pdf.embed_page_image = v;
    }
    if let Some(v) = raw.chunk.image.max_edge_px {
        cfg.chunk.image.max_edge_px = v;
    }
    if let Some(v) = raw.chunk.audio.segment_secs {
        cfg.chunk.audio.segment_secs = v;
    }

    let x = raw.extract;
    if let Some(v) = x.poppler_dir {
        cfg.extract.poppler_dir = v;
    }
    if let Some(v) = x.pdfinfo {
        cfg.extract.pdfinfo = v;
    }
    if let Some(v) = x.pdftotext {
        cfg.extract.pdftotext = v;
    }
    if let Some(v) = x.pdftoppm {
        cfg.extract.pdftoppm = v;
    }
    if let Some(v) = x.ffmpeg_dir {
        cfg.extract.ffmpeg_dir = v;
    }

    let f = raw.files;
    if let Some(v) = f.follow_symlinks {
        cfg.files.follow_symlinks = v;
    }
    if let Some(v) = f.include_hidden {
        cfg.files.include_hidden = v;
    }
    if let Some(v) = f.respect_gitignore {
        cfg.files.respect_gitignore = v;
    }
    if let Some(v) = f.heic {
        cfg.files.heic = v;
    }
    if let Some(v) = f.max_size {
        cfg.files.max_size = parse_size(v)?;
    }
    if let Some(v) = f.include_globs {
        cfg.files.include_globs = v;
    }
    if let Some(v) = f.exclude_globs {
        cfg.files.exclude_globs = v;
    }
    // User entries extend/override the compiled default map; the merged map
    // keeps every compiled default (spec §13 meta fingerprint).
    if let Some(v) = f.extension_kind {
        for (ext, kind) in v {
            cfg.files.extension_kind.insert(ext, kind);
        }
    }

    Ok(cfg)
}

/// Build root entries: tilde-expand each path, canonicalize if it exists, and
/// derive the collection name (shorthand: last component of the resulting
/// path; `[[index.roots]] collection` overrides).
fn build_roots(spec: RootSpec) -> Result<Vec<RootEntry>> {
    let entries = match spec {
        RootSpec::Paths(paths) => paths
            .iter()
            .map(|p| {
                let expanded = expand_tilde(p);
                let path = std::fs::canonicalize(&expanded).unwrap_or(expanded);
                // `/` has no file name; fall back to the full path so the
                // collection is never empty (mirrors `init`).
                let collection = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned());
                RootEntry { path, collection }
            })
            .collect::<Vec<_>>(),
        RootSpec::Tables(roots) => roots
            .iter()
            .map(|r| {
                let expanded = expand_tilde(&r.path);
                let path = std::fs::canonicalize(&expanded).unwrap_or(expanded);
                RootEntry {
                    path,
                    collection: r.collection.clone(),
                }
            })
            .collect::<Vec<_>>(),
    };
    Ok(entries)
}

pub(crate) fn expand_tilde(p: &str) -> PathBuf {
    let home = dirs::home_dir().unwrap_or_default();
    if p == "~" {
        if home.as_os_str().is_empty() {
            return PathBuf::from(p);
        }
        home
    } else if let Some(rest) = p.strip_prefix("~/") {
        if home.as_os_str().is_empty() {
            return PathBuf::from(p);
        }
        home.join(rest)
    } else {
        PathBuf::from(p)
    }
}

// ---------------------------------------------------------------------------
// Validation (all exit 1)
// ---------------------------------------------------------------------------

fn validate(cfg: &Config) -> Result<()> {
    if cfg.chunk.text.overlap >= cfg.chunk.text.max_chars {
        return Err(Error::fatal(format!(
            "config: chunk.text.overlap ({}) must be less than chunk.text.max_chars ({})",
            cfg.chunk.text.overlap, cfg.chunk.text.max_chars
        )));
    }
    if cfg.chunk.audio.segment_secs == 0 {
        return Err(Error::fatal(
            "config: chunk.audio.segment_secs must be at least 1".to_string(),
        ));
    }
    if !matches!(cfg.embedder.dimensions, 768 | 512 | 256 | 128) {
        return Err(Error::fatal(format!(
            "config: embedder.dimensions ({}) must be one of 768, 512, 256, 128",
            cfg.embedder.dimensions
        )));
    }
    let mut seen = BTreeSet::new();
    for root in &cfg.index.roots {
        if !seen.insert(root.collection.clone()) {
            return Err(Error::fatal(format!(
                "config: duplicate collection `{}` (collection names must be unique)",
                root.collection
            )));
        }
    }
    Ok(())
}
