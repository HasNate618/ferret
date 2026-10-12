//! CLI: clap structure, `init`, `config show` / `config path`, exit codes.
//!
//! Exit 2 usage (unknown flag, `search` with nothing to search for, `init`
//! onto an existing file) comes back as `Error::usage`; everything else the
//! CLI reports as `Error::fatal` (exit 1). `main` prints the `Display` on
//! stderr and exits with `err.exit_code()`.
//!
//! Subcommands whose behavior lands in later tasks are stubs: `index` and
//! `status` return `Error::fatal("not implemented")` after their usage
//! validation, and `rm` / `gc` are usage-ok no-ops. `search` runs the
//! hybrid pipeline (Task 8).

use crate::config::{self, LayerSource, LoadOpts};
use crate::embedder;
use crate::error::{Error, Result};
use crate::paths::ferret_dirs;
use crate::search::{self, FilterFlags, SearchRequest};
use crate::store;
use crate::types::{FileKind, SearchMode};
use clap::{Parser, Subcommand};
use std::collections::BTreeSet;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Captured output of an in-process run, for tests (no binary spawn needed).
#[derive(Debug)]
pub struct RunOutput {
    pub stdout: String,
    pub stderr: String,
}

/// Run as if invoked with `args` (argv[0] included), empty non-TTY stdin.
pub fn try_run(args: &[&str]) -> Result<RunOutput> {
    try_run_with(args, "", false)
}

/// Run as if invoked with `args` (argv[0] included) on the given stdin.
pub fn try_run_with(args: &[&str], stdin: &str, is_tty: bool) -> Result<RunOutput> {
    let mut reader = stdin.as_bytes();
    let mut stdout: Vec<u8> = Vec::new();
    let mut stderr: Vec<u8> = Vec::new();
    execute(args, &mut reader, is_tty, &mut stdout, &mut stderr)?;
    Ok(RunOutput {
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
    })
}

/// Real entry point: process argv, process stdin (one line if not a TTY).
pub fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let is_tty = std::io::stdin().is_terminal();
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    execute(
        &refs,
        &mut lock,
        is_tty,
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    )
}

#[derive(Debug, Parser)]
#[command(
    name = "ferret",
    version,
    about = "Local-first semantic multimodal file search"
)]
struct Cli {
    /// Use FILE as the user config layer instead of `{config_dir}/config.toml`.
    /// A read error for this file is fatal.
    #[arg(long, value_name = "FILE", global = true)]
    config: Option<PathBuf>,
    /// Override `index.path`.
    #[arg(long, value_name = "PATH", global = true)]
    index: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
// The flat field lists are fixed by the CLI spec; the resulting variant
// size is fine for a parse-once value.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Write a starter `{config_dir}/config.toml` with the given roots.
    Init {
        /// Root to index (repeatable; collection = last path component).
        #[arg(long = "root", value_name = "DIR", action = clap::ArgAction::Append)]
        roots: Vec<PathBuf>,
        /// Overwrite an existing config file.
        #[arg(long)]
        force: bool,
    },
    /// Index configured roots plus any extra PATHs.
    Index {
        /// Extra paths in addition to configured roots.
        #[arg(value_name = "PATH")]
        paths: Vec<PathBuf>,
        /// Emit NDJSON events to stdout instead of the summary.
        #[arg(long)]
        json: bool,
        /// Per-file errors become exit 1.
        #[arg(long)]
        strict: bool,
    },
    /// Semantic / hybrid search.
    Search {
        /// Query text; omitted reads one line from stdin (only when stdin is
        /// not a TTY). A filter flag alone is also enough to run.
        query: Option<String>,
        /// Emit JSON to stdout.
        #[arg(long)]
        json: bool,
        /// File kind filter (repeatable; comma-separated values union).
        #[arg(long, value_name = "KIND", action = clap::ArgAction::Append)]
        kind: Vec<String>,
        /// Extension filter, e.g. `--ext md` (repeatable; comma-separated
        /// values union).
        #[arg(long, value_name = "EXT", action = clap::ArgAction::Append)]
        ext: Vec<String>,
        /// Restrict to files under DIR. On a large index this can starve the
        /// semantic arm of the search; hybrid mode still has FTS.
        #[arg(long, value_name = "DIR")]
        under: Option<PathBuf>,
        /// Restrict to a collection.
        #[arg(long)]
        collection: Option<String>,
        /// Filter by file name GLOB (case-sensitive, e.g. `*.md`). Like
        /// `--under`, it can starve the semantic arm on a large index; hybrid
        /// still has FTS.
        #[arg(long, value_name = "GLOB")]
        name: Option<String>,
        /// Only files modified on or after DATE (`YYYY-MM-DD`).
        #[arg(long, value_name = "DATE")]
        after: Option<String>,
        /// Only files modified on or before DATE (`YYYY-MM-DD`).
        #[arg(long, value_name = "DATE")]
        before: Option<String>,
        /// Drop results below this fused score (0..1).
        #[arg(long)]
        min_score: Option<f64>,
        /// Max results (default `search.default_limit`).
        #[arg(long)]
        limit: Option<usize>,
        /// Fusion mode: `hybrid`, `semantic`, or `keyword`.
        #[arg(long)]
        mode: Option<SearchMode>,
        /// Bias results toward code-like content.
        #[arg(long)]
        code: bool,
    },
    /// Delete those paths from the index.
    Rm {
        /// Paths to remove.
        #[arg(value_name = "PATH")]
        paths: Vec<PathBuf>,
    },
    /// Delete DB rows whose files are gone.
    Gc,
    /// Index status.
    Status {
        /// Emit JSON to stdout.
        #[arg(long)]
        json: bool,
    },
    /// Show or locate config.
    Config {
        #[command(subcommand)]
        command: ConfigCmd,
    },
}

#[derive(Debug, Subcommand)]
enum ConfigCmd {
    /// Print the merged TOML (api_key redacted).
    Show,
    /// Print the config file paths that loaded, plus `defaults` / `env` /
    /// `flags` layers that contributed.
    Path,
}

fn execute(
    args: &[&str],
    stdin: &mut dyn std::io::BufRead,
    is_tty: bool,
    stdout: &mut dyn Write,
    _stderr: &mut dyn Write,
) -> Result<()> {
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(e)
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            write!(stdout, "{}", e.render())?;
            return Ok(());
        }
        Err(e) => return Err(Error::usage(e.to_string().trim_end_matches(['\n', '\r']))),
    };

    let opts = LoadOpts {
        config_flag: cli.config,
        index_flag: cli.index,
    };

    match cli.command {
        Command::Init { roots, force } => init(&roots, force, stdout),
        Command::Config { command } => match command {
            ConfigCmd::Show => config_show(&opts, stdout),
            ConfigCmd::Path => config_path(&opts, stdout),
        },
        Command::Index { .. } => Err(Error::fatal("not implemented")),
        Command::Status { .. } => Err(Error::fatal("not implemented")),
        Command::Gc => Ok(()),        // stub; Task 13 fills this in
        Command::Rm { .. } => Ok(()), // stub; Task 13 fills this in
        Command::Search {
            query,
            kind,
            ext,
            under,
            collection,
            name,
            after,
            before,
            min_score,
            limit,
            mode,
            code,
            json,
        } => {
            let mut query = query;
            // No QUERY and stdin is not a TTY: read one line. Never prompt.
            if query.is_none() && !is_tty {
                let mut line = String::new();
                stdin
                    .read_line(&mut line)
                    .map_err(|e| Error::fatal(e.to_string()))?;
                let line = line.trim_end_matches(['\n', '\r']);
                if !line.is_empty() {
                    query = Some(line.to_string());
                }
            }
            let has_filter = !kind.is_empty()
                || !ext.is_empty()
                || under.is_some()
                || collection.is_some()
                || name.is_some()
                || after.is_some()
                || before.is_some()
                || min_score.is_some()
                || limit.is_some()
                || mode.is_some()
                || code;
            if query.is_none() && !has_filter {
                return Err(Error::usage(
                    "ferret search needs a QUERY or a filter flag (e.g. --name, --under, --ext)",
                ));
            }
            run_search_cli(
                &opts,
                query.as_deref(),
                SearchArgs {
                    kind,
                    ext,
                    under,
                    collection,
                    name,
                    after,
                    before,
                    min_score,
                    limit,
                    mode,
                    code,
                    json,
                },
                stdout,
            )?;
            Ok(())
        }
    }
}

/// Flattened `Command::Search` payload for `run_search_cli`.
struct SearchArgs {
    kind: Vec<String>,
    ext: Vec<String>,
    under: Option<PathBuf>,
    collection: Option<String>,
    name: Option<String>,
    after: Option<String>,
    before: Option<String>,
    min_score: Option<f64>,
    limit: Option<usize>,
    mode: Option<SearchMode>,
    code: bool,
    json: bool,
}

/// `ferret search`: parse the query, run the hybrid pipeline, render.
/// Fatal errors (config, DB, embedder) propagate as exit 1 with empty
/// stdout; zero hits is exit 0 with an empty hit list (spec §6.5).
fn run_search_cli(
    opts: &LoadOpts,
    query: Option<&str>,
    args: SearchArgs,
    stdout: &mut dyn Write,
) -> Result<()> {
    let cfg = config::load(opts)?;
    let flags = FilterFlags {
        kind: split_kinds(&args.kind)?,
        ext: split_strings(&args.ext),
        under: args.under,
        collection: args.collection,
        name: args.name,
        after: args.after,
        before: args.before,
        code: args.code,
    };
    let parsed = search::parse_query(query.unwrap_or(""), &flags)?;
    let embedder = embedder::build_embedder(&cfg, &|k| std::env::var(k).ok())?;
    let store = store::open(&cfg.index.path, &cfg, embedder.images_supported())?;
    let req = SearchRequest {
        parsed,
        mode: args.mode.unwrap_or(cfg.search.mode),
        limit: args.limit.unwrap_or(cfg.search.default_limit),
        min_score: args.min_score.unwrap_or(cfg.search.min_score),
        json: args.json,
    };
    let out = search::run_search(&store, embedder.as_ref(), &cfg, req)?;
    if args.json {
        writeln!(stdout, "{}", search::render_json(&out))?;
    } else {
        let human = search::render_human(&out);
        if !human.is_empty() {
            writeln!(stdout, "{human}")?;
        }
    }
    Ok(())
}

/// Comma-split repeatable `--kind` values (`--kind pdf,md` == `--kind pdf
/// --kind md`); an unknown kind is a usage error (exit 2).
fn split_kinds(values: &[String]) -> Result<Vec<FileKind>> {
    let mut out = Vec::new();
    for v in values {
        for part in v.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            out.push(
                FileKind::from_str(part)
                    .map_err(|e| Error::usage(format!("--kind: {e}")))?,
            );
        }
    }
    Ok(out)
}

/// Comma-split repeatable `--ext` values; `parse_query` lowercases and
/// strips any leading dot.
fn split_strings(values: &[String]) -> Vec<String> {
    values
        .iter()
        .flat_map(|v| v.split(','))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// `ferret init`: create `{config_dir}` (0700), write a short TOML with
/// `roots` from `--root`. Never writes `api_key` or `multimodal`; never
/// contacts the network. Existing file: usage error (exit 2) unless
/// `--force`. Two roots whose collection names collide: fatal (exit 1).
fn init(roots: &[PathBuf], force: bool, stdout: &mut dyn Write) -> Result<()> {
    let dirs = ferret_dirs(&|k| std::env::var(k).ok());

    // Collection = last component of the canonical root; unique.
    let mut seen = BTreeSet::new();
    for root in roots {
        let canon = std::fs::canonicalize(root).unwrap_or_else(|_| root.clone());
        let name = canon
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| canon.to_string_lossy().into_owned());
        if !seen.insert(name.clone()) {
            return Err(Error::fatal(format!(
                "duplicate collection `{name}` from root `{}`; collection names must be unique",
                root.display()
            )));
        }
    }

    std::fs::create_dir_all(&dirs.config_dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&dirs.config_dir)?.permissions();
        perms.set_mode(0o700);
        std::fs::set_permissions(&dirs.config_dir, perms)?;
    }

    let file = dirs.config_dir.join("config.toml");
    if file.exists() && !force {
        return Err(Error::usage(format!(
            "config file {} already exists; use --force to overwrite it",
            file.display()
        )));
    }

    let mut table = toml::Table::new();
    let mut index = toml::Table::new();
    index.insert(
        "roots".into(),
        toml::Value::Array(
            roots
                .iter()
                .map(|r| toml::Value::from(r.to_string_lossy().into_owned()))
                .collect(),
        ),
    );
    table.insert("index".into(), toml::Value::Table(index));
    let body = format!(
        "# ferret config; `ferret config show` prints the fully merged values\n{}",
        toml::to_string(&toml::Value::Table(table))
            .map_err(|e| Error::fatal(format!("serialize init config: {e}")))?
    );
    std::fs::write(&file, body)?;

    writeln!(stdout, "wrote {}", file.display())?;
    Ok(())
}

/// `ferret config show`: merged TOML, comments stripped, `api_key` redacted.
fn config_show(opts: &LoadOpts, stdout: &mut dyn Write) -> Result<()> {
    let cfg = config::load(opts)?;
    write!(stdout, "{}", config::to_display_toml(&cfg))?;
    Ok(())
}

/// `ferret config path`: loaded file paths one per line, then `defaults` /
/// `env` / `flags` if those layers contributed.
fn config_path(opts: &LoadOpts, stdout: &mut dyn Write) -> Result<()> {
    let (_cfg, layers) = load_with_layers(opts)?;
    for layer in &layers {
        let path = match layer {
            LayerSource::Etc => Some(Path::new("/etc/ferret/config.toml")),
            LayerSource::UserFile(p) | LayerSource::CwdFile(p) => Some(p.as_path()),
            LayerSource::Defaults | LayerSource::Env | LayerSource::Flags => None,
        };
        if let Some(path) = path {
            writeln!(stdout, "{}", path.display())?;
        }
    }
    for layer in &layers {
        match layer {
            LayerSource::Defaults => writeln!(stdout, "defaults")?,
            LayerSource::Env => writeln!(stdout, "env")?,
            LayerSource::Flags => writeln!(stdout, "flags")?,
            LayerSource::Etc | LayerSource::UserFile(_) | LayerSource::CwdFile(_) => {}
        }
    }
    Ok(())
}

/// `config::load` with the layers it merged, using the real environment,
/// filesystem, and cwd (the same wiring `config::load` uses).
fn load_with_layers(opts: &LoadOpts) -> Result<(crate::config::Config, Vec<LayerSource>)> {
    let dirs = ferret_dirs(&|k| std::env::var(k).ok());
    let cwd = std::env::current_dir()?;
    config::load_from(opts, &dirs, &|k| std::env::var(k).ok(), &cwd, &|p| {
        std::fs::read_to_string(p).map_err(|e| Error::fatal(e.to_string()))
    })
}
