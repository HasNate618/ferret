//! Indexing pipeline (spec §8 walk → extract → embed → write, §10 events).
//!
//! `walk` enumerates the corpus and classifies paths; `run` drives the
//! pipeline: fingerprint-gated re-extraction, per-file transactions, the
//! per-file extract-error path (which never aborts the walk), and the
//! stderr summary / NDJSON events.

pub mod skip;
pub mod walk;

pub use skip::SkipReason;
pub use walk::{kind_for_ext, walk, WalkEvent};

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::Config;
use crate::embedder::{Embedder, EmbedInput};
use crate::error::{Error, Result};
use crate::extract::{self, Chunk, EmbedDecision};
use crate::fingerprint;
use crate::paths::ferret_dirs;
use crate::store::{self, NewChunk, NewFile, Store};
use crate::types::{FileKind, HeicMode};

/// Index run counters (stderr summary: seen / skipped / embedded / errors).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexSummary {
    pub seen: u64,
    pub skipped: u64,
    pub embedded: u64,
    pub errors: u64,
}

/// Index every configured root plus `extra_paths`.
///
/// Per-file extract errors set `files.error`, leave `complete = 0`, and
/// continue (spec §4.5); `--strict` turns any error into exit 1 after the
/// walk. Fatal failures (DB, embedder unreachable, schema/model mismatch)
/// abort with the error.
pub fn run(
    cfg: &Config,
    embedder: &dyn Embedder,
    extra_paths: Vec<PathBuf>,
    json_ndjson: bool,
    strict: bool,
    out: &mut dyn std::io::Write,
) -> Result<IndexSummary> {
    // Collection names are unique across roots; an extra path outside every
    // root must not reuse one.
    check_collections(cfg, &extra_paths)?;

    let images_supported = embedder.images_supported();
    let store = store::open(&cfg.index.path, cfg, images_supported)?;
    let config_fp = fingerprint::config_fingerprint(cfg, images_supported);

    // Pipeline granularity: delete the chunk side a changed fingerprint
    // invalidates, so per-file re-extraction has a clean slate while the
    // other side's vec rows survive.
    apply_pipeline_changes(&store, cfg, images_supported, &config_fp)?;

    let dirs = ferret_dirs(&|k| std::env::var(k).ok());
    let mut summary = IndexSummary::default();

    let events = walk::walk(
        &cfg.index.roots,
        &extra_paths,
        &cfg.files,
        cfg.files.follow_symlinks,
    )?;
    summary.seen = events.len() as u64;

    for event in events {
        match event {
            WalkEvent::File {
                path,
                collection,
                kind,
                ext,
            } => index_file(
                &store,
                embedder,
                cfg,
                &config_fp,
                images_supported,
                &dirs.cache_dir,
                EventSink {
                    json: json_ndjson,
                    out,
                },
                path,
                collection,
                kind,
                ext,
                &mut summary,
            )?,
            WalkEvent::Skip { path, reason } => {
                summary.skipped += 1;
                emit_skip(json_ndjson, &path, reason, out)?;
                // A path that now matches a walk-skip loses its rows.
                store.delete_file_by_path(&path.to_string_lossy())?;
            }
        }
    }

    // Record the fingerprints this run was built with (also when zero files
    // were walked, so the next run knows a reindex is not pending), plus
    // the image capability `status` reports without an embedder call.
    store.set_meta("config_fingerprint", &config_fp)?;
    store.set_meta("text_pipeline", &fingerprint::text_pipeline_json(cfg))?;
    store.set_meta(
        "image_pipeline",
        &fingerprint::image_pipeline_json(cfg, images_supported),
    )?;
    store.set_meta(
        "images_supported",
        if images_supported { "true" } else { "false" },
    )?;

    if strict && summary.errors > 0 {
        return Err(Error::fatal(format!(
            "{} file(s) failed to index (--strict)",
            summary.errors
        )));
    }
    Ok(summary)
}

/// NDJSON/stdout sink shared by the per-file helpers.
struct EventSink<'a> {
    json: bool,
    out: &'a mut dyn std::io::Write,
}

fn emit_ok(json: bool, path: &Path, out: &mut dyn std::io::Write) -> Result<()> {
    if json {
        writeln!(
            out,
            "{}",
            serde_json::json!({
                "event": "file",
                "path": path.to_string_lossy(),
                "status": "ok",
                "reason": serde_json::Value::Null,
                "error": serde_json::Value::Null,
            })
        )?;
    }
    Ok(())
}

fn emit_skip(
    json: bool,
    path: &Path,
    reason: SkipReason,
    out: &mut dyn std::io::Write,
) -> Result<()> {
    if json {
        writeln!(
            out,
            "{}",
            serde_json::json!({
                "event": "file",
                "path": path.to_string_lossy(),
                "status": "skip",
                "reason": reason.as_str(),
                "error": serde_json::Value::Null,
            })
        )?;
    }
    Ok(())
}

/// Collection-name uniqueness across roots and non-under-root extras.
fn check_collections(cfg: &Config, extra: &[PathBuf]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for r in &cfg.index.roots {
        if !seen.insert(r.collection.clone()) {
            return Err(Error::fatal(format!(
                "collection `{}` is used by more than one root; collection names must be unique",
                r.collection
            )));
        }
    }
    for p in extra {
        if cfg.index.roots.iter().any(|r| p.starts_with(&r.path)) {
            continue; // inherits the root's collection
        }
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !seen.insert(name.clone()) {
            return Err(Error::fatal(format!(
                "extra path `{}` would reuse collection `{name}`; collection names must be unique",
                p.display()
            )));
        }
    }
    Ok(())
}

/// Which stored meta layers changed decides which chunk side is dropped
/// before the walk (spec §5.2).
fn apply_pipeline_changes(
    store: &Store,
    cfg: &Config,
    images_supported: bool,
    config_fp: &str,
) -> Result<()> {
    let stored_fp = store.get_meta("config_fingerprint")?;
    let Some(stored_fp) = stored_fp else {
        return Ok(()); // fresh index: nothing stored yet
    };
    if stored_fp == config_fp {
        return Ok(()); // skip fast-path; per-file fingerprints decide re-extract
    }
    let text_changed = store.get_meta("text_pipeline")?.as_deref()
        != Some(fingerprint::text_pipeline_json(cfg).as_str());
    let image_changed = store.get_meta("image_pipeline")?.as_deref()
        != Some(fingerprint::image_pipeline_json(cfg, images_supported).as_str());

    if text_changed == image_changed {
        // Both changed, or only the embed recipe / prefix version changed:
        // full reindex from scratch.
        let conn = store.conn();
        conn.execute("DELETE FROM chunks_vec", [])?;
        conn.execute("DELETE FROM chunks_fts", [])?;
        conn.execute("DELETE FROM chunks", [])?;
        conn.execute("DELETE FROM files", [])?;
    } else if text_changed {
        conn_delete_kind(store, "kind = 'text'");
    } else {
        conn_delete_kind(store, "kind IN ('image', 'pdf_page')");
    }
    Ok(())
}

fn conn_delete_kind(store: &Store, kind_filter: &str) {
    let conn = store.conn();
    let _ = conn.execute(
        &format!("DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks WHERE {kind_filter})"),
        [],
    );
    let _ = conn.execute(
        &format!("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE {kind_filter})"),
        [],
    );
    let _ = conn.execute(&format!("DELETE FROM chunks WHERE {kind_filter}"), []);
}

/// Index one walked file. Text files extract/embed/commit in one
/// transaction; image and PDF hit the Task 14/15 stubs, whose errors are
/// recorded per file (never fatal to the walk). Embedder failures DO
/// propagate.
#[allow(clippy::too_many_arguments)]
fn index_file(
    store: &Store,
    embedder: &dyn Embedder,
    cfg: &Config,
    config_fp: &str,
    images_supported: bool,
    cache_dir: &Path,
    mut sink: EventSink<'_>,
    path: PathBuf,
    collection: String,
    kind: FileKind,
    ext: String,
    summary: &mut IndexSummary,
) -> Result<()> {
    let path_str = path.to_string_lossy().into_owned();
    let metadata = match std::fs::metadata(&path) {
        Ok(md) => md,
        Err(e) => {
            // Unreadable file: record it like any extract error.
            record_extract_error(
                store,
                &mut sink,
                &path,
                &collection,
                kind,
                &ext,
                config_fp,
                Error::fatal(e.to_string()),
                summary,
                false,
            );
            return Ok(());
        }
    };
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let size = metadata.len();
    let file_fp = fingerprint::file_fingerprint(&path_str, mtime, size, config_fp);

    // Capability skips: the walk does not know the embedder, so an image
    // file no backend can embed (and HEIC with no converter) is skipped
    // here with its NDJSON reason. Unlike walk-skips, these persist a
    // complete row carrying `skip_reason`: a later capability flip must
    // find and re-evaluate the file (spec §9: skip, not error).
    if kind == FileKind::Image && !images_supported {
        record_capability_skip(
            store,
            &mut sink,
            &path,
            &path_str,
            &collection,
            kind,
            &ext,
            size,
            mtime,
            &file_fp,
            SkipReason::NoImageBackend,
            summary,
        )?;
        return Ok(());
    }
    if matches!(ext.to_ascii_lowercase().as_str(), "heic" | "heif")
        && matches!(cfg.files.heic, HeicMode::Magick)
        && extract::image::find_magick().is_none()
    {
        record_capability_skip(
            store,
            &mut sink,
            &path,
            &path_str,
            &collection,
            kind,
            &ext,
            size,
            mtime,
            &file_fp,
            SkipReason::NoMagick,
            summary,
        )?;
        return Ok(());
    }

    // Skip fast-path: unchanged fingerprint, complete, and no recorded
    // skip. Skip rows always re-process: the capability that skipped them
    // may have flipped without moving the fingerprint (e.g. a converter
    // appearing on PATH).
    if let Some(row) = store.get_file_by_path(&path_str)? {
        if row.fingerprint == file_fp && row.complete && row.skip_reason.is_none() {
            summary.skipped += 1;
            emit_ok(sink.json, &path, &mut *sink.out)?;
            return Ok(());
        }
    }

    // Re-extract replaces any previous rows for this path.
    store.delete_file_by_path(&path_str)?;

    // --- extract phase (per-file errors recorded, walk continues) --------
    let chunks: Vec<Chunk> = match kind {
        FileKind::Text => {
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    record_extract_error(
                        store,
                        &mut sink,
                        &path,
                        &collection,
                        kind,
                        &ext,
                        config_fp,
                        Error::fatal(e.to_string()),
                        summary,
                        false,
                    );
                    return Ok(());
                }
            };
            match extract::extract_text(&path, &bytes, &cfg.chunk.text) {
                Ok(chunks) => chunks,
                Err(e) => {
                    record_extract_error(
                        store, &mut sink, &path, &collection, kind, &ext, config_fp, e, summary, false,
                    );
                    return Ok(());
                }
            }
        }
        FileKind::Image => match extract::image::extract_image(
            &path,
            &cfg.chunk.image,
            cache_dir,
            cfg.files.heic,
        ) {
            Ok(chunks) => chunks,
            Err(e) => {
                record_extract_error(
                    store, &mut sink, &path, &collection, kind, &ext, config_fp, e, summary, false,
                );
                return Ok(());
            }
        },
        FileKind::Pdf => match extract::pdf::extract_pdf(
            &path,
            &cfg.chunk.text,
            &cfg.chunk.pdf,
            &cfg.chunk.image,
            cache_dir,
        ) {
            Ok(chunks) => chunks,
            Err(e) => {
                record_extract_error(
                    store, &mut sink, &path, &collection, kind, &ext, config_fp, e, summary, false,
                );
                return Ok(());
            }
        },
    };

    // --- embed phase (embedder failures are FATAL, not per-file) ---------
    let title = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut inputs = Vec::new();
    for c in &chunks {
        if !matches!(c.embed, EmbedDecision::Vec) {
            continue;
        }
        // Image chunks carry their PNG bytes; everything else embeds as
        // prefixed text (Task 15 adds the PdfPage arm).
        if c.kind == crate::types::ChunkKind::Image {
            match c.image_bytes.clone() {
                Some(bytes) => inputs.push(EmbedInput::Image {
                    bytes,
                    mime: "image/png".to_string(),
                }),
                None => {
                    return Err(Error::fatal(format!(
                        "image chunk for {} has no PNG bytes",
                        path.display()
                    )));
                }
            }
        } else {
            inputs.push(EmbedInput::Document {
                title: Some(title.clone()),
                text: c.text.clone().unwrap_or_default(),
            });
        }
    }
    // --- embed phase ----------------------------------------------------
    // A file-scoped embed failure (e.g. the server 4xx'd one chunk) is a
    // per-file error row, not a fatal abort; transport/5xx still propagate.
    let vectors = match embedder.embed_batch(&inputs) {
        Ok(vectors) => vectors,
        Err(e) if e.is_file_error() => {
            record_extract_error(
                store, &mut sink, &path, &collection, kind, &ext, config_fp, e, summary, false,
            );
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    // --- write phase: one transaction ------------------------------------
    // The collection is vec metadata, so it is kept out of `NewFile`'s move.
    let collection_for_vec = collection.clone();
    let tx = store.begin()?;
    let file_id = store.upsert_file(&NewFile {
        path: path_str.clone(),
        collection,
        kind,
        ext: ext.clone(),
        size,
        mtime,
        fingerprint: file_fp,
        complete: false,
        next_page: None,
        skip_reason: None,
        indexed_at: now_unix(),
        error: None,
    })?;
    let mut vec_iter = vectors.into_iter();
    for (ordinal, c) in chunks.iter().enumerate() {
        let chunk_id = store.insert_chunk(
            &tx,
            NewChunk {
                file_id,
                ordinal: ordinal as i64,
                page: c.page,
                kind: c.kind,
                text: c.text.clone(),
                start_byte: c.start_byte,
                end_byte: c.end_byte,
            },
        )?;
        if matches!(c.embed, EmbedDecision::Vec) {
            let v = vec_iter
                .next()
                .ok_or_else(|| Error::fatal("embedder returned fewer vectors than inputs"))?;
            store.insert_vec(&tx, chunk_id, &v, kind, &ext, mtime, &collection_for_vec)?;
        }
    }
    tx.commit()?;
    store.mark_complete(file_id)?;

    summary.embedded += 1;
    emit_ok(sink.json, &path, &mut *sink.out)?;
    Ok(())
}

/// Write the per-file error row and bump the error counter. Not fatal to
/// the walk; `--strict` collects these after the walk.
#[allow(clippy::too_many_arguments)]
fn record_extract_error(
    store: &Store,
    sink: &mut EventSink<'_>,
    path: &Path,
    collection: &str,
    kind: FileKind,
    ext: &str,
    config_fp: &str,
    err: Error,
    summary: &mut IndexSummary,
    complete: bool,
) {
    let _ = complete;
    summary.errors += 1;
    let message = err.to_string();
    let path_str = path.to_string_lossy().into_owned();
    let (mtime, size) = std::fs::metadata(path)
        .map(|m| {
            (
                m.modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0),
                m.len(),
            )
        })
        .unwrap_or((0, 0));
    if sink.json {
        let _ = writeln!(
            sink.out,
            "{}",
            serde_json::json!({
                "event": "file",
                "path": path_str,
                "status": "error",
                "reason": serde_json::Value::Null,
                "error": message,
            })
        );
    }
    let row = NewFile {
        path: path_str.clone(),
        collection: collection.to_string(),
        kind,
        ext: ext.to_string(),
        size,
        mtime,
        fingerprint: fingerprint::file_fingerprint(&path_str, mtime, size, config_fp),
        complete: false,
        next_page: None,
        skip_reason: None,
        indexed_at: now_unix(),
        error: Some(message),
    };
    let _ = store.upsert_file(&row);
}

/// Persist a capability-skip row (complete, reason recorded, zero chunks,
/// no error) and emit its NDJSON event. Upsert refreshes the fingerprint
/// so a capability flip re-evaluates the file.
#[allow(clippy::too_many_arguments)]
fn record_capability_skip(
    store: &Store,
    sink: &mut EventSink<'_>,
    path: &Path,
    path_str: &str,
    collection: &str,
    kind: FileKind,
    ext: &str,
    size: u64,
    mtime: i64,
    file_fp: &str,
    reason: SkipReason,
    summary: &mut IndexSummary,
) -> Result<()> {
    store.delete_file_by_path(path_str)?;
    let file_id = store.upsert_file(&NewFile {
        path: path_str.to_string(),
        collection: collection.to_string(),
        kind,
        ext: ext.to_string(),
        size,
        mtime,
        fingerprint: file_fp.to_string(),
        complete: true,
        next_page: None,
        skip_reason: Some(reason.as_str().to_string()),
        indexed_at: now_unix(),
        error: None,
    })?;
    let _ = file_id;
    summary.skipped += 1;
    emit_skip(sink.json, path, reason, &mut *sink.out)?;
    Ok(())
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
