//! Indexing pipeline (spec §8 walk → extract → embed → write, §10 events).
//!
//! `walk` enumerates the corpus and classifies paths; `run` drives the
//! pipeline: fingerprint-gated re-extraction, per-file transactions, the
//! per-file extract-error path (which never aborts the walk), and the
//! stderr summary / NDJSON events.

pub mod skip;
pub mod walk;

pub use skip::SkipReason;
pub use walk::{WalkEvent, kind_for_ext, walk};

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::Config;
use crate::embedder::{EmbedInput, Embedder};
use crate::error::{Error, Result};
use crate::extract::{self, Chunk, EmbedDecision};
use crate::fingerprint;
use crate::paths::ferret_dirs;
use crate::store::{self, NewChunk, NewFile, Store};
use crate::types::FileKind;

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
#[allow(clippy::too_many_arguments)]
pub fn run(
    cfg: &Config,
    embedder: &dyn Embedder,
    extra_paths: Vec<PathBuf>,
    json_ndjson: bool,
    strict: bool,
    only: bool,
    out: &mut dyn std::io::Write,
    err: &mut dyn std::io::Write,
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
    apply_pipeline_changes(&store, cfg, images_supported, &config_fp, err)?;

    let dirs = ferret_dirs(&|k| std::env::var(k).ok());
    let mut summary = IndexSummary::default();

    // `--only`: skip the roots walk entirely. `roots` are still passed to
    // `walk` so extra paths under a configured root inherit its collection.
    let events = walk::walk(
        &cfg.index.roots,
        &extra_paths,
        &cfg.files,
        cfg.files.follow_symlinks,
        only,
    )?;
    summary.seen = events.len() as u64;

    // NDJSON `plan` event (spec §10): progress totals before any per-file
    // event. `files` counts File events only (walk-skips excluded); `bytes`
    // sums each event's walk-time size (0 when it could not be statted).
    if json_ndjson {
        let mut plan_files = 0u64;
        let mut plan_bytes = 0u64;
        for event in &events {
            if let WalkEvent::File { size, .. } = event {
                plan_files += 1;
                plan_bytes += size;
            }
        }
        writeln!(
            out,
            "{}",
            serde_json::json!({
                "event": "plan",
                "files": plan_files,
                "bytes": plan_bytes,
            })
        )?;
    }

    // PDFs defer to the two-pass scheduler below (spec §4.4): every PDF
    // first advances through `first_pass_pages`, then the remainder.
    // Their NDJSON events therefore come after text/image events.
    let mut pdf_jobs: Vec<PdfJob> = Vec::new();
    for event in events {
        match event {
            WalkEvent::File {
                path,
                collection,
                kind: FileKind::Pdf,
                ext,
                ..
            } => pdf_jobs.push(PdfJob {
                path,
                collection,
                ext,
            }),
            WalkEvent::File {
                path,
                collection,
                kind,
                ext,
                ..
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
    index_pdfs_in_passes(
        &store,
        embedder,
        cfg,
        &config_fp,
        images_supported,
        &dirs.cache_dir,
        json_ndjson,
        out,
        err,
        pdf_jobs,
        &mut summary,
    )?;

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
    // NDJSON `summary` event (spec §10): the last stdout line, with the
    // exact counters the stderr human summary prints from this struct.
    // (Emitted after the --strict gate: on that path neither appears.)
    if json_ndjson {
        writeln!(
            out,
            "{}",
            serde_json::json!({
                "event": "summary",
                "seen": summary.seen,
                "skipped": summary.skipped,
                "embedded": summary.embedded,
                "errors": summary.errors,
            })
        )?;
    }
    Ok(summary)
}

/// A walked PDF awaiting the two-pass scheduler. Metadata and the file
/// row are resolved inside `index_pdf_file` (it needs the row id for
/// raster paths); the walk only classifies.
#[derive(Clone)]
struct PdfJob {
    path: PathBuf,
    collection: String,
    ext: String,
}

/// Two-pass PDF scheduling (spec §4.4): pass 1 brings every PDF from its
/// cursor through `min(first_pass_pages, max_pages, n)`; pass 2 continues
/// through `min(max_pages, n)`. A file that completes or fails in pass 1
/// is not revisited in pass 2 (a failed page must never be jumped over).
#[allow(clippy::too_many_arguments)]
fn index_pdfs_in_passes(
    store: &Store,
    embedder: &dyn Embedder,
    cfg: &Config,
    config_fp: &str,
    images_supported: bool,
    cache_dir: &Path,
    json_ndjson: bool,
    out: &mut dyn std::io::Write,
    err: &mut dyn std::io::Write,
    jobs: Vec<PdfJob>,
    summary: &mut IndexSummary,
) -> Result<()> {
    // Pass 1: every job advances up to first_pass_pages (capped by
    // max_pages inside); unfinished jobs continue in pass 2.
    let mut pending: Vec<PdfJob> = Vec::new();
    for job in &jobs {
        let finished = index_pdf_file(
            store,
            embedder,
            cfg,
            config_fp,
            images_supported,
            cache_dir,
            EventSink {
                json: json_ndjson,
                out: &mut *out,
            },
            err,
            job,
            cfg.chunk.pdf.first_pass_pages.min(cfg.chunk.pdf.max_pages),
            summary,
        )?;
        if !finished {
            pending.push(job.clone());
        }
    }
    // Pass 2: remainder up to max_pages.
    for job in &pending {
        index_pdf_file(
            store,
            embedder,
            cfg,
            config_fp,
            images_supported,
            cache_dir,
            EventSink {
                json: json_ndjson,
                out: &mut *out,
            },
            err,
            job,
            cfg.chunk.pdf.max_pages,
            summary,
        )?;
    }
    Ok(())
}

/// Resolved poppler binaries for one PDF file's planned work.
struct PdfBins {
    pdfinfo: PathBuf,
    pdftotext: PathBuf,
    pdftoppm: Option<PathBuf>,
}

/// Resolve the binaries a PDF needs: pdfinfo + pdftotext always (page
/// text is always stored); pdftoppm only when page images may embed.
fn resolve_pdf_bins(
    extract_cfg: &crate::config::ExtractConfig,
    need_render: bool,
) -> Result<PdfBins> {
    let pdfinfo = crate::extract::pdf::resolve_tool(extract_cfg, "pdfinfo")?;
    let pdftotext = crate::extract::pdf::resolve_tool(extract_cfg, "pdftotext")?;
    let pdftoppm = if need_render {
        Some(crate::extract::pdf::resolve_tool(extract_cfg, "pdftoppm")?)
    } else {
        None
    };
    Ok(PdfBins {
        pdfinfo,
        pdftotext,
        pdftoppm,
    })
}

/// One page's embed input: image+text, image-only (no prefix), text-only,
/// or skip (no chunk, cursor still advances).
enum PageInput {
    Pdf { png: Vec<u8>, text: Option<String> },
    Img(Vec<u8>),
    Doc(String),
    Skip,
}

/// Best-effort raster cleanup (spec §4.4: cache rasters never linger).
fn remove_raster(png_path: &Option<PathBuf>) {
    if let Some(p) = png_path {
        let _ = std::fs::remove_file(p);
    }
}

/// One page's outcome: embedded, deliberately skipped (no chunk, cursor
/// advances), or file-failing (record the file error, stop this file).
enum PageOutcome {
    Done,
    Skipped,
    Failed(String),
}

/// Index one PDF through this pass's page ceiling. Returns true when the
/// file needs no more work this run (complete, fast-path-skipped, or
/// failed — a failed page must never be jumped over, so failure ends
/// this file's run).
///
/// One SQLite transaction per page: extract page, embed, insert chunk +
/// vec, bump `next_page`, commit. An abort (fatal embedder/DB error)
/// rolls back only the open page; committed pages stay with `complete=0`.
#[allow(clippy::too_many_arguments)]
fn index_pdf_file(
    store: &Store,
    embedder: &dyn Embedder,
    cfg: &Config,
    config_fp: &str,
    images_supported: bool,
    cache_dir: &Path,
    mut sink: EventSink<'_>,
    err: &mut dyn std::io::Write,
    job: &PdfJob,
    pass_limit: u32,
    summary: &mut IndexSummary,
) -> Result<bool> {
    let path = &job.path;
    let path_str = path.to_string_lossy().into_owned();
    let metadata = match std::fs::metadata(path) {
        Ok(md) => md,
        Err(e) => {
            record_pdf_error(
                store,
                &mut sink,
                path,
                &path_str,
                &job.collection,
                &job.ext,
                config_fp,
                1,
                &Error::fatal(e.to_string()),
                summary,
            )?;
            return Ok(true);
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

    // Fast-path: unchanged fingerprint and complete (PDF rows never carry
    // skip_reason, so no reason guard is needed).
    if let Some(row) = store.get_file_by_path(&path_str)? {
        if row.fingerprint == file_fp && row.complete {
            summary.skipped += 1;
            emit_ok(sink.json, path, &mut *sink.out)?;
            return Ok(true);
        }
    }

    // Poppler: pdfinfo + pdftotext always (page text is always stored);
    // pdftoppm only when page images may embed.
    let need_render = cfg.chunk.pdf.embed_page_image && images_supported;
    let bins = match resolve_pdf_bins(&cfg.extract, need_render) {
        Ok(b) => b,
        Err(e) => {
            record_pdf_error(
                store,
                &mut sink,
                path,
                &path_str,
                &job.collection,
                &job.ext,
                config_fp,
                1,
                &e,
                summary,
            )?;
            return Ok(true);
        }
    };
    let n = match crate::extract::pdf::page_count(path, &bins.pdfinfo) {
        Ok(n) => n,
        Err(e) => {
            record_pdf_error(
                store,
                &mut sink,
                path,
                &path_str,
                &job.collection,
                &job.ext,
                config_fp,
                1,
                &e,
                summary,
            )?;
            return Ok(true);
        }
    };

    // Row: reuse on fingerprint match (resume), else wipe and restart at 1.
    let (file_id, was_complete, mut next_page) = match store.get_file_by_path(&path_str)? {
        Some(row) if row.fingerprint == file_fp => {
            (row.id, row.complete, row.next_page.unwrap_or(1))
        }
        _ => {
            store.delete_file_by_path(&path_str)?;
            let id = store.upsert_file(&NewFile {
                path: path_str.clone(),
                collection: job.collection.clone(),
                kind: FileKind::Pdf,
                ext: job.ext.clone(),
                size,
                mtime,
                fingerprint: file_fp,
                complete: false,
                next_page: Some(1),
                skip_reason: None,
                indexed_at: now_unix(),
                error: None,
            })?;
            (id, false, 1)
        }
    };
    if next_page < 1 {
        next_page = 1;
    }

    // Backfill: a page below the cursor with no chunk is attempted first.
    let existing = store.chunk_pages(file_id)?;
    let mut start = next_page;
    for p in 1..next_page {
        if !existing.contains(&p) {
            start = p;
            break;
        }
    }

    let limit = (pass_limit as i64)
        .min(cfg.chunk.pdf.max_pages as i64)
        .min(n as i64);
    let title = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut failed: Option<String> = None;
    let mut current = start;
    while current <= limit {
        if !page_needs_work(store, file_id, current)? {
            // Already embedded (e.g. untouched side of a pipeline bump):
            // advance the cursor without re-embedding.
            let tx = store.begin()?;
            store.set_pdf_progress(&tx, file_id, current + 1, false, None)?;
            tx.commit()?;
            current += 1;
            continue;
        }
        match process_pdf_page(
            store,
            embedder,
            cfg,
            images_supported,
            cache_dir,
            &bins,
            file_id,
            path,
            &title,
            &job.collection,
            &job.ext,
            mtime,
            current as u32,
        )? {
            PageOutcome::Done | PageOutcome::Skipped => current += 1,
            PageOutcome::Failed(message) => {
                failed = Some(message);
                break;
            }
        }
    }
    next_page = current;

    // Close out: complete iff no error and the cursor passed the ceiling.
    // A pass that ends mid-file (pass 1, or a capped max_pages) leaves
    // complete=0 with the error cleared: reaching here means every
    // attempted page committed, so a stale failure must not linger.
    let ceiling = (cfg.chunk.pdf.max_pages as i64).min(n as i64);
    if let Some(message) = failed {
        record_pdf_error(
            store,
            &mut sink,
            path,
            &path_str,
            &job.collection,
            &job.ext,
            config_fp,
            next_page,
            &Error::fatal(message),
            summary,
        )?;
        return Ok(true);
    }
    let complete = next_page > ceiling;
    let tx = store.begin()?;
    store.set_pdf_progress(&tx, file_id, next_page, complete, None)?;
    tx.commit()?;
    if complete {
        summary.embedded += 1;
        emit_ok(sink.json, path, &mut *sink.out)?;
        if !was_complete && (n as i64) > cfg.chunk.pdf.max_pages as i64 {
            let _ = writeln!(
                err,
                "warning: {} has {} pages, indexed first {} (chunk.pdf.max_pages)",
                path.display(),
                n,
                cfg.chunk.pdf.max_pages,
            );
        }
    }
    Ok(complete)
}

/// True when the page has no chunk, or a chunk with no vector (an
/// image-pipeline bump deletes vecs only). Pages with vectors are done.
/// Point queries per page (indexed); the full page list is only scanned
/// once per file for the backfill check.
fn page_needs_work(store: &Store, file_id: i64, page: i64) -> Result<bool> {
    if store.chunk_id_for_page(file_id, page)?.is_none() {
        return Ok(true);
    }
    Ok(!store.page_has_vec(file_id, page)?)
}

/// Extract, embed, and commit one page in a single transaction. The raster
/// is deleted after the commit (best-effort on failure too). Fatal
/// embedder/DB errors propagate (the open transaction rolls back).
#[allow(clippy::too_many_arguments)]
fn process_pdf_page(
    store: &Store,
    embedder: &dyn Embedder,
    cfg: &Config,
    images_supported: bool,
    cache_dir: &Path,
    bins: &PdfBins,
    file_id: i64,
    path: &Path,
    title: &str,
    collection: &str,
    ext: &str,
    mtime: i64,
    page: u32,
) -> Result<PageOutcome> {
    // Page text is always extracted (stored for FTS when non-whitespace).
    let text = match crate::extract::pdf::page_text(path, page, &bins.pdftotext) {
        Ok(t) => t,
        Err(e) => return Ok(PageOutcome::Failed(e.to_string())),
    };
    let stored_text = if text.trim().is_empty() {
        None
    } else {
        Some(text.trim().to_string())
    };

    // Render only when the page image may embed.
    let render = cfg.chunk.pdf.embed_page_image && images_supported;
    let png_path = if render {
        match bins.pdftoppm.as_ref() {
            Some(bin) => match crate::extract::pdf::render_page(
                path,
                page,
                file_id,
                cfg.chunk.pdf.render_dpi,
                bin,
                cache_dir,
            ) {
                Ok(p) => Some(p),
                Err(e) => return Ok(PageOutcome::Failed(e.to_string())),
            },
            None => {
                return Ok(PageOutcome::Failed(format!(
                    "page {page} needs pdftoppm but it was not resolved"
                )));
            }
        }
    } else {
        None
    };

    // Embed decision (spec §4.4): image+text, image-only, text-only,
    // or skip. `embed_text = false` with images sends the page as a
    // bare image (no document prefix); the stored FTS text is unaffected.
    let png_bytes = match &png_path {
        Some(p) => match std::fs::read(p) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                remove_raster(&png_path);
                return Ok(PageOutcome::Failed(e.to_string()));
            }
        },
        None => None,
    };
    let input = match (png_bytes, cfg.chunk.pdf.embed_text, stored_text.clone()) {
        (Some(bytes), false, _) => PageInput::Img(bytes),
        (Some(bytes), true, text) => PageInput::Pdf { png: bytes, text },
        (None, true, Some(t)) => PageInput::Doc(t),
        (None, _, _) => PageInput::Skip,
    };

    let embedded: Option<Vec<f32>> = match input {
        PageInput::Pdf { png, text } => {
            let input = EmbedInput::PdfPage {
                image_png: png,
                text,
            };
            match embedder.embed(input) {
                Ok(v) => Some(v),
                Err(e) if e.is_file_error() => {
                    remove_raster(&png_path);
                    return Ok(PageOutcome::Failed(e.to_string()));
                }
                Err(e) => {
                    remove_raster(&png_path);
                    return Err(e);
                }
            }
        }
        PageInput::Img(bytes) => {
            let input = EmbedInput::Image {
                bytes,
                mime: "image/png".to_string(),
            };
            match embedder.embed(input) {
                Ok(v) => Some(v),
                Err(e) if e.is_file_error() => {
                    remove_raster(&png_path);
                    return Ok(PageOutcome::Failed(e.to_string()));
                }
                Err(e) => {
                    remove_raster(&png_path);
                    return Err(e);
                }
            }
        }
        PageInput::Doc(t) => {
            let input = EmbedInput::Document {
                title: Some(title.to_string()),
                text: t,
            };
            match embedder.embed(input) {
                Ok(v) => Some(v),
                Err(e) if e.is_file_error() => {
                    return Ok(PageOutcome::Failed(e.to_string()));
                }
                Err(e) => return Err(e),
            }
        }
        PageInput::Skip => None,
    };

    // One transaction: replace the page's chunks, store the vector, bump
    // the cursor. Skipped pages keep no chunk (cursor still advances).
    let tx = store.begin()?;
    store.delete_page_chunks(&tx, file_id, page as i64)?;
    let done = embedded.is_some();
    if let Some(vector) = embedded {
        let chunk_id = store.insert_chunk(
            &tx,
            NewChunk {
                file_id,
                ordinal: page as i64 - 1,
                page: Some(page as i64),
                kind: crate::types::ChunkKind::PdfPage,
                text: stored_text,
                start_byte: None,
                end_byte: None,
            },
        )?;
        store.insert_vec(
            &tx,
            chunk_id,
            &vector,
            FileKind::Pdf,
            ext,
            mtime,
            collection,
        )?;
    }
    store.set_pdf_progress(&tx, file_id, page as i64 + 1, false, None)?;
    tx.commit()?;
    // The raster lives until the page transaction commits (spec §4.4);
    // failure exits above delete it on the way out, so a retry re-renders.
    remove_raster(&png_path);
    Ok(if done {
        PageOutcome::Done
    } else {
        PageOutcome::Skipped
    })
}

/// PDF file-error path: upsert the row with `complete=0`, the sticky
/// cursor, and the message; emit the NDJSON error event; bump errors.
/// Unlike `record_extract_error` this preserves `next_page` (resume) AND
/// keeps already-committed page chunks (spec §9 partial-file): only the
/// files row is upserted, never wiped.
#[allow(clippy::too_many_arguments)]
fn record_pdf_error(
    store: &Store,
    sink: &mut EventSink<'_>,
    path: &Path,
    path_str: &str,
    collection: &str,
    ext: &str,
    config_fp: &str,
    next_page: i64,
    err: &Error,
    summary: &mut IndexSummary,
) -> Result<()> {
    let message = err.to_string();
    let (size, mtime) = std::fs::metadata(path)
        .map(|md| {
            let mtime = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            (md.len(), mtime)
        })
        .unwrap_or((0, 0));
    let file_fp = fingerprint::file_fingerprint(path_str, mtime, size, config_fp);
    // No delete: committed pages stay (spec §9); the upsert below only
    // touches the files row (cursor, error, fingerprint).
    store.upsert_file(&NewFile {
        path: path_str.to_string(),
        collection: collection.to_string(),
        kind: FileKind::Pdf,
        ext: ext.to_string(),
        size,
        mtime,
        fingerprint: file_fp,
        complete: false,
        next_page: Some(next_page),
        skip_reason: None,
        indexed_at: now_unix(),
        error: Some(message.clone()),
    })?;
    summary.errors += 1;
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
    Ok(())
}
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
    err: &mut dyn std::io::Write,
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
        conn_delete_kind(store, "kind = 'text'")?;
        // `embed_text` flips change page vectors (B.3 with text vs B.2
        // image-only): re-embed PDF pages like an image bump. Other text
        // bumps keep pdf vecs and only refresh stored FTS text.
        if pdf_embed_text_changed(store, cfg) {
            reset_pdf_vecs(store)?;
        } else {
            refresh_pdf_texts(store, cfg, err);
        }
    } else {
        // Image-only change: image files re-extract from scratch, but PDF
        // page chunks (and their text) stay — only their vectors are
        // re-embedded from `next_page = 1`.
        conn_delete_kind(store, "kind = 'image'")?;
        reset_pdf_vecs(store)?;
    }
    Ok(())
}

/// True when the stored text pipeline's `pdf.embed_text` differs from the
/// merged config's. A flip changes page vectors (text in/out of the
/// embedding), so PDF pages must re-embed despite the text-only bump.
fn pdf_embed_text_changed(store: &Store, cfg: &Config) -> bool {
    store
        .get_meta("text_pipeline")
        .ok()
        .flatten()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.pointer("/pdf/embed_text")?.as_bool())
        .is_some_and(|old| old != cfg.chunk.pdf.embed_text)
}
/// Re-extract page text for every indexed PDF (text-pipeline bump keeps
/// vectors; FTS must follow the new text). Best-effort per page: poppler
/// failures warn once and leave that page's text stale rather than
/// failing the run. The `chunks_fts_au` trigger mirrors each rewrite.
fn refresh_pdf_texts(store: &Store, cfg: &Config, err: &mut dyn std::io::Write) {
    let Ok(pdftotext) = crate::extract::pdf::resolve_tool(&cfg.extract, "pdftotext") else {
        let _ = writeln!(
            err,
            "warning: skipping PDF text refresh (pdftotext not found)"
        );
        return;
    };
    let files: Vec<(i64, String)> = match (|| -> Result<Vec<(i64, String)>> {
        let mut stmt = store
            .conn()
            .prepare("SELECT id, path FROM files WHERE kind = 'pdf'")?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<std::result::Result<Vec<(i64, String)>, _>>()?;
        Ok(rows)
    })() {
        Ok(rows) => rows,
        Err(_) => return,
    };
    for (file_id, path) in files {
        let Ok(pages) = store.chunk_pages(file_id) else {
            continue;
        };
        for page in pages {
            let Ok(text) =
                crate::extract::pdf::page_text(Path::new(&path), page as u32, &pdftotext)
            else {
                let _ = writeln!(
                    err,
                    "warning: PDF text refresh failed for {} page {page} (keeping stale text)",
                    Path::new(&path).display(),
                );
                continue;
            };
            let trimmed = text.trim();
            if let Err(e) = store.update_chunk_text(file_id, page, trimmed) {
                let _ = writeln!(
                    err,
                    "warning: PDF text refresh failed for {} page {page} (keeping stale text: {e})",
                    Path::new(&path).display(),
                );
                break;
            }
        }
    }
}

/// Image-pipeline bump for PDFs: drop every `pdf_page` vector and rewind
/// each file's cursor to 1. Chunks (and their text) stay; the next run
/// re-embeds pages whose vectors are missing (`page_needs_work`).
fn reset_pdf_vecs(store: &Store) -> Result<()> {
    let mut stmt = store
        .conn()
        .prepare("SELECT id FROM files WHERE kind = 'pdf'")?;
    let ids: Vec<i64> = stmt
        .query_map([], |r| r.get(0))?
        .collect::<std::result::Result<_, _>>()?;
    for id in ids {
        store.delete_pdf_vecs(id)?;
        let tx = store.begin()?;
        store.set_pdf_progress(&tx, id, 1, false, None)?;
        tx.commit()?;
    }
    Ok(())
}

fn conn_delete_kind(store: &Store, kind_filter: &str) -> Result<()> {
    let conn = store.conn();
    conn.execute(
        &format!(
            "DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks WHERE {kind_filter})"
        ),
        [],
    )?;
    conn.execute(
        &format!(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE {kind_filter})"
        ),
        [],
    )?;
    conn.execute(&format!("DELETE FROM chunks WHERE {kind_filter}"), [])?;
    Ok(())
}

/// Index one walked file. Text and image files extract/embed/commit in one
/// transaction (PDF per-page transactions land in Task 15); extract
/// failures are recorded per file (never fatal to the walk). Embedder
/// transport failures DO propagate.
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
                Error::file(e.to_string()),
                summary,
            )?;
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
    // file no backend can embed is skipped here with its NDJSON reason.
    // Unlike walk-skips, these persist a complete row carrying
    // `skip_reason`: a later probe flip must find and re-evaluate the
    // file (spec §9: skip, not error). (HEIC converter absence is a
    // walk-skip decided in `walk`, not here.)
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

    // Skip fast-path: unchanged fingerprint, complete, and no recorded
    // skip. no_image_backend rows always re-process: the probe result may
    // have flipped without moving the fingerprint.
    if let Some(row) = store.get_file_by_path(&path_str)? {
        if row.fingerprint == file_fp && row.complete && row.skip_reason.is_none() {
            summary.skipped += 1;
            emit_ok(sink.json, &path, &mut *sink.out)?;
            return Ok(());
        }
    }

    // No DB mutation before extract/embed succeed: the write tx below
    // deletes + replaces atomically, so a fatal abort keeps prior rows.

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
                        Error::file(e.to_string()),
                        summary,
                    )?;
                    return Ok(());
                }
            };
            match extract::extract_text(&path, &bytes, &cfg.chunk.text) {
                Ok(chunks) => chunks,
                Err(e) => {
                    record_extract_error(
                        store,
                        &mut sink,
                        &path,
                        &collection,
                        kind,
                        &ext,
                        config_fp,
                        e,
                        summary,
                    )?;
                    return Ok(());
                }
            }
        }
        FileKind::Image => {
            match extract::image::extract_image(&path, &cfg.chunk.image, cache_dir, cfg.files.heic)
            {
                Ok(chunks) => chunks,
                Err(e) => {
                    record_extract_error(
                        store,
                        &mut sink,
                        &path,
                        &collection,
                        kind,
                        &ext,
                        config_fp,
                        e,
                        summary,
                    )?;
                    return Ok(());
                }
            }
        }
        FileKind::Pdf => {
            // Unreachable via run(): PDFs route through index_pdf_file
            // (two-pass scheduler). Defensive: never silently embed a PDF
            // as flat text.
            return Err(Error::fatal(
                "internal error: PDF reached text indexing (report a bug)",
            ));
        }
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
            match &c.image_bytes {
                Some(bytes) => inputs.push(EmbedInput::Image {
                    bytes: bytes.clone(),
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
                store,
                &mut sink,
                &path,
                &collection,
                kind,
                &ext,
                config_fp,
                e,
                summary,
            )?;
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    // --- write phase: one transaction ------------------------------------
    // The collection is vec metadata, so it is kept out of `NewFile`'s move.
    // Prior rows are dropped here (after extract/embed succeed), so a fatal
    // abort during extract/embed keeps the previous index rows.
    let collection_for_vec = collection.clone();
    store.delete_file_by_path(&path_str)?;
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
    store.mark_complete_in_tx(&tx, file_id)?;
    tx.commit()?;

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
) -> Result<()> {
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
    // Replace prior rows with the error row; a DB failure here is fatal
    // (the walk cannot record the outcome), unlike the extract failure.
    store.delete_file_by_path(&path_str)?;
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
    store.upsert_file(&row)?;
    Ok(())
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
    store.upsert_file(&NewFile {
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
