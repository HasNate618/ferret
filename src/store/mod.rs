//! SQLite store: schema creation, index-meta validation, and CRUD.
//!
//! sqlite-vec is registered in-process before any connection opens (no
//! runtime `.so` load); registration failure is fatal. Opening an index
//! whose `model_id`, `dimensions`, or `prefix_profile` differ from the
//! merged config is fatal — the user must pass a new `--index` or delete the
//! index file. A `backend`-only difference is a stderr warning, not an
//! error. There is deliberately no method that rewrites meta (`ferret rm`
//! must not change it).

mod schema;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::fingerprint;
use crate::types::{ChunkKind, FileKind};
use rusqlite::functions::FunctionFlags;
use rusqlite::{Connection, Transaction, params};
use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::sync::OnceLock;
use std::time::Duration;

/// An open ferret index. `dimensions` is the vec0 embedding width the index
/// was created with (from the merged config at open time).
pub struct Store {
    conn: Connection,
    pub dimensions: u32,
}

impl fmt::Debug for Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Store")
            .field("dimensions", &self.dimensions)
            .finish_non_exhaustive()
    }
}

/// New `files` row for `Store::upsert_file`.
#[derive(Debug, Clone)]
pub struct NewFile {
    pub path: String,
    pub collection: String,
    pub kind: FileKind,
    pub ext: String,
    pub size: u64,
    pub mtime: i64,
    pub fingerprint: String,
    pub complete: bool,
    pub next_page: Option<i64>,
    pub skip_reason: Option<String>,
    pub indexed_at: i64,
    pub error: Option<String>,
}

/// A `files` row as stored.
#[derive(Debug, Clone)]
pub struct FileRow {
    pub id: i64,
    pub path: String,
    pub collection: String,
    pub kind: FileKind,
    pub ext: String,
    pub size: u64,
    pub mtime: i64,
    pub fingerprint: String,
    pub complete: bool,
    pub next_page: Option<i64>,
    pub skip_reason: Option<String>,
    pub indexed_at: i64,
    pub error: Option<String>,
}

/// New `chunks` row for `Store::insert_chunk`. Empty text is stored as SQL
/// NULL.
#[derive(Debug, Clone)]
pub struct NewChunk {
    pub file_id: i64,
    pub ordinal: i64,
    pub page: Option<i64>,
    pub kind: ChunkKind,
    pub text: Option<String>,
    pub start_byte: Option<i64>,
    pub end_byte: Option<i64>,
    /// Audio segment timestamps in milliseconds (`None` otherwise).
    pub t_start_ms: Option<i64>,
    pub t_end_ms: Option<i64>,
}

/// Counts for `ferret status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StatusCounts {
    pub files_text: u64,
    pub files_image: u64,
    pub files_pdf: u64,
    pub chunks: u64,
    pub incomplete: u64,
    pub errors: u64,
    pub indexed_at_max: Option<i64>,
}

impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Error::fatal(e.to_string())
    }
}

/// Cache of the `sqlite3_auto_extension` return code so the C extension is
/// registered exactly once per process, before the first connection opens.
static SQLITE_VEC_REGISTRATION: OnceLock<i32> = OnceLock::new();

fn register_sqlite_vec() -> Result<()> {
    let rc = *SQLITE_VEC_REGISTRATION.get_or_init(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut std::os::raw::c_char,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> std::os::raw::c_int,
        >(
            sqlite_vec::sqlite3_vec_init as *const ()
        )))
    });
    if rc != rusqlite::ffi::SQLITE_OK {
        return Err(Error::fatal(format!(
            "failed to register sqlite-vec in-process (sqlite3_auto_extension rc = {rc}); \
             the sqlite-vec C extension did not build or link"
        )));
    }
    Ok(())
}

/// Open (creating if absent) the index at `path` and validate its meta
/// against the merged config.
pub fn open(path: &Path, cfg: &Config, images_supported: bool) -> Result<Store> {
    register_sqlite_vec()?;
    // `ferret search` / `status` work against an index that `ferret index`
    // has not created yet: create the parent dir (e.g. `{data_dir}/ferret`)
    // so a fresh open yields an empty index (zero hits) instead of a raw
    // SQLite "unable to open database file".
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let new_index = !path.exists();
    let conn = Connection::open(path)?;
    let store = Store {
        conn,
        dimensions: cfg.embedder.dimensions,
    };
    store.apply_pragmas()?;
    store.register_functions()?;
    if new_index {
        store
            .conn
            .execute_batch(&schema::ddl(cfg.embedder.dimensions))?;
        store.write_initial_meta(cfg, images_supported)?;
    } else {
        store.validate_meta(cfg, path)?;
        // Idempotent backfill AFTER validation: only a real ferret index
        // reaches here, so `chunks` exists for the trigger bodies.
        // Fresh databases already got them from ddl().
        store.conn.execute_batch(schema::triggers())?;
        // Audio-milestone backfill: pre-audio indexes lack the segment
        // timestamp columns; ADD COLUMN only when PRAGMA says missing.
        store.backfill_audio_columns()?;
    }
    Ok(store)
}

/// Open an existing index; unlike `open`, never creates parent dirs or a
/// new database. Read-only commands (`status`, `search`, `rm`, `gc`)
/// use this so a typo'd `--index` errors instead of materializing an
/// empty index as a side effect.
pub fn open_existing(path: &Path, cfg: &Config, images_supported: bool) -> Result<Store> {
    if !path.exists() {
        return Err(Error::fatal(format!(
            "index {} does not exist; run `ferret index` first (refusing to create it on a read-only command)",
            path.display()
        )));
    }
    open(path, cfg, images_supported)
}

impl Store {
    /// Read-only connection handle for tests and read-only CLI output; all
    /// writes go through the methods above.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// WAL, a 5s busy timeout, and enforced foreign keys; also refuses
    /// SQLite older than 3.38.
    fn apply_pragmas(&self) -> Result<()> {
        let version: String = self
            .conn
            .query_row("select sqlite_version()", [], |r| r.get(0))?;
        if !sqlite_at_least(&version, 3, 38) {
            return Err(Error::fatal(format!(
                "SQLite {version} is too old; ferret requires >= 3.38.0 (rusqlite bundled)"
            )));
        }
        let mode: String = self
            .conn
            .pragma_update_and_check(None, "journal_mode", "WAL", |r| r.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(Error::fatal(format!(
                "could not enable WAL journal mode (got `{mode}`)"
            )));
        }
        self.conn.busy_timeout(Duration::from_millis(5000))?;
        self.conn.pragma_update(None, "foreign_keys", "ON")?;
        let fk: i64 = self
            .conn
            .pragma_query_value(None, "foreign_keys", |r| r.get(0))?;
        if fk != 1 {
            return Err(Error::fatal(
                "could not enforce foreign_keys pragma (cascades unreliable)".to_string(),
            ));
        }
        Ok(())
    }

    /// Per-connection scalar functions. `ferret_basename(path)` feeds the
    /// `chunks_fts` filename column from the triggers added with indexing.
    fn register_functions(&self) -> Result<()> {
        self.conn.create_scalar_function(
            "ferret_basename",
            1,
            FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
            |ctx| {
                let path: String = ctx.get(0)?;
                Ok(path.rsplit('/').next().unwrap_or("").to_string())
            },
        )?;
        Ok(())
    }

    fn write_initial_meta(&self, cfg: &Config, images_supported: bool) -> Result<()> {
        let entries: [(&str, String); 10] = [
            ("schema_version", "1".to_string()),
            ("model_id", cfg.embedder.model.clone()),
            ("dimensions", cfg.embedder.dimensions.to_string()),
            ("backend", cfg.embedder.backend.to_string()),
            ("prefix_profile", cfg.embedder.prefix_profile.to_string()),
            (
                "images_supported",
                if images_supported { "true" } else { "false" }.to_string(),
            ),
            (
                "config_fingerprint",
                fingerprint::config_fingerprint(cfg, images_supported),
            ),
            ("text_pipeline", fingerprint::text_pipeline_json(cfg)),
            (
                "image_pipeline",
                fingerprint::image_pipeline_json(cfg, images_supported),
            ),
            // No probe at open (that would dial on every search): Task 6's
            // index run records the probed value.
            ("audio_supported", "false".to_string()),
        ];
        for (key, value) in entries {
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES (?1, ?2)",
                params![key, value],
            )?;
        }
        Ok(())
    }

    /// Hard errors on `model_id`, `dimensions`, or `prefix_profile`
    /// mismatch; warns on a `backend`-only mismatch.
    fn validate_meta(&self, cfg: &Config, path: &Path) -> Result<()> {
        let has_meta: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta')",
            [],
            |r| r.get(0),
        )?;
        if !has_meta {
            return Err(Error::fatal(format!(
                "{} is not a ferret index (no meta table); pass a new --index path or delete the file",
                path.display()
            )));
        }

        let expected = [
            ("model_id", cfg.embedder.model.clone()),
            ("dimensions", cfg.embedder.dimensions.to_string()),
            ("prefix_profile", cfg.embedder.prefix_profile.to_string()),
        ];
        for (key, want) in expected {
            let found = self.get_meta(key)?;
            if found.as_deref() != Some(want.as_str()) {
                return Err(meta_mismatch(path, key, found.as_deref(), &want));
            }
        }

        let backend = self.get_meta("backend")?;
        let want_backend = cfg.embedder.backend.to_string();
        if backend.as_deref() != Some(want_backend.as_str()) {
            eprintln!(
                "warning: index backend is `{}` but config backend is `{want_backend}`; \
                 model, dimensions, and prefix profile match, so continuing",
                backend.as_deref().unwrap_or("<missing>"),
            );
        }
        Ok(())
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = stmt.query([key])?;
        Ok(match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        })
    }

    /// Upsert a meta key (used to record the pipeline fingerprints each
    /// `index` run was built with).
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)\
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            rusqlite::params![key, value],
        )?;
        Ok(())
    }

    /// Mark a file row complete (all chunks committed).
    pub fn mark_complete(&self, file_id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET complete = 1, error = NULL WHERE id = ?1",
            [file_id],
        )?;
        Ok(())
    }

    pub fn upsert_file(&self, row: &NewFile) -> Result<i64> {
        let id: i64 = self.conn.query_row(
            "INSERT INTO files (path, collection, kind, ext, size, mtime, fingerprint, \
             complete, next_page, skip_reason, indexed_at, error)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT (path) DO UPDATE SET
               collection = excluded.collection,
               kind = excluded.kind,
               ext = excluded.ext,
               size = excluded.size,
               mtime = excluded.mtime,
               fingerprint = excluded.fingerprint,
               complete = excluded.complete,
               next_page = excluded.next_page,
               skip_reason = excluded.skip_reason,
               indexed_at = excluded.indexed_at,
               error = excluded.error
             RETURNING id",
            params![
                row.path,
                row.collection,
                row.kind.to_string(),
                row.ext,
                row.size as i64,
                row.mtime,
                row.fingerprint,
                row.complete as i64,
                row.next_page,
                row.skip_reason,
                row.indexed_at,
                row.error,
            ],
            |r| r.get(0),
        )?;
        Ok(id)
    }

    pub fn get_file_by_path(&self, path: &str) -> Result<Option<FileRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, collection, kind, ext, size, mtime, fingerprint, complete, \
             next_page, skip_reason, indexed_at, error
             FROM files WHERE path = ?1",
        )?;
        let mut rows = stmt.query([path])?;
        Ok(match rows.next()? {
            Some(row) => Some(file_row(row)?),
            None => None,
        })
    }

    pub fn delete_file_by_path(&self, path: &str) -> Result<bool> {
        // chunks_vec and chunks_fts are keyed by chunks.id and have no
        // foreign keys, so clean them before the files delete cascades.
        for table in ["chunks_vec", "chunks_fts"] {
            self.conn.execute(
                &format!(
                    "DELETE FROM {table} WHERE rowid IN \
                     (SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id \
                      WHERE f.path = ?1)"
                ),
                [path],
            )?;
        }
        let deleted = self
            .conn
            .execute("DELETE FROM files WHERE path = ?1", [path])?;
        Ok(deleted > 0)
    }

    /// Mark complete inside an open transaction (folds the final flag flip
    /// into the write tx so a crash cannot leave fully-written rows at
    /// `complete=0`).
    pub fn mark_complete_in_tx(&self, tx: &Transaction<'_>, file_id: i64) -> Result<()> {
        tx.execute(
            "UPDATE files SET complete = 1, error = NULL WHERE id = ?1",
            [file_id],
        )?;
        Ok(())
    }

    /// Page numbers present in `chunks` for a file, ascending. Drives the
    /// PDF backfill check (a page below `next_page` with no chunk).
    pub fn chunk_pages(&self, file_id: i64) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT page FROM chunks WHERE file_id = ?1 AND page IS NOT NULL ORDER BY page",
        )?;
        let pages = stmt
            .query_map([file_id], |r| r.get(0))?
            .collect::<std::result::Result<Vec<Option<i64>>, _>>()?;
        Ok(pages.into_iter().flatten().collect())
    }

    /// Chunk id for one file page, if any. Point query for the per-page
    /// scheduler (cheaper than re-scanning `chunk_pages` per page).
    pub fn chunk_id_for_page(&self, file_id: i64, page: i64) -> Result<Option<i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM chunks WHERE file_id = ?1 AND page = ?2 LIMIT 1")?;
        let mut rows = stmt.query(rusqlite::params![file_id, page])?;
        Ok(match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        })
    }

    /// True when the page has a chunk WITH a vector. A chunk without a vec
    /// (image-pipeline bump deleted vecs only) still needs embedding work.
    pub fn page_has_vec(&self, file_id: i64, page: i64) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM chunks c JOIN chunks_vec v ON v.rowid = c.id \
             WHERE c.file_id = ?1 AND c.page = ?2",
            rusqlite::params![file_id, page],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// Delete one page's chunks (vec/fts follow via the sync triggers).
    /// Makes page processing idempotent: reprocessing a page never dupes.
    pub fn delete_page_chunks(
        &self,
        tx: &rusqlite::Transaction<'_>,
        file_id: i64,
        page: i64,
    ) -> Result<()> {
        tx.execute(
            "DELETE FROM chunks WHERE file_id = ?1 AND page = ?2",
            rusqlite::params![file_id, page],
        )?;
        Ok(())
    }

    /// Advance the resume cursor, optionally closing the file. `error`
    /// `None` clears a previous failure (a later success heals the row).
    pub fn set_pdf_progress(
        &self,
        tx: &rusqlite::Transaction<'_>,
        file_id: i64,
        next_page: i64,
        complete: bool,
        error: Option<&str>,
    ) -> Result<()> {
        tx.execute(
            "UPDATE files SET next_page = ?1, complete = ?2, error = ?3 WHERE id = ?4",
            rusqlite::params![next_page, complete as i64, error, file_id],
        )?;
        Ok(())
    }

    /// Delete one file's `pdf_page` vec rows (image-pipeline bump keeps
    /// chunks + text, re-embeds images from `next_page = 1`).
    pub fn delete_pdf_vecs(&self, file_id: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM chunks_vec WHERE rowid IN \
             (SELECT id FROM chunks WHERE file_id = ?1 AND kind = 'pdf_page')",
            [file_id],
        )?;
        Ok(())
    }

    /// Rewrite one page's stored text (text-pipeline bump refreshes FTS
    /// without touching vectors; the `chunks_fts_au` trigger follows).
    /// Empty text stores as NULL, like `insert_chunk`.
    pub fn update_chunk_text(&self, file_id: i64, page: i64, text: &str) -> Result<()> {
        let text = if text.is_empty() { None } else { Some(text) };
        self.conn.execute(
            "UPDATE chunks SET text = ?1 WHERE file_id = ?2 AND page = ?3",
            rusqlite::params![text, file_id, page],
        )?;
        Ok(())
    }

    /// ADD COLUMN backfill for pre-audio indexes (mirrors the trigger
    /// backfill above): only adds a column PRAGMA reports missing, so
    /// fresh and already-migrated opens are no-ops.
    fn backfill_audio_columns(&self) -> Result<()> {
        for col in ["t_start_ms", "t_end_ms"] {
            let missing: i64 = self.conn.query_row(
                "SELECT COUNT(*) FROM pragma_table_info('chunks') WHERE name = ?1",
                [col],
                |r| r.get(0),
            )?;
            if missing == 0 {
                self.conn
                    .execute_batch(&format!("ALTER TABLE chunks ADD COLUMN {col} INTEGER"))?;
            }
        }
        Ok(())
    }

    pub fn insert_chunk(&self, tx: &Transaction<'_>, chunk: NewChunk) -> Result<i64> {
        // Empty text is stored as SQL NULL.
        let text = chunk.text.as_deref().filter(|t| !t.is_empty());
        tx.execute(
            "INSERT INTO chunks (file_id, ordinal, page, kind, text, start_byte, end_byte,
             t_start_ms, t_end_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                chunk.file_id,
                chunk.ordinal,
                chunk.page,
                chunk.kind.to_string(),
                text,
                chunk.start_byte,
                chunk.end_byte,
                chunk.t_start_ms,
                chunk.t_end_ms,
            ],
        )?;
        Ok(tx.last_insert_rowid())
    }

    /// Insert one embedding; `rowid` is the `chunks.id` and `file_kind` is
    /// the FILES kind (text|image|pdf), never the chunk kind.
    #[allow(clippy::too_many_arguments)] // signature fixed by the task brief
    pub fn insert_vec(
        &self,
        tx: &Transaction<'_>,
        chunk_id: i64,
        embedding: &[f32],
        file_kind: FileKind,
        ext: &str,
        mtime: i64,
        collection: &str,
    ) -> Result<()> {
        // Global constraint: store only finite f32.
        if let Some(pos) = embedding.iter().position(|f| !f.is_finite()) {
            return Err(Error::fatal(format!(
                "refusing to store non-finite embedding value at index {pos} \
                 for chunk {chunk_id}"
            )));
        }
        if embedding.len() as u32 != self.dimensions {
            return Err(Error::fatal(format!(
                "embedding length {} does not match index dimensions {}",
                embedding.len(),
                self.dimensions
            )));
        }
        tx.execute(
            "INSERT INTO chunks_vec (rowid, embedding, file_kind, ext, mtime, collection)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                chunk_id,
                f32_blob(embedding),
                file_kind.to_string(),
                ext,
                mtime,
                collection,
            ],
        )?;
        Ok(())
    }

    /// Delete a file's `text` chunks (keeping `image` and `pdf_page`).
    pub fn delete_text_chunks_keep_images(&self, file_id: i64) -> Result<()> {
        self.delete_chunks_where(file_id, "kind = 'text'")
    }

    /// Delete a file's `image` and `pdf_page` chunks (keeping `text`).
    pub fn delete_image_chunks_keep_text(&self, file_id: i64) -> Result<()> {
        self.delete_chunks_where(file_id, "kind IN ('image', 'pdf_page')")
    }

    /// `kind_filter` is always a compile-time literal, never user input.
    fn delete_chunks_where(&self, file_id: i64, kind_filter: &str) -> Result<()> {
        for table in ["chunks_vec", "chunks_fts"] {
            self.conn.execute(
                &format!(
                    "DELETE FROM {table} WHERE rowid IN \
                     (SELECT id FROM chunks WHERE file_id = ?1 AND {kind_filter})"
                ),
                [file_id],
            )?;
        }
        self.conn.execute(
            &format!("DELETE FROM chunks WHERE file_id = ?1 AND {kind_filter}"),
            [file_id],
        )?;
        Ok(())
    }

    pub fn status_counts(&self) -> Result<StatusCounts> {
        let count = |sql: &str| -> Result<u64> {
            let n: i64 = self.conn.query_row(sql, [], |r| r.get(0))?;
            Ok(n as u64)
        };
        let indexed_at_max: Option<i64> =
            self.conn
                .query_row("SELECT max(indexed_at) FROM files", [], |r| r.get(0))?;
        Ok(StatusCounts {
            files_text: count("SELECT count(*) FROM files WHERE kind = 'text'")?,
            files_image: count("SELECT count(*) FROM files WHERE kind = 'image'")?,
            files_pdf: count("SELECT count(*) FROM files WHERE kind = 'pdf'")?,
            chunks: count("SELECT count(*) FROM chunks")?,
            incomplete: count("SELECT count(*) FROM files WHERE complete = 0")?,
            errors: count("SELECT count(*) FROM files WHERE error IS NOT NULL")?,
            indexed_at_max,
        })
    }

    /// `&self`, not `&mut self`: callers keep using the store's `&self`
    /// methods (`upsert_file`, `insert_chunk`, `insert_vec`) while the
    /// transaction is open, so an exclusive borrow here would not compile
    /// at those call sites. `new_unchecked` trades the compile-time
    /// no-nesting guarantee for a runtime one (BEGIN fails if a transaction
    /// is already open on this connection).
    pub fn begin(&self) -> Result<Transaction<'_>> {
        Ok(Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Deferred,
        )?)
    }
}

fn file_row(row: &rusqlite::Row<'_>) -> Result<FileRow> {
    let kind: String = row.get("kind")?;
    Ok(FileRow {
        id: row.get("id")?,
        path: row.get("path")?,
        collection: row.get("collection")?,
        kind: FileKind::from_str(&kind).map_err(Error::fatal)?,
        ext: row.get("ext")?,
        size: row.get::<_, i64>("size")? as u64,
        mtime: row.get("mtime")?,
        fingerprint: row.get("fingerprint")?,
        complete: row.get::<_, i64>("complete")? != 0,
        next_page: row.get("next_page")?,
        skip_reason: row.get("skip_reason")?,
        indexed_at: row.get("indexed_at")?,
        error: row.get("error")?,
    })
}

/// `"3.38.0"`-style version at or above `major.minor`.
fn sqlite_at_least(version: &str, want_major: u32, want_minor: u32) -> bool {
    let mut parts = version.split('.');
    let major: u32 = parts.next().unwrap_or("0").parse().unwrap_or(0);
    let minor: u32 = parts.next().unwrap_or("0").parse().unwrap_or(0);
    (major, minor) >= (want_major, want_minor)
}

/// The hard-error message for a model/dimensions/prefix_profile mismatch.
fn meta_mismatch(path: &Path, key: &str, found: Option<&str>, expected: &str) -> Error {
    Error::fatal(format!(
        "index {} mismatch: this index was built with `{}` but the merged config has \
         `{expected}`; pass a new --index path or delete the index file {} to rebuild",
        key,
        found.unwrap_or("<missing>"),
        path.display()
    ))
}

/// sqlite-vec reads `FLOAT[d]` blobs as a C `float` array, so the byte order
/// is the platform's native order. Written explicitly little-endian to
/// match the query-side `vec_blob` (identical bytes on all v1 targets,
/// which are all little-endian).
fn f32_blob(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}
