//! Index schema DDL (spec §5.1).
//!
//! `dimensions` is interpolated only into the vec0 embedding width
//! (`FLOAT[{d}]`); no path or other user input is ever concatenated into
//! this SQL.

/// Complete DDL for a new index: `meta`, `files`, `chunks`, the FTS5 index
/// over chunk text + filename + path, and the `vec0` vector table whose
/// `rowid` mirrors `chunks.id`. Its `file_kind` metadata column is the
/// `files.kind` (text|image|pdf), never the chunk kind.
///
/// Includes [`triggers`] so a fresh database is fully wired in one shot.
pub fn ddl(dimensions: u32) -> String {
    let d = dimensions;
    format!(
        r#"
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
  embedding FLOAT[{d}],
  file_kind TEXT,
  ext TEXT,
  mtime INTEGER,
  collection TEXT
);
{trigs}
"#,
        trigs = triggers(),
    )
}

/// Idempotent sync triggers, applied on EVERY `store::open` (not just
/// creation) so indexes built before a trigger existed get backfilled:
/// `chunks_fts` mirrors `chunks` (spec §5.1; `chunks_fts` is CONTENTFUL,
/// so the delete form MUST be the plain `DELETE ... WHERE rowid = old.id`),
/// and `chunks_vec` rows die with their chunk (vec0 has no foreign keys).
/// filename/path come from the file row via the `ferret_basename` scalar
/// `store::open` registers.
pub fn triggers() -> &'static str {
    r#"
CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text, filename, path)
  SELECT new.id, new.text, ferret_basename(f.path), f.path
  FROM files f WHERE f.id = new.file_id;
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
  DELETE FROM chunks_fts WHERE rowid = old.id;
  INSERT INTO chunks_fts(rowid, text, filename, path)
  SELECT new.id, new.text, ferret_basename(f.path), f.path
  FROM files f WHERE f.id = new.file_id;
END;
CREATE TRIGGER IF NOT EXISTS chunks_vec_ad AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_vec WHERE rowid = old.id;
END;
"#
}
