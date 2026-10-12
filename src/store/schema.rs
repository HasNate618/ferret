//! Index schema DDL (spec §5.1).
//!
//! `dimensions` is interpolated only into the vec0 embedding width
//! (`FLOAT[{d}]`); no path or other user input is ever concatenated into
//! this SQL.
//!
//! The `chunks_fts` triggers (after insert/update/delete on `chunks`, and a
//! rewrite when `files.path` changes) land with indexing, not here; they use
//! the `ferret_basename(path)` scalar function that `store::open` registers
//! on every connection.

/// Complete DDL for a new index: `meta`, `files`, `chunks`, the FTS5 index
/// over chunk text + filename + path, and the `vec0` vector table whose
/// `rowid` mirrors `chunks.id`. Its `file_kind` metadata column is the
/// `files.kind` (text|image|pdf), never the chunk kind.
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
"#
    )
}
