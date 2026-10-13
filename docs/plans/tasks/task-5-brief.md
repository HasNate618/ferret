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

