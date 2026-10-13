# Task 5 Report: Store schema, meta, CRUD

**Status:** DONE_WITH_CONCERNS (behavior complete and tested; three
brief-internal conflicts resolved with the least-deviating option, flagged
below for Task 11 owners)
**Commit:** `95a44a5` (`feat: add SQLite store with sqlite-vec and FTS5 schema`)

## Implemented behavior

- `store::open(path, cfg, images_supported)`:
  - Registers sqlite-vec 0.1.9 in-process via
    `rusqlite::ffi::sqlite3_auto_extension` (guarded by a `OnceLock` so it
    runs once per process, before the first connection; nonzero rc →
    `Error::fatal`). No runtime `.so` load.
  - Pragmas on every open: `journal_mode=WAL` (verified, fatal if not WAL),
    `busy_timeout=5000`, `foreign_keys=ON`; refuses `sqlite_version()` <
    3.38.0 with a fatal error.
  - Registers the `ferret_basename(path)` scalar function
    (`create_scalar_function`, UTF8 | DETERMINISTIC) on every connection —
    it returns the segment after the last `/`. The `chunks_fts` triggers
    that consume it land with indexing (Task 11), per the task direction.
  - New file: runs `schema::ddl(cfg.embedder.dimensions)` and writes meta:
    `schema_version=1`, `model_id` (= `embedder.model`), `dimensions`,
    `backend`, `prefix_profile`, `images_supported` ("true"/"false"),
    `config_fingerprint`, `text_pipeline`, `image_pipeline` (Task 4
    canonical JSON).
  - Existing file: fatal (exit 1) if `model_id`, `dimensions`, or
    `prefix_profile` differ from the merged config — message tells the user
    to pass a new `--index` or delete the sqlite file. A `backend`-only
    difference prints an `eprintln!` warning and continues. A file without
    a `meta` table (non-ferret sqlite) is fatal. No method rewrites meta, so
    `ferret rm` cannot change it.
- `schema::ddl(dimensions)` (src/store/schema.rs): the brief's DDL block
  verbatim — `meta`, `files`, `chunks` (FK `ON DELETE CASCADE`), FTS5
  `chunks_fts(text, filename, path, tokenize='unicode61')`, vec0
  `chunks_vec(embedding FLOAT[{d}], file_kind, ext, mtime, collection)`.
  Only `{d}` is interpolated; no paths are concatenated.
- `Store` methods per the brief's signatures: `get_meta`, `upsert_file`
  (INSERT ... ON CONFLICT(path) DO UPDATE ... RETURNING id), `get_file_by_path`,
  `delete_file_by_path`, `insert_chunk` (tx-scoped, empty text → SQL NULL),
  `insert_vec` (tx-scoped, `rowid = chunks.id`, `file_kind` is the FILES
  kind; refuses non-finite f32 and wrong-length embeddings), 
  `delete_text_chunks_keep_images` / `delete_image_chunks_keep_text`
  (also clean the matching `chunks_vec`/`chunks_fts` rows, since vec0 has no
  FK cascades), `status_counts`, `begin`, plus `conn()` / `dimensions`.
- `delete_file_by_path` removes the file's `chunks_vec` and `chunks_fts`
  rows before deleting the `files` row (whose FK cascade removes `chunks`).

## Files changed

- `Cargo.toml` — `rusqlite = { version = "0.40", features = ["bundled", "functions"] }`,
  `sqlite-vec = "=0.1.9"` (+ `Cargo.lock`).
- `src/store/mod.rs` (new) — store open/validate/CRUD as above.
- `src/store/schema.rs` (new) — `ddl(dimensions)`.
- `src/lib.rs` — `pub mod store;`.
- `tests/store_schema.rs` (new) — the brief's four tests verbatim (modulo
  the `mut`/signature fix below) plus source-named tests for meta round-trip,
  backend-only warning, WAL mode, sqlite ≥ 3.38, empty-text-as-NULL, upsert/
  get/delete round-trip, file-delete cascades, chunk-kind delete sides, and
  status counts (15 tests total).

## Test results

- `cargo test` — 51 passed, 0 failed (5 unit, 12 cli_init, 16 config_merge,
  3 paths, 15 store_schema, 0 doctests), zero warnings.
  Tail of `cargo test --test store_schema`:
  ```
  test delete_file_cascades_chunks_and_vec_rows ... ok
  test status_counts_reflects_rows ... ok
  test delete_chunk_kind_sides_keep_the_other_side ... ok
  test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
  ```
- `cargo clippy --all-targets` — zero warnings.
- TDD followed: test file written first; `cargo test --test store_schema`
  failed to compile (no `ferret::store`, no `rusqlite`), then passed after
  implementation.

## Self-review findings and fixes

1. **Vacuous version check (fixed):** first draft of `sqlite_at_least`
   shadowed its `major`/`minor` parameters with the parsed values, making
   the comparison always true. Renamed to `want_major`/`want_minor`.
2. **`begin(&mut self)` could not compile the brief's own test (fixed):**
   the brief's test calls `store.upsert_file(&self, ...)` *after*
   `store.begin()`, which is impossible while the returned
   `Transaction<'_>` holds the exclusive borrow. Changed `begin` to
   `&self` using the public, safe `Transaction::new_unchecked(&conn,
   Deferred)`; nesting is rejected at runtime instead ("cannot start a
   transaction within a transaction"). Consequence: the tests' `let mut tx`
   became `let tx` (commit consumes the transaction), removing an
   `unused_mut` warning.
3. **Transmute needed an explicit annotation (fixed):** clippy's
   deny-by-default `missing_transmute_annotations` rejected the brief's
   un-annotated `transmute`; the annotation had to be the full 3-arg
   extension-init signature (the un-annotated form infers it from the
   `sqlite3_auto_extension` parameter, so semantics are identical).
4. **FTS5 'delete' command verified empirically before relying on it:**
   on a *contentful* FTS5 table (the brief's DDL block), 
   `INSERT INTO chunks_fts(chunks_fts, rowid) VALUES('delete', old.id)`
   errors ("SQL logic error"); on a *contentless* table it is accepted but
   silently leaves stale index entries (needs the old column values).
   Since the DDL block is the stated contract and triggers are Task 11's,
   no triggers were added here; see concern 1 for what Task 11 must do.

## Concerns / deviations for the plan owner

1. **Brief-internal conflict on `chunks_fts` (affects Task 11):** the prose
   calls `chunks_fts` "contentless", but the DDL block is a plain contentful
   FTS5 table (no `content=''`). I followed the DDL block verbatim as
   directed. Task 11's delete trigger must therefore use
   `DELETE FROM chunks_fts WHERE rowid = old.id` (works on contentful
   tables), **not** the brief's `INSERT INTO chunks_fts(chunks_fts, rowid)
   VALUES('delete', old.id)` (errors on contentful; silently corrupts
   contentless-without-values). Verified against SQLite 3.53.3.
2. **`begin` signature:** `&self` instead of the brief's `&mut self` —
   forced by the brief's own verbatim test (see self-review 2). All other
   signatures match the brief exactly.
3. **`functions` feature added to rusqlite:** the brief's Cargo.toml line
   had only `bundled`, but the brief also requires
   `conn.create_scalar_function`, which is behind rusqlite's `functions`
   feature.
4. **`conn()` is a public accessor returning `&Connection`:** the brief says
   keep `conn` private "except a `#[cfg(test)]` accessor", but integration
   tests (which the brief's test code is) cannot see `#[cfg(test)]` items,
   and the brief's test calls `store.conn()` verbatim. The field stays
   private; the accessor hands out only a shared reference.
5. **Chunk-kind "sides" interpretation:** `delete_text_chunks_keep_images`
   deletes `kind='text'`; `delete_image_chunks_keep_text` deletes
   `kind IN ('image','pdf_page')` (pdf_page is a page-image embedding, i.e.
   the image pipeline). If the spec splits pdf_page differently, Task 11/12
   should revisit; the split is centralized in `delete_chunks_where` calls.
6. **Formatting:** the repo is not rustfmt-formatted (pre-existing files
   fail `cargo fmt --check`), so new files were left in the codebase's
   existing compact style rather than diverging.

## Commit

- `95a44a5` — feat: add SQLite store with sqlite-vec and FTS5 schema
  (`Cargo.toml`, `Cargo.lock`, `src/store/`, `src/lib.rs`,
  `tests/store_schema.rs`).
