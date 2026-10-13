### Task 11: Index text files (fake embedder)

**Files:**
- Fill: `src/index/mod.rs`
- Modify: `src/cli.rs` (`index`, `status` minimal), `tests/common.rs` (`test_env`)
- Test: `tests/index_pipeline.rs`
- Create stubs: `src/extract/image.rs`, `src/extract/pdf.rs` (replaced in Tasks 14–15)

**Test helper** (`tests/common.rs`):

```rust
pub struct TestEnv {
    pub tmp: tempfile::TempDir,
    pub root: PathBuf,
    pub cfg: PathBuf,
    pub index: PathBuf,
}

pub fn test_env() -> TestEnv {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("XDG_CONFIG_HOME", tmp.path().join("xdg-cfg"));
    std::env::set_var("XDG_DATA_HOME", tmp.path().join("xdg-data"));
    std::env::set_var("XDG_CACHE_HOME", tmp.path().join("xdg-cache"));
    std::env::set_var("FERRET_EMBEDDER", "fake");
    let root = tmp.path().join("corpus");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("pets.md"), "pets in the lease").unwrap();
    ferret::cli::try_run(&[
        "ferret", "init", "--root", root.to_str().unwrap(),
    ]).unwrap();
    TestEnv {
        cfg: tmp.path().join("xdg-cfg/ferret/config.toml"),
        index: tmp.path().join("xdg-data/ferret/index.sqlite"),
        root,
        tmp,
    }
}
```

Mark CLI tests that call `test_env` / `set_var` with `#[serial]`.

**Interfaces:**
- Consumes: walk, extract_text, embedder, store, fingerprint
- Produces: `index::run(cfg, embedder, extra_paths, json_ndjson: bool, strict: bool) -> Result<IndexSummary>`

```rust
pub struct IndexSummary {
    pub seen: u64,
    pub skipped: u64,
    pub embedded: u64,
    pub errors: u64,
}
```

Stderr summary: files seen / skipped / embedded / errors.

`--json` on index: NDJSON stdout events `{"event":"file","path":"...","status":"ok|skip|error","reason":null or str,"error":null or str}`. Walk-skip has `path` and no DB id.

**Per text file:** one SQLite transaction (extract + embed + insert). `numeric_heavy` chunks: insert chunk+FTS, skip vec. Letterless dropped before insert.

Fingerprint skip: unchanged fingerprint **and** `complete=1` → skip extract/embed. `complete=0` → retry whole text file.

Pipeline granularity vs stored meta:
- `text_pipeline` only changed: delete text chunks (and PDF text per Task 15); keep image vecs
- `image_pipeline` only: Task 14
- both / `extension_kind` / `embed_recipe` / `prefix_version`: full reindex `next_page=1`

Unchanged `config_fingerprint` is the skip fast-path.

**Collections:** unique. Default last component of canonical root. Explicit `collection` overrides. Collision at start of `index`/`init` → exit 1. Extra paths under a root inherit; extra paths not under a root use last component and error if that name exists.

`index` does not delete missing-from-disk rows (`gc` does). It **does** delete walked paths that now match walk-skip.

Per-file extract errors: set `files.error`, `complete=0`, continue. `--strict` → exit 1 if any. Fatal: DB, embedder unreachable, schema mismatch.

Upload warning once on stderr if backend is `openai` or `sglang`.

Probe `images_supported` once per process (Task 14). For text-only this task: if fake, true; if HTTP not implemented, still index text.

- [ ] **Step 1: Write failing tests** using `FERRET_EMBEDDER=fake` and temp index/roots:

```rust
#[test]
fn index_markdown_then_search() {
    let env = test_env(); // sets XDG_*, FERRET_EMBEDDER=fake, writes a root with pets.md
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out = ferret::cli::try_run(&["ferret", "search", "--json", "pets"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["hits"][0]["kind"], "text");
}

#[test]
fn bump_max_chars_reembeds_text() {
    let env = test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let fp1 = meta_fingerprint(&env.index);
    std::fs::write(env.cfg, "[chunk.text]\nmax_chars = 100\noverlap = 10\n").unwrap();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let fp2 = meta_fingerprint(&env.index);
    assert_ne!(fp1, fp2);
}

#[test]
fn collection_collision_is_fatal() {
    let err = index_two_roots_named_docs().unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
fn walk_skip_deletes_existing_row() {
    let env = test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    std::fs::write(env.root.join("gone.md"), "x").unwrap();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    std::fs::remove_file(env.root.join("gone.md")).unwrap();
    // still in DB until gc — this test instead adds exclude_globs = ["gone.md"] and reindexes
    assert!(file_row(&env.index, "gone.md").is_none());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test index_pipeline`
Expected: FAIL

- [ ] **Step 3: Implement text indexing**

Put this match in `index::run`. Do **not** invent a skip reason. Image/Pdf call extract functions that Task 14/15 replace; until then they return `Err(Error::fatal("not implemented"))` which `index::run` **catches per file** (does not abort the walk): writes `files.error`, `complete=0`, NDJSON `status=error`, continues. Embedder-down (`endpoint` unreachable) is still fatal and aborts.

```rust
match kind {
    FileKind::Text => index_text_file(/* one SQLite transaction */)?,
    FileKind::Image => match extract::image::extract_image(&path, &cfg.chunk.image, &cache, cfg.files.heic) {
        Ok(chunks) => commit_file(chunks)?,
        Err(e) => record_extract_error(&path, e),
    },
    FileKind::Pdf => match extract::pdf::extract_pdf(/* ... */) {
        Ok(_) => { /* Task 15 uses per-page txs; this arm becomes that loop */ }
        Err(e) => record_extract_error(&path, e),
    },
}

pub fn extract_image(...) -> Result<Vec<Chunk>> {
    Err(Error::fatal("image extract not implemented"))
}
pub fn extract_pdf(...) -> Result<Vec<Chunk>> {
    Err(Error::fatal("pdf extract not implemented"))
}
```

`record_extract_error` is not `Error::fatal` to the CLI: increment `IndexSummary.errors`, set the row, continue. `--strict` → after the walk, if `errors > 0` return `Error::fatal`. Create `src/extract/image.rs` and `src/extract/pdf.rs` with those stubs in this task so the crate compiles.

- [ ] **Step 4: Run tests**

Run: `cargo test --test index_pipeline`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/index src/cli.rs tests/index_pipeline.rs
git commit -m "$(cat <<'EOF'
feat: index text files with fake embeddings and fingerprints

EOF
)"
```

---

