### Task 10: Walk, skip rules, gitignore, globs

**Files:**
- Create: `src/index/skip.rs`, `src/index/walk.rs`, `src/index/mod.rs` (stubs)
- Modify: `Cargo.toml` (`ignore = "0.4"`, `globset = "0.4"`)
- Test: `tests/walk_skip.rs`

**Interfaces:**
- Consumes: `Config.files`, `FileKind`, `extension_kind`
- Produces:

```rust
pub enum SkipReason {
    UnsupportedExt,
    SkipRule,
    NoImageBackend,
    HeicSkip,
    NoMagick,
    Gitignore,
    IncludeGlobs,
}

impl SkipReason {
    pub fn as_str(&self) -> &'static str; // spec NDJSON reasons
}

pub enum WalkEvent {
    File { path: PathBuf, collection: String, kind: FileKind, ext: String },
    Skip { path: PathBuf, reason: SkipReason },
}

pub fn walk(roots: &[RootEntry], extra: &[PathBuf], files_cfg: &FilesConfig, follow_symlinks: bool) -> Result<Vec<WalkEvent>>;
pub fn kind_for_ext(ext: &str, map: &BTreeMap<String, FileKind>) -> Option<FileKind>;
```

Walk order after extension typing: skip-rule list → include_globs (if non-empty, relative path from collection root must match ≥1 glob) → respect_gitignore via `ignore` crate when a `.git` exists at or above the file (at or below a configured root). A configured root path is never gitignored away.

Skip-rule list (walk-skip), names apply even when the root is hidden:
- path component in `{.git, node_modules, .cache, target, dist, __pycache__, .venv, venv}`
- hidden components (name starts with `.`) except the configured root itself, unless `include_hidden`
- basename globs: `*.pem` `*.key` `*.p8` `*.p12` `*.pfx` `.netrc` `*.netrc` `.env` `.env.*` `credentials.json` `*.sqlite-journal`
- basename `id_rsa` `id_dsa` `id_ecdsa` `id_ed25519` or `id_*` not ending in `.pub`
- size > `max_size`
- user `exclude_globs`
- include_globs miss
- gitignore hit

Never follow symlinks unless configured. Unknown extensions: walk-skip (no new files row). `.heic`/`.heif` with `heic=skip` → `HeicSkip`. Do not run magick here; missing magick is Task 14.

If a walk-skipped path **already exists in DB**, Task 11 deletes it. Walk only reports the skip.

- [ ] **Step 1: Write failing tests**

```rust
fn fixture_tree() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("corpus");
    std::fs::create_dir_all(root.join(".hidden")).unwrap();
    std::fs::write(root.join("ok.md"), "hi").unwrap();
    std::fs::write(root.join("secret.pem"), "k").unwrap();
    std::fs::write(root.join(".hidden/a.txt"), "x").unwrap();
    std::fs::write(root.join("skipme.rs"), "fn x(){}").unwrap();
    (tmp, root)
}

#[test]
fn pem_is_skip_rule() {
    let (_tmp, root) = fixture_tree();
    let ev = walk_one(&root);
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::Skip { reason: SkipReason::SkipRule, .. })));
}

#[test]
fn hidden_skipped_unless_include_hidden() {
    let (_tmp, root) = fixture_tree();
    let ev = walk_one(&root);
    assert!(ev.iter().any(|e| match e {
        WalkEvent::Skip { path, reason: SkipReason::SkipRule } => path.ends_with("a.txt"),
        _ => false,
    }));
}

#[test]
fn include_globs_skips_non_matching() {
    let (_tmp, root) = fixture_tree();
    let mut cfg = files_cfg();
    cfg.include_globs = vec!["*.md".into()];
    let ev = walk(&[root], &cfg);
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::Skip { reason: SkipReason::IncludeGlobs, .. })));
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::File { kind: FileKind::Text, ext, .. } if ext == "md")));
}

#[test]
fn gitignore_skip() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("repo");
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::write(root.join(".gitignore"), "skipme.txt\n").unwrap();
    std::fs::write(root.join("skipme.txt"), "x").unwrap();
    std::fs::write(root.join("keep.md"), "y").unwrap();
    let ev = walk(&[root], &files_cfg());
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::Skip { reason: SkipReason::Gitignore, .. })));
}
```

Helper `walk_one` in the test file wraps `walk` with compiled `FilesConfig` defaults. Extra path under a root inherits that root's collection — assert in a fifth test: `walk(&[root], extra: &[root.join("ok.md")])` yields `collection == root.file_name()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test walk_skip`
Expected: FAIL

- [ ] **Step 3: Implement walk using `ignore::WalkBuilder`** only when `respect_gitignore`; otherwise `walkdir` **or** `ignore` with git disabled. Prefer `ignore::WalkBuilder::new.hidden(false)` and apply hidden rules ourselves so the configured root may be `~/.ssh`. Do **not** add `walkdir` if `ignore` can do both (`git_ignore(false)`).

- [ ] **Step 4: Run tests**

Run: `cargo test --test walk_skip`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/index tests/walk_skip.rs Cargo.toml
git commit -m "$(cat <<'EOF'
feat: walk corpus with skip rules, gitignore, and include globs

EOF
)"
```

---

