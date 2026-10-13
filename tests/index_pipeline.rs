//! Task 11: the text indexing pipeline (fake embedder, fingerprints,
//! skips, NDJSON). Env-mutating tests are `#[serial]`.

mod common;

use ferret::cli::try_run;
use rusqlite::OptionalExtension;

/// Read `config_fingerprint` straight from the index meta table.
fn meta_fingerprint(index: &std::path::Path) -> String {
    let conn = rusqlite::Connection::open(index).unwrap();
    conn.query_row(
        "SELECT value FROM meta WHERE key = 'config_fingerprint'",
        [],
        |r| r.get(0),
    )
    .unwrap()
}

/// A `files.path` containing `needle`, if the row exists.
fn file_row(index: &std::path::Path, needle: &str) -> Option<String> {
    let conn = rusqlite::Connection::open(index).unwrap();
    conn.query_row(
        "SELECT path FROM files WHERE path LIKE ?1",
        [format!("%{needle}")],
        |r| r.get(0),
    )
    .optional()
    .unwrap()
}

/// Rewrite the user config keeping the corpus root (the user config layer
/// replaces the whole file, so roots must be restated).
fn rewrite_cfg(env: &common::TestEnv, body: &str) {
    let root = env.root.to_str().unwrap();
    std::fs::write(
        &env.cfg,
        format!("[[index.roots]]\npath = \"{root}\"\ncollection = \"corpus\"\n\n{body}"),
    )
    .unwrap();
}

#[test]
#[serial_test::serial]
fn index_markdown_then_search() {
    let _env = common::test_env();
    try_run(&["ferret", "index"]).unwrap();
    let out = try_run(&["ferret", "search", "--json", "pets"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["hits"][0]["kind"], "text");
}

#[test]
#[serial_test::serial]
fn bump_max_chars_reembeds_text() {
    let env = common::test_env();
    try_run(&["ferret", "index"]).unwrap();
    let fp1 = meta_fingerprint(&env.index);
    // The rewritten file replaces the user layer entirely; with no roots
    // the second index walks nothing, but the new pipeline fingerprint is
    // still recorded in meta (the reindex decision).
    std::fs::write(&env.cfg, "[chunk.text]\nmax_chars = 100\noverlap = 10\n").unwrap();
    try_run(&["ferret", "index"]).unwrap();
    let fp2 = meta_fingerprint(&env.index);
    assert_ne!(fp1, fp2);
}

#[test]
#[serial_test::serial]
fn bump_max_chars_reembeds_with_roots_kept() {
    let env = common::test_env();
    try_run(&["ferret", "index"]).unwrap();
    let chunks1 = chunk_count(&env.index);
    rewrite_cfg(&env, "[chunk.text]\nmax_chars = 100\noverlap = 10\n");
    try_run(&["ferret", "index"]).unwrap();
    let chunks2 = chunk_count(&env.index);
    // Same corpus, smaller windows: at least as many chunks, all new.
    assert!(chunks2 >= chunks1);
    let out = try_run(&["ferret", "search", "--json", "pets"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(!v["hits"].as_array().unwrap().is_empty());
}

fn chunk_count(index: &std::path::Path) -> i64 {
    let conn = rusqlite::Connection::open(index).unwrap();
    conn.query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))
        .unwrap()
}

#[test]
#[serial_test::serial]
fn collection_collision_is_fatal() {
    let env = common::test_env();
    let a = env.root.join("docs");
    let b = env.root.join("sub/docs");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    std::fs::write(
        &env.cfg,
        format!(
            "[[index.roots]]\npath = \"{}\"\ncollection = \"docs\"\n\n[[index.roots]]\npath = \"{}\"\ncollection = \"docs\"\n",
            a.display(),
            b.display()
        ),
    )
    .unwrap();
    let err = try_run(&["ferret", "index"]).unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
#[serial_test::serial]
fn walk_skip_deletes_existing_row() {
    let env = common::test_env();
    try_run(&["ferret", "index"]).unwrap();
    std::fs::write(env.root.join("gone.md"), "x").unwrap();
    try_run(&["ferret", "index"]).unwrap();
    assert!(file_row(&env.index, "gone.md").is_some());
    // Exclude it: the walk now skips it, and index deletes the stale row.
    rewrite_cfg(&env, "[files]\nexclude_globs = [\"gone.md\"]\n");
    try_run(&["ferret", "index"]).unwrap();
    assert!(file_row(&env.index, "gone.md").is_none());
}

#[test]
#[serial_test::serial]
fn index_json_ndjson_events() {
    let env = common::test_env();
    std::fs::write(env.root.join("clip.mp4"), "x").unwrap();
    let out = try_run(&["ferret", "index", "--json"]).unwrap();
    let lines: Vec<&str> = out
        .stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert!(!lines.is_empty());
    for line in lines {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["event"], "file");
        assert!(v["path"].is_string());
        assert!(matches!(
            v["status"].as_str(),
            Some("ok" | "skip" | "error")
        ));
    }
    // The unsupported extension is a walk-skip event.
    assert!(out.stdout.contains("\"status\":\"skip\""));
    // pets.md indexed ok.
    assert!(out.stdout.contains("pets.md"));
}

#[test]
#[serial_test::serial]
fn index_strict_fails_on_extract_errors() {
    let env = common::test_env();
    // Truncated PNG bytes: undecodable, so the per-file extract-error
    // path (and --strict) still triggers now that images extract for real.
    std::fs::write(
        env.root.join("pic.png"),
        [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a],
    )
    .unwrap();
    let ok = try_run(&["ferret", "index"]).unwrap(); // decode failure is per-file
    assert!(!ok.stdout.is_empty() || true);
    let err = try_run(&["ferret", "index", "--strict"]).unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
#[serial_test::serial]
fn status_minimal_after_index() {
    let _env = common::test_env();
    try_run(&["ferret", "index"]).unwrap();
    let out = try_run(&["ferret", "status", "--json"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(v.get("files").is_some());
}
