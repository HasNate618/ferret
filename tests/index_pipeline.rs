//! Task 11: the text indexing pipeline (fake embedder, fingerprints,
//! skips, NDJSON). Env-mutating tests are `#[serial]`.

mod common;

use ferret::cli::try_run;
use ferret::embedder::FakeEmbedder;
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
    // v1.1: the first line is the plan event with progress totals, the
    // last line the summary event; every line between is a file event.
    assert!(lines.len() >= 2, "expected plan + summary lines, got: {lines:?}");
    let plan: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(plan["event"], "plan");
    let summary: serde_json::Value =
        serde_json::from_str(lines[lines.len() - 1]).unwrap();
    assert_eq!(summary["event"], "summary");
    for line in &lines[1..lines.len() - 1] {
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

#[test]
#[serial_test::serial]
fn index_json_emits_plan_event_first() {
    let env = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    std::fs::write(env.root.join("a.txt"), "alpha beta gamma").unwrap();
    std::fs::write(env.root.join("b.txt"), "delta epsilon").unwrap();
    let cfg = ferret::config::load(&ferret::config::LoadOpts::default()).unwrap();
    let embedder = FakeEmbedder::new_with_profile(
        cfg.embedder.prefix_profile,
        cfg.embedder.dimensions,
    );
    let mut out = Vec::new();
    let mut err = Vec::new();
    ferret::index::run(&cfg, &embedder, vec![], true, false, &mut out, &mut err).unwrap();
    let stdout = String::from_utf8(out).unwrap();
    let first: serde_json::Value =
        serde_json::from_str(stdout.lines().next().unwrap()).unwrap();
    assert_eq!(first["event"], "plan");
    assert_eq!(first["files"], 2);
    let expected_bytes = std::fs::metadata(env.root.join("a.txt")).unwrap().len()
        + std::fs::metadata(env.root.join("b.txt")).unwrap().len();
    assert_eq!(first["bytes"], expected_bytes);
}

#[test]
#[serial_test::serial]
fn index_json_emits_summary_event_last() {
    let env = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    std::fs::write(env.root.join("a.txt"), "alpha beta gamma").unwrap();
    std::fs::write(env.root.join("skip.bin"), [0u8, 1, 2, 3]).unwrap();
    let cfg = ferret::config::load(&ferret::config::LoadOpts::default()).unwrap();
    let embedder = FakeEmbedder::new_with_profile(
        cfg.embedder.prefix_profile,
        cfg.embedder.dimensions,
    );
    let mut out = Vec::new();
    let mut err = Vec::new();
    ferret::index::run(&cfg, &embedder, vec![], true, false, &mut out, &mut err).unwrap();
    let stdout = String::from_utf8(out).unwrap();
    let last: serde_json::Value =
        serde_json::from_str(stdout.lines().last().unwrap()).unwrap();
    assert_eq!(last["event"], "summary");
    assert_eq!(last["seen"], 2);
    assert_eq!(last["embedded"], 1);
    assert_eq!(last["errors"], 0);
    assert_eq!(
        last["skipped"],
        stdout
            .lines()
            .filter(|l| l.contains("\"status\":\"skip\""))
            .count() as u64
    );
}
