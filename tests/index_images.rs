//! Task 14: image indexing — capability-skip rows, re-embed on probe
//! flip, text-pipeline bumps keep image vecs, HEIC magick gating.

mod common;

use common::{EnvGuard, TestEnv};
use ferret::embedder::Embedder;
use serial_test::serial;

fn png_bytes() -> Vec<u8> {
    std::fs::read("testdata/one.png").unwrap()
}

fn files_row(index: &std::path::Path, name: &str) -> Option<(i64, Option<String>, Option<String>)> {
    // (complete, skip_reason, error) for the file whose path ends in `name`.
    let conn = rusqlite::Connection::open(index).unwrap();
    let mut stmt = conn
        .prepare("SELECT complete, skip_reason, error FROM files WHERE path LIKE ?1")
        .unwrap();
    let mut rows = stmt.query([format!("%{name}")]).unwrap();
    rows.next()
        .unwrap()
        .map(|r| (r.get(0).unwrap(), r.get(1).unwrap(), r.get(2).unwrap()))
}

fn count(index: &std::path::Path, table: &str) -> i64 {
    let conn = rusqlite::Connection::open(index).unwrap();
    conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
        .unwrap()
}

fn files_fingerprint(index: &std::path::Path, name: &str) -> String {
    let conn = rusqlite::Connection::open(index).unwrap();
    conn.query_row(
        "SELECT fingerprint FROM files WHERE path LIKE ?1",
        [format!("%{name}")],
        |r| r.get(0),
    )
    .unwrap()
}

fn vec_blob_for(index: &std::path::Path, name: &str) -> Vec<u8> {
    let conn = rusqlite::Connection::open(index).unwrap();
    conn.query_row(
        "SELECT v.embedding FROM chunks_vec v JOIN chunks c ON c.id = v.rowid \
         JOIN files f ON f.id = c.file_id WHERE f.path LIKE ?1",
        [format!("%{name}")],
        |r| r.get(0),
    )
    .unwrap()
}

fn load_cfg() -> ferret::config::Config {
    ferret::config::load(&ferret::config::LoadOpts::default()).unwrap()
}

fn fake_for(cfg: &ferret::config::Config) -> ferret::embedder::FakeEmbedder {
    ferret::embedder::FakeEmbedder::new_with_profile(
        cfg.embedder.prefix_profile,
        cfg.embedder.dimensions,
    )
}

#[test]
#[serial]
fn multimodal_false_records_skip_row_not_error() {
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    std::fs::write(env.root.join("pic.png"), png_bytes()).unwrap();
    // The fake always reports images_supported=true, so the false comes
    // from a real HttpEmbedder with multimodal=false (no I/O: the
    // constructor is pure and the override short-circuits the probe).
    let _multimodal = EnvGuard::set("FERRET__EMBEDDER__MULTIMODAL", "false");
    let cfg = load_cfg();
    let http = ferret::embedder::HttpEmbedder::new(&cfg).unwrap();
    assert!(!http.images_supported());
    let mut stdout: Vec<u8> = Vec::new();
    let summary = ferret::index::run(
        &cfg,
        &http,
        vec![],
        true,
        false,
        &mut stdout,
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(summary.errors, 0, "capability skip is not an error");
    assert_eq!(summary.skipped, 1);
    let events = String::from_utf8(stdout).unwrap();
    assert!(
        events.contains("\"reason\":\"no_image_backend\""),
        "expected a no_image_backend skip, got: {events}"
    );
    // Skip rows persist: complete=1, reason recorded, zero chunks, no error.
    let (complete, reason, error) = files_row(&env.index, "pic.png").expect("skip row must exist");
    assert_eq!(complete, 1);
    assert_eq!(reason.as_deref(), Some("no_image_backend"));
    assert_eq!(error, None);
    assert_eq!(count(&env.index, "chunks"), 0);
}

#[test]
#[serial]
fn flip_to_supported_reembeds_skipped_image() {
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    std::fs::write(env.root.join("pic.png"), png_bytes()).unwrap();
    let guard = EnvGuard::set("FERRET__EMBEDDER__MULTIMODAL", "false");
    let cfg = load_cfg();
    let http = ferret::embedder::HttpEmbedder::new(&cfg).unwrap();
    let mut stdout: Vec<u8> = Vec::new();
    ferret::index::run(
        &cfg,
        &http,
        vec![],
        true,
        false,
        &mut stdout,
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(count(&env.index, "chunks"), 0);
    drop(guard);
    // Probe now succeeds (fake): the image fingerprint changed with
    // images_supported, so the skip row re-extracts into real vecs.
    let cfg = load_cfg();
    let fake = fake_for(&cfg);
    let mut stdout: Vec<u8> = Vec::new();
    let summary = ferret::index::run(
        &cfg,
        &fake,
        vec![],
        true,
        false,
        &mut stdout,
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(summary.errors, 0);
    assert_eq!(count(&env.index, "chunks"), 1, "image must gain a chunk");
    assert_eq!(count(&env.index, "chunks_vec"), 1, "image must gain a vec");
    let (complete, reason, _) = files_row(&env.index, "pic.png").expect("row must exist");
    assert_eq!(complete, 1);
    assert_eq!(reason, None, "skip reason clears on success");
}

#[test]
#[serial]
fn text_pipeline_bump_reproduces_image_vecs() {
    let env: TestEnv = common::test_env();
    std::fs::write(env.root.join("pic.png"), png_bytes()).unwrap();
    let cfg = load_cfg();
    let mut stdout: Vec<u8> = Vec::new();
    ferret::index::run(
        &cfg,
        &fake_for(&cfg),
        vec![],
        false,
        false,
        &mut stdout,
        &mut Vec::new(),
    )
    .unwrap();
    let before = vec_blob_for(&env.index, "pic.png");
    assert!(!before.is_empty());
    let pets_fp_before = files_fingerprint(&env.index, "pets.md");
    // Text-only change (overlap must stay below max_chars).
    let cfg_text = std::fs::read_to_string(&env.cfg).unwrap();
    std::fs::write(
        &env.cfg,
        format!("{cfg_text}\n[chunk.text]\nmax_chars = 12\noverlap = 5\n"),
    )
    .unwrap();
    let cfg = load_cfg();
    let mut stdout: Vec<u8> = Vec::new();
    ferret::index::run(
        &cfg,
        &fake_for(&cfg),
        vec![],
        false,
        false,
        &mut stdout,
        &mut Vec::new(),
    )
    .unwrap();
    let after = vec_blob_for(&env.index, "pic.png");
    // End-state keep (§5.2 singular file fingerprint): image vecs are
    // re-derived byte-identical while text re-embeds. True avoided-work
    // keep needs per-pipeline file fingerprints (follow-up, needs a spec
    // Appendix A amendment).
    assert_eq!(before, after, "image vec bytes must survive a text bump");
    // ...while the text file really did re-embed (new fingerprint).
    let pets_fp_after = files_fingerprint(&env.index, "pets.md");
    assert_ne!(
        pets_fp_before, pets_fp_after,
        "text fingerprint must change on a text bump"
    );
}

#[test]
#[serial]
fn heic_magick_without_binary_is_walk_skip() {
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    // Empty file: the no_magick gate fires in the walk, so no real
    // HEIC bytes are needed (nor any read of the file).
    std::fs::write(env.root.join("pic.heic"), b"").unwrap();
    let _heic = EnvGuard::set("FERRET__FILES__HEIC", "magick");
    // Deterministic regardless of host tools: hide every binary from PATH.
    let empty = env.tmp.path().join("empty-path");
    std::fs::create_dir_all(&empty).unwrap();
    let _path = EnvGuard::set("PATH", empty.to_str().unwrap());
    let cfg = load_cfg();
    assert_eq!(cfg.files.heic.to_string(), "magick");
    let mut stdout: Vec<u8> = Vec::new();
    let summary = ferret::index::run(
        &cfg,
        &fake_for(&cfg),
        vec![],
        true,
        false,
        &mut stdout,
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(
        summary.errors, 0,
        "missing converter is a skip, not an error"
    );
    let events = String::from_utf8(stdout).unwrap();
    assert!(
        events.contains("\"reason\":\"no_magick\""),
        "expected a no_magick skip, got: {events}"
    );
    // Walk-skips leave no row (spec §4.5); a later run with a converter
    // re-walks the file into the image path.
    assert!(files_row(&env.index, "pic.heic").is_none());
}
