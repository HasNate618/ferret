//! Task 13: rm, gc, status, NDJSON polish, upload warning. All CLI
//! tests run through `try_run` with `FERRET_EMBEDDER=fake`.

mod common;

use common::{EnvGuard, TestEnv};
use ferret::embedder::{EmbedInput, Embedder, FakeEmbedder};
use serial_test::serial;

fn file_row(index: &std::path::Path, name: &str) -> Option<(i64, i64)> {
    // (complete, error-is-null) for the file whose path ends in `name`.
    let conn = rusqlite::Connection::open(index).unwrap();
    let mut stmt = conn
        .prepare("SELECT complete, error IS NULL FROM files WHERE path LIKE ?1")
        .unwrap();
    let mut rows = stmt.query([format!("%{name}")]).unwrap();
    rows.next()
        .unwrap()
        .map(|r| (r.get(0).unwrap(), r.get(1).unwrap()))
}

fn chunk_count(index: &std::path::Path) -> i64 {
    let conn = rusqlite::Connection::open(index).unwrap();
    conn.query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))
        .unwrap()
}

fn vec_count(index: &std::path::Path) -> i64 {
    let conn = rusqlite::Connection::open(index).unwrap();
    conn.query_row("SELECT count(*) FROM chunks_vec", [], |r| r.get(0))
        .unwrap()
}

#[test]
#[serial]
fn rm_missing_path_is_success() {
    let _env = common::test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out = ferret::cli::try_run(&["ferret", "rm", "/no/such/file.md"]).unwrap();
    assert!(
        out.stderr.contains("not in index") || out.stderr.to_lowercase().contains("missing"),
        "rm of a missing path must warn on stderr, got: {:?}",
        out.stderr
    );
}

#[test]
#[serial]
fn rm_present_path_deletes_row_chunks_vec_and_fts() {
    let env: TestEnv = common::test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    assert!(chunk_count(&env.index) > 0);
    assert!(vec_count(&env.index) > 0);
    let target = env.root.join("pets.md");
    ferret::cli::try_run(&["ferret", "rm", target.to_str().unwrap()]).unwrap();
    assert!(
        file_row(&env.index, "pets.md").is_none(),
        "file row must go"
    );
    assert_eq!(chunk_count(&env.index), 0, "chunks must cascade");
    assert_eq!(vec_count(&env.index), 0, "vec rows must not orphan");
    let conn = rusqlite::Connection::open(&env.index).unwrap();
    let fts: i64 = conn
        .query_row("SELECT count(*) FROM chunks_fts", [], |r| r.get(0))
        .unwrap();
    assert_eq!(fts, 0, "fts rows must not orphan");
}

#[test]
#[serial]
fn gc_deletes_gone_files() {
    let env: TestEnv = common::test_env();
    std::fs::write(env.root.join("gone.md"), "x").unwrap();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    assert!(file_row(&env.index, "gone.md").is_some());
    std::fs::remove_file(env.root.join("gone.md")).unwrap();
    let out = ferret::cli::try_run(&["ferret", "gc"]).unwrap();
    assert!(file_row(&env.index, "gone.md").is_none());
    assert!(
        file_row(&env.index, "pets.md").is_some(),
        "gc keeps live files"
    );
    assert_eq!(vec_count(&env.index), chunk_count(&env.index));
    let _ = out;
}

#[test]
#[serial]
fn status_json_shape_and_no_embed() {
    let _env = common::test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out = ferret::cli::try_run(&["ferret", "status", "--json"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    for k in [
        "index",
        "model_id",
        "dimensions",
        "backend",
        "images_supported",
        "files",
        "chunks",
        "incomplete",
        "errors",
        "indexed_at_max",
    ] {
        assert!(v.get(k).is_some(), "{k}");
    }
    assert!(v["files"].get("text").is_some());
    assert!(v["files"].get("image").is_some());
    assert!(v["files"].get("pdf").is_some());
    assert_eq!(v["files"]["text"], 1);
    assert_eq!(v["images_supported"], true);
}

#[test]
#[serial]
fn init_openai_does_not_warn_until_index() {
    let env: TestEnv = common::test_env();
    // Fresh init with default (local) backend: no upload warning.
    let out = ferret::cli::try_run(&[
        "ferret",
        "init",
        "--force",
        "--root",
        env.root.to_str().unwrap(),
    ])
    .unwrap();
    assert!(
        !out.stderr.to_lowercase().contains("upload"),
        "local init must not warn, got: {:?}",
        out.stderr
    );
    // `init --force` rewrites the user file, so the openai layer comes
    // from the environment (which outranks the file init just wrote).
    let _backend = EnvGuard::set("FERRET__EMBEDDER__BACKEND", "openai");
    let out = ferret::cli::try_run(&[
        "ferret",
        "init",
        "--force",
        "--root",
        env.root.to_str().unwrap(),
    ])
    .unwrap();
    assert!(
        out.stderr.to_lowercase().contains("upload"),
        "openai init must warn, got: {:?}",
        out.stderr
    );
    // ...and so does index (FERRET_EMBEDDER=fake still wins for the
    // actual embeddings, so index runs and warns).
    let out = ferret::cli::try_run(&["ferret", "index"]).unwrap();
    assert!(
        out.stderr.to_lowercase().contains("upload"),
        "openai index must warn, got: {:?}",
        out.stderr
    );
}

/// An embedder that refuses images: drives the `no_image_backend`
/// capability-skip path through the real `index::run`.
struct NoImages(FakeEmbedder);

impl Embedder for NoImages {
    fn embed(&self, input: EmbedInput) -> ferret::error::Result<Vec<f32>> {
        self.0.embed(input)
    }
    fn images_supported(&self) -> bool {
        false
    }
}

#[test]
#[serial]
fn image_file_without_image_backend_is_skip_not_error() {
    let env: TestEnv = common::test_env();
    // 1x1 PNG body; content does not matter (extract is stubbed in v1).
    std::fs::write(env.root.join("pic.png"), b"\x89PNG\r\n\x1a\n").unwrap();
    // Load the real merged config (XDG env vars already point at `env`).
    let loaded = ferret::config::load(&ferret::config::LoadOpts::default()).unwrap();
    let no_img = NoImages(FakeEmbedder::new_with_profile(
        loaded.embedder.prefix_profile,
        loaded.embedder.dimensions,
    ));
    // Run the real pipeline with the image-refusing embedder.
    let mut stdout: Vec<u8> = Vec::new();
    let summary = ferret::index::run(
        &loaded,
        &no_img,
        vec![],
        true,
        false,
        &mut stdout,
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(summary.errors, 0, "capability skip is not an error");
    let events = String::from_utf8(stdout).unwrap();
    assert!(
        events.contains("\"reason\":\"no_image_backend\""),
        "expected a no_image_backend skip, got: {events}"
    );
    // Skip rows persist (Task 14): complete=1, reason recorded, no error.
    let conn = rusqlite::Connection::open(&env.index).unwrap();
    let (complete, reason): (i64, Option<String>) = conn
        .query_row(
            "SELECT complete, skip_reason FROM files WHERE path LIKE '%pic.png'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(complete, 1);
    assert_eq!(reason.as_deref(), Some("no_image_backend"));
}
