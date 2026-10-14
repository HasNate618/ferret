//! Task 15: PDF indexing — per-page chunks, resume cursor, backfill,
//! missing-poppler error. Poppler-gated like the Task 14 magick tests.

mod common;

use common::{EnvGuard, TestEnv};
use serial_test::serial;

fn host_has_poppler() -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| {
            ["pdfinfo", "pdftotext", "pdftoppm"]
                .iter()
                .all(|bin| dir.join(bin).is_file())
        })
    })
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

fn run_json(cfg: &ferret::config::Config) -> (ferret::index::IndexSummary, String, String) {
    let mut stdout: Vec<u8> = Vec::new();
    let mut stderr: Vec<u8> = Vec::new();
    let summary = ferret::index::run(
        cfg,
        &fake_for(cfg),
        vec![],
        true,
        false,
        false,
        &mut stdout,
        &mut stderr,
    )
    .unwrap();
    (
        summary,
        String::from_utf8(stdout).unwrap(),
        String::from_utf8(stderr).unwrap(),
    )
}

/// (complete, next_page, error) for the file whose path ends in `name`.
fn pdf_row(index: &std::path::Path, name: &str) -> Option<(i64, Option<i64>, Option<String>)> {
    let conn = rusqlite::Connection::open(index).unwrap();
    let mut stmt = conn
        .prepare("SELECT complete, next_page, error FROM files WHERE path LIKE ?1")
        .unwrap();
    let mut rows = stmt.query([format!("%{name}")]).unwrap();
    rows.next()
        .unwrap()
        .map(|r| (r.get(0).unwrap(), r.get(1).unwrap(), r.get(2).unwrap()))
}

/// Sorted page numbers present in chunks for the file.
fn chunk_pages(index: &std::path::Path, name: &str) -> Vec<i64> {
    let conn = rusqlite::Connection::open(index).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT c.page FROM chunks c JOIN files f ON f.id = c.file_id \
             WHERE f.path LIKE ?1 ORDER BY c.page",
        )
        .unwrap();
    stmt.query_map([format!("%{name}")], |r| r.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn fixture_pdf(root: &std::path::Path) {
    std::fs::copy("testdata/two_pages.pdf", root.join("doc.pdf")).unwrap();
}

#[test]
#[serial]
fn two_pages_yield_two_chunks_with_pages_and_ordinals() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    fixture_pdf(&env.root);
    let cfg = load_cfg();
    let (summary, _, _) = run_json(&cfg);
    assert_eq!(summary.errors, 0);
    assert_eq!(chunk_pages(&env.index, "doc.pdf"), vec![1, 2]);
    let conn = rusqlite::Connection::open(&env.index).unwrap();
    let kinds: Vec<String> = {
        let mut stmt = conn
            .prepare(
                "SELECT c.kind FROM chunks c JOIN files f ON f.id = c.file_id \
                 WHERE f.path LIKE '%doc.pdf' ORDER BY c.ordinal",
            )
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    };
    assert_eq!(kinds, vec!["pdf_page".to_string(), "pdf_page".to_string()]);
    let ordinals: Vec<i64> = {
        let mut stmt = conn
            .prepare(
                "SELECT c.ordinal FROM chunks c JOIN files f ON f.id = c.file_id \
                 WHERE f.path LIKE '%doc.pdf' ORDER BY c.ordinal",
            )
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    };
    assert_eq!(ordinals, vec![0, 1]);
    let (complete, next_page, _) = pdf_row(&env.index, "doc.pdf").expect("pdf row must exist");
    assert_eq!(complete, 1);
    assert!(
        next_page.unwrap_or(0) > 2,
        "cursor must advance past page 2"
    );
    // Page text is stored for FTS.
    let text: String = conn
        .query_row(
            "SELECT c.text FROM chunks c JOIN files f ON f.id = c.file_id \
             WHERE f.path LIKE '%doc.pdf' AND c.page = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        text.contains("page one pets"),
        "unexpected stored text: {text:?}"
    );
}

#[test]
#[serial]
fn max_pages_caps_file_and_warns() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    fixture_pdf(&env.root);
    let _guard = EnvGuard::set("FERRET__CHUNK__PDF__MAX_PAGES", "1");
    let cfg = load_cfg();
    let (summary, _, stderr) = run_json(&cfg);
    assert_eq!(summary.errors, 0);
    // Capped: only page 1, cursor at 2 — and per the completion formula
    // (next_page > min(max_pages, n)) the file counts as complete.
    assert_eq!(chunk_pages(&env.index, "doc.pdf"), vec![1]);
    let (complete, next_page, _) = pdf_row(&env.index, "doc.pdf").expect("row");
    assert_eq!(complete, 1);
    assert_eq!(next_page, Some(2));
    assert!(
        stderr.contains("max_pages"),
        "capped file must warn once, got: {stderr:?}"
    );
}

#[test]
#[serial]
fn resume_after_page_failure_embeds_page_two_next() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    fixture_pdf(&env.root);
    let cfg = load_cfg();
    let _ = run_json(&cfg);
    assert_eq!(chunk_pages(&env.index, "doc.pdf"), vec![1, 2]);
    // Simulate a page-2 failure after page 1 committed: drop page 2's
    // chunk, rewind the cursor, record the error.
    let conn = rusqlite::Connection::open(&env.index).unwrap();
    conn.execute(
        "DELETE FROM chunks WHERE page = 2 AND file_id = (SELECT id FROM files WHERE path LIKE '%doc.pdf')",
        [],
    )
    .unwrap();
    conn.execute(
        "UPDATE files SET complete = 0, next_page = 2, error = 'boom page 2' WHERE path LIKE '%doc.pdf'",
        [],
    )
    .unwrap();
    drop(conn);
    // Resume: page 2 embeds next (never jumped over); page 1 is kept
    // via its vector, not duplicated; the error heals.
    let cfg = load_cfg();
    let (summary, _, _) = run_json(&cfg);
    assert_eq!(summary.errors, 0);
    assert_eq!(chunk_pages(&env.index, "doc.pdf"), vec![1, 2]);
    let (complete, next_page, error) = pdf_row(&env.index, "doc.pdf").expect("row");
    assert_eq!(complete, 1);
    assert!(next_page.unwrap_or(0) > 2);
    assert_eq!(error, None);
}

#[test]
#[serial]
fn missing_page_below_cursor_is_backfilled() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    fixture_pdf(&env.root);
    let cfg = load_cfg();
    let _ = run_json(&cfg);
    assert_eq!(chunk_pages(&env.index, "doc.pdf"), vec![1, 2]);
    // Corrupt the state: drop page 1's chunk, rewind the cursor past it.
    let conn = rusqlite::Connection::open(&env.index).unwrap();
    conn.execute(
        "DELETE FROM chunks WHERE page = 1 AND file_id = (SELECT id FROM files WHERE path LIKE '%doc.pdf')",
        [],
    )
    .unwrap();
    conn.execute(
        "UPDATE files SET next_page = 2, complete = 0 WHERE path LIKE '%doc.pdf'",
        [],
    )
    .unwrap();
    drop(conn);
    let cfg = load_cfg();
    let (summary, _, _) = run_json(&cfg);
    assert_eq!(summary.errors, 0);
    // Page 1 backfilled first; page 2 kept (not duplicated, not skipped).
    assert_eq!(chunk_pages(&env.index, "doc.pdf"), vec![1, 2]);
    let (complete, _, _) = pdf_row(&env.index, "doc.pdf").expect("row");
    assert_eq!(complete, 1);
}

#[test]
#[serial]
fn missing_poppler_is_extract_error() {
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    fixture_pdf(&env.root);
    // Bogus explicit binaries: deterministic without uninstalling poppler.
    let _a = EnvGuard::set("FERRET__EXTRACT__PDFINFO", "/no/such/pdfinfo");
    let _b = EnvGuard::set("FERRET__EXTRACT__PDFTOTEXT", "/no/such/pdftotext");
    let _c = EnvGuard::set("FERRET__EXTRACT__PDFTOPPM", "/no/such/pdftoppm");
    let cfg = load_cfg();
    let (summary, _, _) = run_json(&cfg);
    assert_eq!(summary.errors, 1);
    let (complete, _, error) = pdf_row(&env.index, "doc.pdf").expect("row");
    assert_eq!(complete, 0);
    let error = error.expect("error must be recorded");
    assert!(
        error.contains("pdfinfo"),
        "resolve order is pdfinfo first: {error}"
    );
    assert!(error.contains("not found"), "unexpected error: {error}");
}

/// Stub embedder failing exactly one page text (C1 regression: committed
/// pages must survive a later page's failure).
struct FailPageTwo {
    inner: ferret::embedder::FakeEmbedder,
}

impl ferret::embedder::Embedder for FailPageTwo {
    fn embed(&self, input: ferret::embedder::EmbedInput) -> ferret::error::Result<Vec<f32>> {
        let haystack = match &input {
            ferret::embedder::EmbedInput::Document { text, .. } => text.clone(),
            ferret::embedder::EmbedInput::PdfPage { text, .. } => text.clone().unwrap_or_default(),
            _ => String::new(),
        };
        if haystack.contains("two ferrets") {
            return Err(ferret::error::Error::file("boom page 2"));
        }
        self.inner.embed(input)
    }
    fn images_supported(&self) -> bool {
        true
    }
}

/// Recorder capturing which input variant each page embeds as.
struct RecordInputs {
    inner: ferret::embedder::FakeEmbedder,
    seen: std::sync::Mutex<Vec<String>>,
}

impl RecordInputs {
    fn kinds(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }
}

impl ferret::embedder::Embedder for RecordInputs {
    fn embed(&self, input: ferret::embedder::EmbedInput) -> ferret::error::Result<Vec<f32>> {
        let kind = match &input {
            ferret::embedder::EmbedInput::PdfPage { .. } => "Pdf",
            ferret::embedder::EmbedInput::Image { .. } => "Image",
            ferret::embedder::EmbedInput::Document { .. } => "Doc",
            ferret::embedder::EmbedInput::Query { .. } => "Query",
            ferret::embedder::EmbedInput::Audio { .. } => "Audio",
            ferret::embedder::EmbedInput::Video { .. } => "Video",
        };
        self.seen.lock().unwrap().push(kind.to_string());
        self.inner.embed(input)
    }
    fn images_supported(&self) -> bool {
        true
    }
}

fn chunk_ids(index: &std::path::Path, name: &str) -> Vec<i64> {
    let conn = rusqlite::Connection::open(index).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id \
             WHERE f.path LIKE ?1 ORDER BY c.page",
        )
        .unwrap();
    stmt.query_map([format!("%{name}")], |r| r.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn pdf_vec_blobs(index: &std::path::Path, name: &str) -> Vec<Vec<u8>> {
    let conn = rusqlite::Connection::open(index).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT v.embedding FROM chunks_vec v JOIN chunks c ON c.id = v.rowid \
             JOIN files f ON f.id = c.file_id WHERE f.path LIKE ?1 ORDER BY c.page",
        )
        .unwrap();
    stmt.query_map([format!("%{name}")], |r| r.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

#[test]
#[serial]
fn failed_page_two_keeps_committed_page_one() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    fixture_pdf(&env.root);
    let cfg = load_cfg();
    let failer = FailPageTwo {
        inner: fake_for(&cfg),
    };
    let mut stdout: Vec<u8> = Vec::new();
    let mut stderr: Vec<u8> = Vec::new();
    let summary = ferret::index::run(
        &cfg,
        &failer,
        vec![],
        true,
        false,
        false,
        &mut stdout,
        &mut stderr,
    )
    .unwrap();
    assert_eq!(summary.errors, 1);
    // Page 1's chunk survives the page-2 failure (spec §9 partial file).
    assert_eq!(chunk_pages(&env.index, "doc.pdf"), vec![1]);
    let (complete, next_page, error) = pdf_row(&env.index, "doc.pdf").expect("row");
    assert_eq!(complete, 0);
    assert_eq!(next_page, Some(2));
    assert!(error.unwrap().contains("boom page 2"));
    // And the next run heals it (page 2 embeds, error clears).
    let cfg = load_cfg();
    let (summary, _, _) = run_json(&cfg);
    assert_eq!(summary.errors, 0);
    assert_eq!(chunk_pages(&env.index, "doc.pdf"), vec![1, 2]);
    let (complete, _, error) = pdf_row(&env.index, "doc.pdf").expect("row");
    assert_eq!(complete, 1);
    assert_eq!(error, None);
}

#[test]
#[serial]
fn embed_text_false_embeds_pages_as_bare_images() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    fixture_pdf(&env.root);
    let _flag = EnvGuard::set("FERRET__CHUNK__PDF__EMBED_TEXT", "false");
    let cfg = load_cfg();
    assert!(!cfg.chunk.pdf.embed_text);
    let recorder = RecordInputs {
        inner: fake_for(&cfg),
        seen: std::sync::Mutex::new(Vec::new()),
    };
    let mut stdout: Vec<u8> = Vec::new();
    let mut stderr: Vec<u8> = Vec::new();
    let summary = ferret::index::run(
        &cfg,
        &recorder,
        vec![],
        true,
        false,
        false,
        &mut stdout,
        &mut stderr,
    )
    .unwrap();
    assert_eq!(summary.errors, 0);
    // Spec §4.4(5): bare image per page, no document prefix.
    assert_eq!(
        recorder.kinds(),
        vec!["Image".to_string(), "Image".to_string()]
    );
    // ...while pdftotext is still stored for FTS.
    let conn = rusqlite::Connection::open(&env.index).unwrap();
    let text: String = conn
        .query_row(
            "SELECT c.text FROM chunks c JOIN files f ON f.id = c.file_id \
             WHERE f.path LIKE '%doc.pdf' AND c.page = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        text.contains("page one pets"),
        "unexpected stored text: {text:?}"
    );
}

#[test]
#[serial]
fn image_bump_reembeds_pdf_pages_keeping_chunks() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    fixture_pdf(&env.root);
    let cfg = load_cfg();
    let _ = run_json(&cfg);
    let ids_before = chunk_ids(&env.index, "doc.pdf");
    assert_eq!(ids_before.len(), 2);
    // Image-only change: chunk rows stay (same ids), vectors are redone.
    let _edge = EnvGuard::set("FERRET__CHUNK__IMAGE__MAX_EDGE_PX", "512");
    let cfg = load_cfg();
    let (summary, _, _) = run_json(&cfg);
    assert_eq!(summary.errors, 0);
    assert_eq!(chunk_ids(&env.index, "doc.pdf"), ids_before);
    assert_eq!(pdf_vec_blobs(&env.index, "doc.pdf").len(), 2);
    let (complete, next_page, _) = pdf_row(&env.index, "doc.pdf").expect("row");
    assert_eq!(complete, 1);
    assert!(next_page.unwrap_or(0) > 2);
}

#[test]
#[serial]
fn text_bump_keeps_pdf_vecs() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    fixture_pdf(&env.root);
    let cfg = load_cfg();
    let _ = run_json(&cfg);
    let vecs_before = pdf_vec_blobs(&env.index, "doc.pdf");
    assert_eq!(vecs_before.len(), 2);
    // Text-only change: vectors byte-identical, FTS text still present.
    let cfg_text = std::fs::read_to_string(&env.cfg).unwrap();
    std::fs::write(
        &env.cfg,
        format!("{cfg_text}\n[chunk.text]\nmax_chars = 12\noverlap = 5\n"),
    )
    .unwrap();
    let cfg = load_cfg();
    let (summary, _, _) = run_json(&cfg);
    assert_eq!(summary.errors, 0);
    assert_eq!(pdf_vec_blobs(&env.index, "doc.pdf"), vecs_before);
}

#[test]
#[serial]
fn three_pages_converge_through_small_first_pass() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let env: TestEnv = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    std::fs::copy("testdata/three_pages.pdf", env.root.join("long.pdf")).unwrap();
    let _first = EnvGuard::set("FERRET__CHUNK__PDF__FIRST_PASS_PAGES", "1");
    let cfg = load_cfg();
    let (summary, _, _) = run_json(&cfg);
    assert_eq!(summary.errors, 0);
    assert_eq!(chunk_pages(&env.index, "long.pdf"), vec![1, 2, 3]);
    let (complete, next_page, _) = pdf_row(&env.index, "long.pdf").expect("row");
    assert_eq!(complete, 1);
    assert!(next_page.unwrap_or(0) > 3);
}
