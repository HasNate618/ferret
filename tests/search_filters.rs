//! Task 8: SQL filters applied in SQL before ranking (spec §6.3).

mod common;

use ferret::config::{compiled_defaults, Config};
use ferret::embedder::{EmbedInput, Embedder, FakeEmbedder};
use ferret::search::{parse_query, run_search, FilterFlags, SearchRequest};
use ferret::store::{self, NewChunk, NewFile, Store};
use ferret::types::{ChunkKind, FileKind, PrefixProfile, SearchMode};
use std::cell::RefCell;
use std::path::PathBuf;

thread_local! {
    static KEEP: RefCell<Vec<tempfile::TempDir>> = const { RefCell::new(Vec::new()) };
}

const DIMS: u32 = 8;
const MTIME: i64 = 1717200000;

fn cfg() -> Config {
    let mut cfg = compiled_defaults(&common::test_dirs(std::path::Path::new("/ferret-t8-filters")));
    cfg.embedder.dimensions = DIMS;
    cfg
}

struct Harness {
    store: Store,
    cfg: Config,
    fake: FakeEmbedder,
}

/// Insert one file + one chunk (+ FTS row, + vec row from the fake embedder)
/// into a fresh temp store.
fn add_file(
    h: &Harness,
    path: &str,
    kind: FileKind,
    chunk_kind: ChunkKind,
    text: Option<&str>,
    image: Option<&[u8]>,
) {
    let ext = path.rsplit('.').next().unwrap_or("");
    let tx = h.store.begin().unwrap();
    let file_id = h
        .store
        .upsert_file(&NewFile {
            path: path.into(),
            collection: "c".into(),
            kind,
            ext: ext.into(),
            size: 1,
            mtime: MTIME,
            fingerprint: "fp".into(),
            complete: true,
            next_page: None,
            skip_reason: None,
            indexed_at: 1,
            error: None,
        })
        .unwrap();
    let chunk_id = h
        .store
        .insert_chunk(
            &tx,
            NewChunk {
                file_id,
                ordinal: 0,
                page: None,
                kind: chunk_kind,
                text: text.map(str::to_string),
                start_byte: None,
                end_byte: None,
            },
        )
        .unwrap();
    let input = match image {
        Some(bytes) => EmbedInput::Image {
            bytes: bytes.to_vec(),
            mime: "image/png".into(),
        },
        None => EmbedInput::Document {
            title: None,
            text: text.unwrap_or("").to_string(),
        },
    };
    let embedding = h.fake.embed(input).unwrap();
    h.store
        .insert_vec(&tx, chunk_id, &embedding, kind, ext, MTIME, "c")
        .unwrap();
    // The chunks_fts row is written by the `chunks_fts_ai` trigger (the
    // production path); a manual insert would now collide on rowid.
    tx.commit().unwrap();
}

fn harness() -> Harness {
    let cfg = cfg();
    let dir = tempfile::tempdir().unwrap();
    let store = store::open(&dir.path().join("index.sqlite"), &cfg, true).unwrap();
    let fake = FakeEmbedder::new_with_profile(PrefixProfile::EmbeddingGemma2, DIMS);
    KEEP.with(|k| k.borrow_mut().push(dir));
    Harness { store, cfg, fake }
}

fn search(h: &Harness, query: &str, flags: &FilterFlags) -> Vec<String> {
    let parsed = parse_query(query, flags).unwrap();
    let req = SearchRequest {
        parsed,
        mode: SearchMode::Hybrid,
        limit: 20,
        min_score: 0.0,
        json: false,
    };
    run_search(&h.store, &h.fake, &h.cfg, req)
        .unwrap()
        .hits
        .into_iter()
        .map(|hit| hit.path)
        .collect()
}

#[test]
fn kind_filter_uses_file_kind_not_chunk_kind() {
    let h = harness();
    add_file(&h, "/x/doc.txt", FileKind::Text, ChunkKind::Text, Some("pets text"), None);
    add_file(&h, "/x/pic.png", FileKind::Image, ChunkKind::Image, None, Some(b"pngbytes"));
    let flags = FilterFlags {
        kind: vec![FileKind::Text],
        ..Default::default()
    };
    let hits = search(&h, "pets text", &flags);
    assert_eq!(hits, vec!["/x/doc.txt".to_string()]);
}

#[test]
fn kind_pdf_prefix_matches_pdf_file_kind() {
    let h = harness();
    add_file(&h, "/x/lease.pdf", FileKind::Pdf, ChunkKind::PdfPage, Some("pets clause"), None);
    let p = parse_query("kind:pdf pets", &FilterFlags::default()).unwrap();
    let req = SearchRequest {
        parsed: p,
        mode: SearchMode::Hybrid,
        limit: 20,
        min_score: 0.0,
        json: false,
    };
    let hits: Vec<String> = run_search(&h.store, &h.fake, &h.cfg, req)
        .unwrap()
        .hits
        .into_iter()
        .map(|hit| hit.path)
        .collect();
    assert_eq!(hits, vec!["/x/lease.pdf".to_string()]);
}

#[test]
fn under_filter_restricts_to_dir() {
    let root = tempfile::tempdir().unwrap();
    let a = root.path().join("a");
    let b = root.path().join("b");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    let h = harness();
    add_file(&h, a.join("in.md").to_str().unwrap(), FileKind::Text, ChunkKind::Text, Some("pets here"), None);
    add_file(&h, b.join("out.md").to_str().unwrap(), FileKind::Text, ChunkKind::Text, Some("pets there"), None);
    let flags = FilterFlags {
        under: Some(PathBuf::from(a.to_str().unwrap())),
        ..Default::default()
    };
    let hits = search(&h, "pets", &flags);
    assert_eq!(hits, vec![a.join("in.md").to_str().unwrap().to_string()]);
}

#[test]
fn name_filter_restricts_to_basename_glob() {
    let h = harness();
    add_file(&h, "/x/lease.md", FileKind::Text, ChunkKind::Text, Some("pets"), None);
    add_file(&h, "/x/other.txt", FileKind::Text, ChunkKind::Text, Some("pets"), None);
    let flags = FilterFlags {
        name: Some("*.md".into()),
        ..Default::default()
    };
    let hits = search(&h, "pets", &flags);
    assert_eq!(hits, vec!["/x/lease.md".to_string()]);
}

#[test]
fn ext_filter_restricts_by_extension() {
    let h = harness();
    add_file(&h, "/x/lease.md", FileKind::Text, ChunkKind::Text, Some("pets"), None);
    add_file(&h, "/x/notes.txt", FileKind::Text, ChunkKind::Text, Some("pets"), None);
    let flags = FilterFlags {
        ext: vec!["md".into()],
        ..Default::default()
    };
    let hits = search(&h, "pets", &flags);
    assert_eq!(hits, vec!["/x/lease.md".to_string()]);
}
