//! Task 8: frozen search JSON contract (spec §6.4).

mod common;

use ferret::config::{compiled_defaults, Config};
use ferret::embedder::{EmbedInput, Embedder, FakeEmbedder};
use ferret::search::{parse_query, render_json, run_search, FilterFlags, SearchRequest};
use ferret::store::{self, NewChunk, NewFile, Store};
use ferret::types::{ChunkKind, FileKind, PrefixProfile, SearchMode};
use std::cell::RefCell;

thread_local! {
    /// Keeps the per-test tempdirs (and their index files) alive until the
    /// test thread ends; the store only borrows the paths.
    static KEEP: RefCell<Vec<tempfile::TempDir>> = const { RefCell::new(Vec::new()) };
}

const DOC_PATH: &str = "/home/nate/Documents/lease.md";
const DOC_TEXT: &str = "pets in the lease are not permitted without written consent";
const MTIME: i64 = 1717200000;

/// A temp store holding one text file with one text chunk, FTS row, and vec
/// row — the shape `ferret index` produces (FTS triggers land in Task 11, so
/// tests insert the `chunks_fts` row directly).
fn fixture() -> (Store, Config, FakeEmbedder) {
    let dims = 8u32;
    let mut cfg = compiled_defaults(&common::test_dirs(std::path::Path::new("/ferret-t8-json")));
    cfg.embedder.dimensions = dims;
    let dir = tempfile::tempdir().unwrap();
    let store = store::open(&dir.path().join("index.sqlite"), &cfg, true).unwrap();
    let fake = FakeEmbedder::new_with_profile(PrefixProfile::EmbeddingGemma2, dims);
    let embedding = fake
        .embed(EmbedInput::Document {
            title: Some("lease.md".into()),
            text: DOC_TEXT.into(),
        })
        .unwrap();
    let tx = store.begin().unwrap();
    let file_id = store
        .upsert_file(&NewFile {
            path: DOC_PATH.into(),
            collection: "Documents".into(),
            kind: FileKind::Text,
            ext: "md".into(),
            size: DOC_TEXT.len() as u64,
            mtime: MTIME,
            fingerprint: "fp".into(),
            complete: true,
            next_page: None,
            skip_reason: None,
            indexed_at: 1,
            error: None,
        })
        .unwrap();
    let chunk_id = store
        .insert_chunk(
            &tx,
            NewChunk {
                file_id,
                ordinal: 0,
                page: None,
                kind: ChunkKind::Text,
                text: Some(DOC_TEXT.into()),
                start_byte: None,
                end_byte: None,
            },
        )
        .unwrap();
    store
        .insert_vec(&tx, chunk_id, &embedding, FileKind::Text, "md", MTIME, "Documents")
        .unwrap();
    // The chunks_fts row is written by the `chunks_fts_ai` trigger (the
    // production path); a manual insert would now collide on rowid.
    tx.commit().unwrap();
    KEEP.with(|k| k.borrow_mut().push(dir));
    (store, cfg, fake)
}

/// Seed the fixture, search `query` in hybrid mode, render the frozen JSON.
fn run_search_json_on_fixture(query: &str) -> String {
    let (store, cfg, fake) = fixture();
    let parsed = parse_query(query, &FilterFlags::default()).unwrap();
    let req = SearchRequest {
        parsed,
        mode: SearchMode::Hybrid,
        limit: 20,
        min_score: 0.0,
        json: true,
    };
    let out = run_search(&store, &fake, &cfg, req).unwrap();
    render_json(&out)
}

#[test]
fn json_object_keys_and_null_page() {
    let out = run_search_json_on_fixture("pets");
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    for k in ["query", "task", "mode", "limit", "min_score", "filters", "hits"] {
        assert!(v.get(k).is_some(), "{k}");
    }
    let f = &v["filters"];
    for k in ["kind", "ext", "under", "collection", "name", "after", "before"] {
        assert!(f.get(k).is_some(), "{k}");
    }
    let h = &v["hits"][0];
    assert!(h.get("start_byte").is_none());
    assert!(h.get("end_byte").is_none());
    assert!(h["scores"]["vec"].as_f64().unwrap() >= 0.0);
    assert!(h["scores"]["vec"].as_f64().unwrap() <= 1.0);
    assert_eq!(h["chunk_kind"], "text");
    assert!(h["page"].is_null());
}

#[test]
fn json_hit_fields_and_echoed_query() {
    let out = run_search_json_on_fixture("pets");
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["query"], "pets");
    assert_eq!(v["task"], "search");
    assert_eq!(v["mode"], "hybrid");
    assert_eq!(v["limit"], 20);
    assert_eq!(v["min_score"], 0.0);
    let h = &v["hits"][0];
    assert_eq!(h["path"], DOC_PATH);
    assert_eq!(h["kind"], "text");
    assert_eq!(h["collection"], "Documents");
    assert_eq!(h["ext"], "md");
    assert_eq!(h["ordinal"], 0);
    assert_eq!(h["mtime"], MTIME);
    assert!(h["score"].as_f64().unwrap() > 0.0);
    assert!(h["snippet"].as_str().unwrap().contains("pets"));
    assert!(h["snippet"].as_str().unwrap().chars().count() <= 200);
}

#[test]
fn json_zero_hits_is_empty_array() {
    // Hybrid mode still returns nearest vec neighbours for unknown words, so
    // a filtered-out corpus (only a text file exists, kind:pdf asked for) is
    // the honest zero-hit case.
    let out = run_search_json_on_fixture("kind:pdf pets");
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["hits"].as_array().unwrap().len(), 0);
}

#[test]
fn min_score_above_fused_rrf_drops_all_hits() {
    let (store, cfg, fake) = fixture();
    let parsed = parse_query("pets", &FilterFlags::default()).unwrap();
    let req = SearchRequest {
        parsed,
        mode: SearchMode::Hybrid,
        limit: 20,
        min_score: 0.99, // fused RRF is ~0.01–0.05
        json: true,
    };
    let out = run_search(&store, &fake, &cfg, req).unwrap();
    assert!(out.hits.is_empty());
    let v: serde_json::Value = serde_json::from_str(&render_json(&out)).unwrap();
    assert_eq!(v["hits"].as_array().unwrap().len(), 0);
    assert_eq!(v["min_score"], 0.99);
}

#[test]
fn keyword_mode_leaves_vec_score_null() {
    let (store, cfg, fake) = fixture();
    let parsed = parse_query("pets", &FilterFlags::default()).unwrap();
    let req = SearchRequest {
        parsed,
        mode: SearchMode::Keyword,
        limit: 20,
        min_score: 0.0,
        json: true,
    };
    let out = run_search(&store, &fake, &cfg, req).unwrap();
    assert_eq!(out.hits.len(), 1);
    let hit = &out.hits[0];
    assert!(hit.scores.vec.is_none());
    assert!(hit.scores.fts.is_some());
    assert!(hit.scores.filename.is_none());
    let v: serde_json::Value = serde_json::from_str(&render_json(&out)).unwrap();
    assert!(v["hits"][0]["scores"]["vec"].is_null());
}
