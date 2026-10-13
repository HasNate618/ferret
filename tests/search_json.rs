//! Task 8: frozen search JSON contract (spec §6.4).

mod common;

use ferret::config::{Config, compiled_defaults};
use ferret::embedder::{EmbedInput, Embedder, FakeEmbedder};
use ferret::search::{FilterFlags, SearchRequest, parse_query, render_json, run_search};
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
                t_start_ms: None,
                t_end_ms: None,
            },
        )
        .unwrap();
    store
        .insert_vec(
            &tx,
            chunk_id,
            &embedding,
            FileKind::Text,
            "md",
            MTIME,
            "Documents",
        )
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
    for k in [
        "query",
        "task",
        "mode",
        "limit",
        "min_score",
        "filters",
        "hits",
    ] {
        assert!(v.get(k).is_some(), "{k}");
    }
    let f = &v["filters"];
    for k in [
        "kind",
        "ext",
        "under",
        "collection",
        "name",
        "after",
        "before",
    ] {
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

#[test]
#[serial_test::serial]
fn search_json_warnings_empty_when_unscoped() {
    let _env = common::test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out = ferret::cli::try_run(&["ferret", "search", "--json", "pets"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["warnings"], serde_json::json!([]));
}

#[test]
#[serial_test::serial]
fn search_json_no_warning_when_under_matches_nothing() {
    let env = common::test_env();
    let empty = env.root.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out = ferret::cli::try_run(&[
        "ferret",
        "search",
        "--json",
        "pets",
        "--under",
        empty.to_str().unwrap(),
    ])
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(v["hits"].as_array().unwrap().is_empty());
    // Empty hits => no vec_starved (rule needs >= 1 hit); warnings present but empty.
    assert_eq!(v["warnings"], serde_json::json!([]));
}

#[test]
#[serial_test::serial]
fn search_json_vec_starved_when_only_fts_hits() {
    let _env = common::test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out = ferret::cli::try_run(&[
        "ferret", "search", "--json", "--mode", "keyword", "--name", "*.md", "pets",
    ])
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let hits = v["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h["scores"]["vec"].is_null()));
    assert!(
        v["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w == "vec_starved")
    );
}

#[test]
#[serial_test::serial]
fn search_json_no_vec_starved_when_hybrid_vec_contributes() {
    // The rule's third conjunct: scoped + hits + any vec score => no warning.
    // (Positive hybrid starvation needs a real embedder + large index — with
    // the fake the vec arm always contributes in scope; the fires-case is
    // covered deterministically by ..._vec_starved_when_only_fts_hits.)
    let _env = common::test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out =
        ferret::cli::try_run(&["ferret", "search", "--json", "--name", "*.md", "pets"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let hits = v["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().any(|h| !h["scores"]["vec"].is_null()));
    assert_eq!(v["warnings"], serde_json::json!([]));
}

#[test]
#[serial_test::serial]
fn search_json_browse_labels_no_vec() {
    let _env = common::test_env();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out = ferret::cli::try_run(&["ferret", "search", "--json", "--kind", "text"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(
        v["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w == "browse_no_vec")
    );
}

#[test]
#[serial_test::serial]
fn search_json_snippet_spans_highlight_match() {
    let env = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    std::fs::write(env.root.join("fox.txt"), "the quick brown fox jumps").unwrap();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out = ferret::cli::try_run(&["ferret", "search", "--json", "quick"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let hit = &v["hits"].as_array().unwrap()[0];
    let snippet = hit["snippet"].as_str().unwrap();
    let spans = hit["snippet_spans"].as_array().unwrap();
    assert!(!spans.is_empty());
    assert!(spans.len() <= 5);
    for s in spans {
        let (start, end) = (
            s["start"].as_u64().unwrap() as usize,
            s["end"].as_u64().unwrap() as usize,
        );
        let chars: Vec<char> = snippet.chars().collect();
        assert!(start < end && end <= chars.len());
    }
    // At least one span covers the matched term.
    let chars: Vec<char> = snippet.chars().collect();
    assert!(spans.iter().any(|s| {
        let (a, b) = (
            s["start"].as_u64().unwrap() as usize,
            s["end"].as_u64().unwrap() as usize,
        );
        chars[a..b].iter().collect::<String>().to_lowercase() == "quick"
    }));
}

#[test]
#[serial_test::serial]
fn search_json_spans_empty_for_image_hits() {
    let env = common::test_env();
    std::fs::remove_file(env.root.join("pets.md")).unwrap();
    std::fs::write(
        env.root.join("pic.png"),
        std::fs::read("testdata/one.png").unwrap(),
    )
    .unwrap();
    ferret::cli::try_run(&["ferret", "index"]).unwrap();
    let out =
        ferret::cli::try_run(&["ferret", "search", "--json", "--kind", "image", "photo"]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let hits = v["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    for h in hits {
        assert_eq!(h["snippet_spans"], serde_json::json!([]));
    }
}

/// A temp store holding one audio file with one 30s segment chunk + vec
/// row (FTS row lands via trigger with NULL text, like images).
fn audio_fixture() -> (Store, Config, FakeEmbedder) {
    let dims = 8u32;
    let mut cfg = compiled_defaults(&common::test_dirs(std::path::Path::new("/ferret-t8-audio")));
    cfg.embedder.dimensions = dims;
    let dir = tempfile::tempdir().unwrap();
    let store = store::open(&dir.path().join("index.sqlite"), &cfg, true).unwrap();
    let fake = FakeEmbedder::new_with_profile(PrefixProfile::EmbeddingGemma2, dims);
    let embedding = fake
        .embed(EmbedInput::Audio {
            wav_bytes: vec![7; 100],
        })
        .unwrap();
    let tx = store.begin().unwrap();
    let file_id = store
        .upsert_file(&NewFile {
            path: "/home/nate/Music/sweep.wav".into(),
            collection: "Documents".into(),
            kind: FileKind::Audio,
            ext: "wav".into(),
            size: 100,
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
                kind: ChunkKind::AudioSegment,
                text: None,
                start_byte: None,
                end_byte: None,
                t_start_ms: Some(0),
                t_end_ms: Some(30000),
            },
        )
        .unwrap();
    store
        .insert_vec(&tx, chunk_id, &embedding, FileKind::Audio, "wav", MTIME, "Documents")
        .unwrap();
    // Second segment with no end timestamp: renders `[mm:ss–?]`.
    let chunk2 = store
        .insert_chunk(
            &tx,
            NewChunk {
                file_id,
                ordinal: 1,
                page: None,
                kind: ChunkKind::AudioSegment,
                text: None,
                start_byte: None,
                end_byte: None,
                t_start_ms: Some(30000),
                t_end_ms: None,
            },
        )
        .unwrap();
    store
        .insert_vec(&tx, chunk2, &embedding, FileKind::Audio, "wav", MTIME, "Documents")
        .unwrap();
    tx.commit().unwrap();
    KEEP.with(|k| k.borrow_mut().push(dir));
    (store, cfg, fake)
}

#[test]
fn search_returns_timestamped_audio_hits() {
    let (store, cfg, fake) = audio_fixture();
    let parsed = parse_query("sine tone", &FilterFlags::default()).unwrap();
    let req = SearchRequest {
        parsed,
        mode: SearchMode::Hybrid,
        limit: 20,
        min_score: 0.0,
        json: true,
    };
    let out = run_search(&store, &fake, &cfg, req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&render_json(&out)).unwrap();
    let hits = v["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits
            .iter()
            .all(|h| h["chunk_kind"] == "audio_segment")
    );
    assert!(
        hits
            .iter()
            .all(|h| h["page"].is_null() && h["snippet_spans"] == serde_json::json!([]))
    );
    assert!(hits.iter().all(|h| h["scores"]["fts"].is_null()));
    let snips: Vec<&str> = hits
        .iter()
        .map(|h| h["snippet"].as_str().unwrap())
        .collect();
    assert!(snips.contains(&"[00:00–00:30] sweep.wav"), "got {snips:?}");
    assert!(snips.contains(&"[00:30–?] sweep.wav"), "got {snips:?}");
}
