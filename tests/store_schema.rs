//! Task 5: store schema, meta validation, and CRUD.

mod common;

use ferret::config::{Config, compiled_defaults};
use ferret::store::{self, NewChunk, NewFile, Store};
use ferret::types::{Backend, ChunkKind, FileKind, PrefixProfile};
use std::cell::RefCell;
use std::path::PathBuf;

thread_local! {
    /// Keeps the per-test tempdirs (and their index files) alive until the
    /// test thread ends; the store only borrows the paths.
    static KEEP: RefCell<Vec<tempfile::TempDir>> = const { RefCell::new(Vec::new()) };
}

fn tmp_index() -> PathBuf {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.sqlite");
    KEEP.with(|k| k.borrow_mut().push(dir));
    path
}

fn cfg_with_dims(dims: u32) -> Config {
    let mut cfg = compiled_defaults(&common::test_dirs(std::path::Path::new("/ferret-t")));
    cfg.embedder.dimensions = dims;
    cfg
}

fn open_temp(dims: u32) -> Store {
    let path = tmp_index();
    let cfg = cfg_with_dims(dims);
    store::open(&path, &cfg, true).unwrap()
}

fn file(path: &str, kind: FileKind) -> NewFile {
    let ext = match kind {
        FileKind::Text => "txt",
        FileKind::Image => "png",
        FileKind::Pdf => "pdf",
    };
    NewFile {
        path: path.into(),
        collection: "Documents".into(),
        kind,
        ext: ext.into(),
        size: 1,
        mtime: 1,
        fingerprint: "x".into(),
        complete: true,
        next_page: None,
        skip_reason: None,
        indexed_at: 1,
        error: None,
    }
}

fn chunk(file_id: i64, ordinal: i64, kind: ChunkKind, text: Option<&str>) -> NewChunk {
    NewChunk {
        file_id,
        ordinal,
        page: None,
        kind,
        text: text.map(String::from),
        start_byte: None,
        end_byte: None,
    }
}

#[test]
fn opens_and_registers_vec() {
    let store = open_temp(768);
    let v: String = store
        .conn()
        .query_row("select vec_version()", [], |r| r.get(0))
        .unwrap();
    assert!(v.starts_with('v'));
}

#[test]
fn sqlite_version_is_at_least_3_38() {
    let store = open_temp(8);
    let v: String = store
        .conn()
        .query_row("select sqlite_version()", [], |r| r.get(0))
        .unwrap();
    let parts: Vec<u32> = v.split('.').map(|p| p.parse().unwrap()).collect();
    assert!((parts[0], parts[1]) >= (3, 38), "sqlite_version() = {v}");
}

#[test]
fn journal_mode_is_wal() {
    let store = open_temp(8);
    let mode: String = store
        .conn()
        .query_row("pragma journal_mode", [], |r| r.get(0))
        .unwrap();
    assert_eq!(mode.to_lowercase(), "wal");
}

#[test]
fn model_mismatch_is_fatal() {
    let path = tmp_index();
    let cfg768 = cfg_with_dims(768);
    store::open(&path, &cfg768, true).unwrap();
    let cfg256 = cfg_with_dims(256);
    let err = store::open(&path, &cfg256, true).unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
fn prefix_profile_mismatch_is_fatal() {
    let path = tmp_index();
    let mut cfg = cfg_with_dims(768);
    store::open(&path, &cfg, true).unwrap();
    cfg.embedder.prefix_profile = PrefixProfile::LitertLm;
    let err = store::open(&path, &cfg, true).unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
fn model_id_mismatch_is_fatal() {
    let path = tmp_index();
    let mut cfg = cfg_with_dims(768);
    store::open(&path, &cfg, true).unwrap();
    cfg.embedder.model = "other-model".into();
    let err = store::open(&path, &cfg, true).unwrap_err();
    assert_eq!(err.exit_code(), 1);
    assert!(err.to_string().contains("model_id"), "{}", err);
    assert!(err.to_string().contains("--index"), "{}", err);
}

#[test]
fn backend_only_change_is_a_warning_not_an_error() {
    let path = tmp_index();
    let cfg = cfg_with_dims(768);
    store::open(&path, &cfg, true).unwrap();
    let mut other = cfg_with_dims(768);
    other.embedder.backend = Backend::Sglang;
    // Must open fine: model_id, dimensions, and prefix_profile all match.
    store::open(&path, &other, true).unwrap();
}

#[test]
fn new_index_writes_initial_meta() {
    let store = open_temp(768);
    assert_eq!(
        store.get_meta("schema_version").unwrap().as_deref(),
        Some("1")
    );
    assert_eq!(
        store.get_meta("model_id").unwrap().as_deref(),
        Some("embeddinggemma-2")
    );
    assert_eq!(
        store.get_meta("dimensions").unwrap().as_deref(),
        Some("768")
    );
    assert_eq!(
        store.get_meta("prefix_profile").unwrap().as_deref(),
        Some("embeddinggemma2")
    );
    assert_eq!(
        store.get_meta("backend").unwrap().as_deref(),
        Some("llama_server")
    );
    for key in [
        "images_supported",
        "config_fingerprint",
        "text_pipeline",
        "image_pipeline",
    ] {
        assert!(
            store.get_meta(key).unwrap().is_some(),
            "meta key {key} missing"
        );
    }
    assert_eq!(store.get_meta("nope").unwrap(), None);
}

#[test]
fn file_kind_metadata_is_files_kind_not_chunk_kind() {
    let store = open_temp(8);
    let tx = store.begin().unwrap();
    let fid = store
        .upsert_file(&NewFile {
            path: "/a/lease.pdf".into(),
            collection: "Documents".into(),
            kind: FileKind::Pdf,
            ext: "pdf".into(),
            size: 1,
            mtime: 1,
            fingerprint: "x".into(),
            complete: true,
            next_page: Some(3),
            skip_reason: None,
            indexed_at: 1,
            error: None,
        })
        .unwrap();
    let cid = store
        .insert_chunk(
            &tx,
            NewChunk {
                file_id: fid,
                ordinal: 0,
                page: Some(1),
                kind: ChunkKind::PdfPage,
                text: Some("pets".into()),
                start_byte: None,
                end_byte: None,
            },
        )
        .unwrap();
    let mut v = vec![0.0_f32; 8];
    v[0] = 1.0;
    store
        .insert_vec(&tx, cid, &v, FileKind::Pdf, "pdf", 1, "Documents")
        .unwrap();
    tx.commit().unwrap();
    let kind: String = store
        .conn()
        .query_row(
            "select file_kind from chunks_vec where rowid = ?",
            [cid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(kind, "pdf");
}

#[test]
fn empty_text_is_stored_as_null() {
    let store = open_temp(8);
    let fid = store
        .upsert_file(&file("/a/empty.txt", FileKind::Text))
        .unwrap();
    let tx = store.begin().unwrap();
    let cid = store
        .insert_chunk(&tx, chunk(fid, 0, ChunkKind::Text, Some("")))
        .unwrap();
    tx.commit().unwrap();
    let text: Option<String> = store
        .conn()
        .query_row("select text from chunks where id = ?", [cid], |r| r.get(0))
        .unwrap();
    assert_eq!(text, None);
}

#[test]
fn upsert_file_roundtrip_and_delete() {
    let store = open_temp(8);
    let mut row = file("/a/lease.pdf", FileKind::Pdf);
    row.next_page = Some(3);
    let id = store.upsert_file(&row).unwrap();
    let got = store.get_file_by_path("/a/lease.pdf").unwrap().unwrap();
    assert_eq!(got.id, id);
    assert_eq!(got.kind, FileKind::Pdf);
    assert_eq!(got.ext, "pdf");
    assert!(got.complete);
    assert_eq!(got.next_page, Some(3));

    // Upsert on the same path updates in place and keeps the id.
    let mut updated = row;
    updated.size = 9;
    updated.complete = false;
    updated.error = Some("boom".into());
    let id2 = store.upsert_file(&updated).unwrap();
    assert_eq!(id2, id);
    let got2 = store.get_file_by_path("/a/lease.pdf").unwrap().unwrap();
    assert_eq!(got2.size, 9);
    assert!(!got2.complete);
    assert_eq!(got2.error.as_deref(), Some("boom"));

    assert!(store.delete_file_by_path("/a/lease.pdf").unwrap());
    assert!(store.get_file_by_path("/a/lease.pdf").unwrap().is_none());
    assert!(!store.delete_file_by_path("/a/lease.pdf").unwrap());
}

#[test]
fn delete_file_cascades_chunks_and_vec_rows() {
    let store = open_temp(8);
    let fid = store
        .upsert_file(&file("/a/one.txt", FileKind::Text))
        .unwrap();
    let tx = store.begin().unwrap();
    let cid = store
        .insert_chunk(&tx, chunk(fid, 0, ChunkKind::Text, Some("hello")))
        .unwrap();
    store
        .insert_vec(
            &tx,
            cid,
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            FileKind::Text,
            "txt",
            1,
            "Documents",
        )
        .unwrap();
    tx.commit().unwrap();

    assert!(store.delete_file_by_path("/a/one.txt").unwrap());
    let counts: (i64, i64) = store
        .conn()
        .query_row(
            "select (select count(*) from chunks), (select count(*) from chunks_vec)",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(counts, (0, 0));
}

#[test]
fn delete_chunk_kind_sides_keep_the_other_side() {
    let store = open_temp(8);
    // Text file: only a 'text' chunk. PDF: 'pdf_page' + 'image' chunks.
    let fid_text = store
        .upsert_file(&file("/a/t.txt", FileKind::Text))
        .unwrap();
    let fid_pdf = store
        .upsert_file(&file("/a/mix.pdf", FileKind::Pdf))
        .unwrap();
    let tx = store.begin().unwrap();
    let c_text = store
        .insert_chunk(&tx, chunk(fid_text, 0, ChunkKind::Text, Some("body")))
        .unwrap();
    let c_page = store
        .insert_chunk(
            &tx,
            NewChunk {
                file_id: fid_pdf,
                ordinal: 0,
                page: Some(1),
                kind: ChunkKind::PdfPage,
                text: Some("page".into()),
                start_byte: None,
                end_byte: None,
            },
        )
        .unwrap();
    let c_img = store
        .insert_chunk(&tx, chunk(fid_pdf, 1, ChunkKind::Image, None))
        .unwrap();
    for cid in [c_text, c_page, c_img] {
        store
            .insert_vec(&tx, cid, &[0.5; 8], FileKind::Text, "txt", 1, "Documents")
            .unwrap();
    }
    tx.commit().unwrap();

    // Text side: delete 'text' chunks, keep 'image' + 'pdf_page'.
    store.delete_text_chunks_keep_images(fid_pdf).unwrap();
    let pdf_kinds: Vec<String> = store
        .conn()
        .prepare("select kind from chunks where file_id = ? order by id")
        .unwrap()
        .query_map([fid_pdf], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(pdf_kinds, vec!["pdf_page".to_string(), "image".to_string()]);
    let text_file_chunks: i64 = store
        .conn()
        .query_row(
            "select count(*) from chunks where file_id = ?",
            [fid_text],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(text_file_chunks, 1);

    // Image side: delete 'image' + 'pdf_page' chunks, keep 'text'.
    store.delete_image_chunks_keep_text(fid_pdf).unwrap();
    let pdf_left: i64 = store
        .conn()
        .query_row(
            "select count(*) from chunks where file_id = ?",
            [fid_pdf],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(pdf_left, 0);
    store.delete_image_chunks_keep_text(fid_text).unwrap();
    let text_left: i64 = store
        .conn()
        .query_row(
            "select count(*) from chunks where file_id = ?",
            [fid_text],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(text_left, 1);
    // Vector rows follow their chunks.
    let vec_rows: i64 = store
        .conn()
        .query_row("select count(*) from chunks_vec", [], |r| r.get(0))
        .unwrap();
    assert_eq!(vec_rows, 1);
}

#[test]
fn status_counts_reflects_rows() {
    let store = open_temp(8);
    let c0 = store.status_counts().unwrap();
    assert_eq!(
        (
            c0.files_text,
            c0.files_image,
            c0.files_pdf,
            c0.chunks,
            c0.incomplete,
            c0.errors
        ),
        (0, 0, 0, 0, 0, 0)
    );
    assert_eq!(c0.indexed_at_max, None);

    let mut t = file("/a/t.txt", FileKind::Text);
    t.indexed_at = 5;
    store.upsert_file(&t).unwrap();
    let mut p = file("/a/p.pdf", FileKind::Pdf);
    p.complete = false;
    p.error = Some("boom".into());
    p.indexed_at = 7;
    store.upsert_file(&p).unwrap();
    store
        .upsert_file(&file("/a/i.png", FileKind::Image))
        .unwrap();
    let c1 = store.status_counts().unwrap();
    assert_eq!((c1.files_text, c1.files_image, c1.files_pdf), (1, 1, 1));
    assert_eq!(c1.incomplete, 1);
    assert_eq!(c1.errors, 1);
    assert_eq!(c1.indexed_at_max, Some(7));
    assert_eq!(c1.chunks, 0);

    let fid = store.get_file_by_path("/a/t.txt").unwrap().unwrap().id;
    let tx = store.begin().unwrap();
    store
        .insert_chunk(&tx, chunk(fid, 0, ChunkKind::Text, Some("x")))
        .unwrap();
    tx.commit().unwrap();
    let c2 = store.status_counts().unwrap();
    assert_eq!(c2.chunks, 1);
}

#[test]
fn non_ferret_sqlite_file_is_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.sqlite");
    let foreign = rusqlite::Connection::open(&path).unwrap();
    foreign.execute("CREATE TABLE other(x)", []).unwrap();
    drop(foreign);
    let err = store::open(&path, &cfg_with_dims(768), true).unwrap_err();
    assert_eq!(err.exit_code(), 1);
    assert!(err.to_string().contains("meta"), "{}", err);
}
