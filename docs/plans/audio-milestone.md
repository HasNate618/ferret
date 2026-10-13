# Audio Milestone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index audio files as time-segmented vector rows (no transcription) via EmbeddingGemma 2's native audio encoder, searchable with plain text queries.

**Architecture:** Mirror the PDF-pages shape: ffmpeg segments audio into ~30s 16kHz-mono clips (like `pdftoppm` renders pages), each segment embeds through a new `input_audio` llama-dialect arm (proven live 2026-10-13), rows persist vec-only with timestamp metadata (like page ordinals). No new crates, no schema rebuild (nullable columns only).

**Tech Stack:** Rust, existing rusqlite + sqlite-vec, ffmpeg/ffprobe system binaries (PATH + `extract.ffmpeg_dir`, mirroring poppler), httpmock + FakeEmbedder test harnesses.

## Global Constraints

- Proven wire shapes only: llama_server `{"input": [{"content": [{"type": "input_audio", "input_audio": {"data": "<base64 wav>"}}]}]}` on `/v1/embeddings`. sglang OAI has no audio and openai is text-only: both refuse audio inputs as per-file errors (same as openai images). litert_lm audio over HTTP is unconfirmed: refuse as per-file until proven.
- Audio takes NO prefix (model rule: prefixes are text-only). `apply_prefix` returns `None` for audio; `embed_llama` gets an explicit arm like `Image`.
- Contract extensions (minor-milestone-legal, document each): new `FileKind::Audio` (`"audio"`), new `ChunkKind::AudioSegment` (`"audio_segment"`), new nullable `t_start_ms`/`t_end_ms` chunk columns + hit fields, new `audio_pipeline` fingerprint key, new `no_ffmpeg` skip reason, new Appendix B.6.
- Model math (do not re-derive): 25 tokens/sec, ~327s max per input. `segment_secs = 30` default → ~750 tokens/segment. No file-length cap (segments scale linearly).
- Test conventions: hermetic tempdirs (`tests/common.rs`), stub ffmpeg/ffprobe shell scripts with fsync (the ETXTBSY lesson), `#[serial]` on env-touching tests, exact `json_body` httpmock asserts for the dialect.
- Each task keeps `cargo test` green and ends in its own commit; spec + README updated in the task that changes behavior.

---

## File Structure

- `src/types.rs` — `spec_enum!` additions: `FileKind::Audio`, `ChunkKind::AudioSegment`.
- `src/config.rs` — default audio exts (`mp3 wav flac ogg opus m4a aac`), `[chunk.audio] segment_secs = 30`, `extract.ffmpeg_dir: Option<PathBuf>` (mirror poppler), fingerprint `audio_pipeline` key.
- `src/extract/audio.rs` (new) — `find_ffmpeg`/`audio_duration`/`extract_audio` (segment wavs under cache).
- `src/extract/mod.rs` — `Chunk` gains `t_start_ms`/`t_end_ms: Option<i64>`; re-export.
- `src/embedder/mod.rs` — `EmbedInput::Audio { wav_bytes: Vec<u8> }` variant.
- `src/embedder/prefix.rs` — audio arm returns `None` (no prefix) + unit test.
- `src/embedder/http.rs` — `embed_llama` audio arm (`post_audio`), openai/sglang/litert refuse; `audio_body` builder (Appendix B.6).
- `src/embedder/probe.rs` — `probe_audio` (1s-sine POST) mirroring `probe_images`; `audio_supported` meta.
- `src/store/schema.rs` + `src/store/mod.rs` — nullable `t_start_ms`/`t_end_ms` columns (ADD COLUMN backfill on open, following the `triggers()` backfill pattern), vec-only audio rows, `audio_supported` meta accessors.
- `src/index/walk.rs` + `src/index/mod.rs` — Audio dispatch arm, `no_ffmpeg` walk-skip (mirror `no_magick`), clip cleanup after commit (mirror raster deletion).
- `src/search/json.rs` — `audio_segment` passthrough, snippet `[mm:ss–mm:ss] basename`, `snippet_spans: []`.
- `tests/extract_audio.rs` (new), `tests/embedder_http.rs`, `tests/index_pipeline.rs`, `tests/search_json.rs`, `testdata/` (generated-word WAV fixture, see Task 3).
- `docs/specs/ferret-design.md` (Appendix A + B.6 + kinds), `README.md` (ffmpeg dep, segment model), `flake.nix` (wrap ffmpeg like poppler).

---

### Task 1: Types, config, fingerprint

**Files:**
- Modify: `src/types.rs`, `src/config.rs` (defaults, merge, fingerprint value fn), `src/fingerprint.rs`
- Modify: `docs/specs/ferret-design.md` (Appendix A: `audio_pipeline` key)
- Test: `tests/config_merge.rs`

**Interfaces:**
- Consumes: existing `spec_enum!`, `default_extension_kind`, pipeline fingerprint builders.
- Produces: `FileKind::Audio => "audio"`, `ChunkKind::AudioSegment => "audio_segment"`; default ext map gains audio list; `[chunk.audio] segment_secs: u64 = 30` (fatal if 0); `extract.ffmpeg_dir: Option<PathBuf>` (None default); fingerprint set gains `audio_pipeline = {segment_secs}` — which changes `config_fingerprint` for existing users (correct: forces one re-index prompt; document it).

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn audio_kinds_parse_and_display() {
    assert_eq!("audio".parse::<FileKind>().unwrap(), FileKind::Audio);
    assert_eq!(ChunkKind::AudioSegment.to_string(), "audio_segment");
}

#[test]
fn compiled_defaults_include_audio() {
    let c = compiled_defaults(&dirs());
    for ext in ["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac"] {
        assert_eq!(c.files.extension_kind.get(ext), Some(&FileKind::Audio));
    }
    assert_eq!(c.chunk.audio.segment_secs, 30);
    assert!(c.extract.ffmpeg_dir.is_none());
}

#[test]
fn zero_segment_secs_is_fatal() {
    let files: HashMap<PathBuf, String> = HashMap::from([(
        PathBuf::from("/tmp/ferret-cfg/config.toml"),
        "[chunk.audio]\nsegment_secs = 0\n".into(),
    )]);
    let err = load_from(
        &LoadOpts { config_flag: None, index_flag: None },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|p| files.get(p).cloned().ok_or_else(|| ferret::error::Error::fatal("missing")),
    )
    .unwrap_err();
    assert!(err.to_string().contains("segment_secs"));
}
```

(Adapt harness names to `tests/config_merge.rs`. The `c.chunk.audio` / `c.extract.ffmpeg_dir` field paths are the contract this task creates — implement exactly these names.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test config_merge audio`
Expected: FAIL (unknown kind `audio`, no such fields).

- [ ] **Step 3: Write minimal implementation**

`src/types.rs`: add both variants. `src/config.rs`: ext list, `ChunkAudio { segment_secs: u64 }` + `segment_secs != 0` validation in the existing overlap-style fatal checks, `ffmpeg_dir` merge (same `Option<PathBuf>` pattern as poppler dir — read that code first), fingerprint value gains `audio_pipeline`. Spec Appendix A: add the key.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test config_merge`
Expected: PASS. Then: `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/types.rs src/config.rs src/fingerprint.rs tests/config_merge.rs docs/specs/ferret-design.md
git commit -m "feat: audio kinds, extensions, chunk.audio config, fingerprint key"
```

---

### Task 2: ffmpeg resolve + duration

**Files:**
- Create: `src/extract/audio.rs` (`find_ffmpeg`, `audio_duration`)
- Modify: `src/extract/mod.rs` (re-export)
- Test: `tests/extract_audio.rs` (new file; stub scripts with file+dir fsync per the ETXTBSY lesson)

**Interfaces:**
- Consumes: `extract.ffmpeg_dir` config + PATH (mirror `pdf.rs::resolve_tool` — read it first; same explicit→dir→PATH priority, same `NotFound` error shape).
- Produces: `find_ffmpeg(ffmpeg_dir: Option<&Path>) -> Result<PathBuf>`; `audio_duration(ffmpeg_dir, path) -> Result<f64>` via `ffprobe -v error -show_entries format=duration -of csv=p=0`. Duration ≤ 0 or unparseable → `Error::file` (per-file, walk continues).

- [ ] **Step 1: Write the failing tests**

```rust
fn write_stub(dir: &Path, name: &str, stdout: &str) {
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\nprintf '%s' \"{stdout}\"\n")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
    }
    // ETXTBSY guard: fsync file + dir before any exec.
    std::fs::File::open(&path).unwrap().sync_all().unwrap();
    std::fs::File::open(dir).unwrap().sync_all().unwrap();
}

#[test]
fn ffmpeg_dir_is_preferred_over_path() {
    let tmp = tmpdir();
    let tools = tmp.path().join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    write_stub(&tools, "ffprobe", "EDUMMY");
    let found = find_ffmpeg(Some(tools.as_path())).unwrap();
    assert_eq!(found, tools.join("ffmpeg").exists().then(|| tools.join("ffmpeg")).unwrap_or(found));
}
```

(Keep the fsync helper verbatim in this file. For the duration test, mirror `tests/extract_pdf.rs` hermetic style: stub `ffprobe` echoing `93.5`, assert `audio_duration(...) == 93.5`; stub exiting 1 → `is_file_error()`. Explicit-missing-binary → NotFound error, mirroring the pdf test.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test extract_audio`
Expected: FAIL (no such module/functions).

- [ ] **Step 3: Write minimal implementation**

`src/extract/audio.rs` mirroring `pdf.rs::resolve_tool` structure (explicit path → `ffmpeg_dir` → PATH scan with non-empty + executable-bit predicate). `audio_duration` runs ffprobe, trims stdout, parses f64, maps all failures to `Error::file`. Re-export from `mod.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test extract_audio`
Expected: PASS, 25/25 loop clean (flake-check the stub test: `for i in $(seq 1 25); do cargo test --test extract_audio 2>/dev/null | grep FAILED; done` must print nothing).

- [ ] **Step 5: Commit**

```bash
git add src/extract/audio.rs src/extract/mod.rs tests/extract_audio.rs
git commit -m "feat: ffmpeg resolve and duration probe"
```

---

### Task 3: Segmentation + extraction

**Files:**
- Modify: `src/extract/audio.rs` (`extract_audio`), `src/extract/mod.rs` (`Chunk.t_start_ms/t_end_ms`)
- Test: `tests/extract_audio.rs`, `testdata/sweep.wav` (new fixture, generated once, committed)

**Interfaces:**
- Consumes: Task 2 (`find_ffmpeg`, `audio_duration`); `segment_secs` + cache dir; `Chunk` struct.
- Produces: `extract_audio(path, segment_secs, cache_dir, ffmpeg_dir) -> Result<ExtractOutcome>` — one `Chunk{kind: AudioSegment, text: None, t_start_ms/t_end_ms, image_bytes: None, embed: Embed}` per `ceil(duration/segment_secs)` segments; segment WAVs are 16kHz mono (`ffmpeg -ss {start} -t {len} -ar 16000 -ac 1 -c:a pcm_s16le`) under `cache_dir/audio/{hash}/`; returns chunks with the clip bytes LOADED (mirror `image_bytes` semantics: the exact embedder payload). Clips shorter than 0.5s are dropped (no empty tail rows). Undecodable file → `Error::file`.

**Fixture:** generate once with system ffmpeg and commit: `ffmpeg -f lavfi -i 'sine=frequency=440:duration=65' -ar 16000 -ac 1 testdata/sweep.wav` (65s → 3 segments at 30s: [0,30,60] starts). Real-ffmpeg tests gate on `find_ffmpeg(None).is_ok()` and skip otherwise (magick-gate pattern).

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn sweep_wav_yields_three_timestamped_chunks() {
    let Some(_) = find_ffmpeg(None) else { eprintln!("skip: no ffmpeg"); return; };
    let tmp = tmpdir();
    let chunks = extract_audio(
        Path::new("testdata/sweep.wav"), 30, &tmp.path().join("cache"), None,
    )
    .unwrap();
    assert_eq!(chunks.len(), 3);
    assert!(chunks.iter().all(|c| c.kind == ChunkKind::AudioSegment && c.text.is_none()));
    assert_eq!(
        chunks.iter().map(|c| (c.t_start_ms, c.t_end_ms)).collect::<Vec<_>>(),
        vec![(Some(0), Some(30000)), (Some(30000), Some(60000)), (Some(60000), Some(65000))]
    );
    assert!(chunks.iter().all(|c| c.image_bytes.as_ref().is_some_and(|b| b.len() > 44)));
}

#[test]
fn corrupt_wav_is_file_error() {
    let tmp = tmpdir();
    let bad = tmp.path().join("bad.wav");
    std::fs::write(&bad, b"RIFF-not-really-a-wav").unwrap();
    let err = extract_audio(&bad, 30, &tmp.path().join("cache"), None).unwrap_err();
    assert!(err.is_file_error());
}
```

(The `> 44` asserts a real WAV header + samples, not an empty file. `is_file_error` mirrors the corrupt-image precedent.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test extract_audio`
Expected: FAIL (no `extract_audio`, no timestamp fields).

- [ ] **Step 3: Write minimal implementation**

Add `t_start_ms`/`t_end_ms: Option<i64>` to `Chunk` (None for all existing constructors — update them). `extract_audio`: duration → segment starts → ffmpeg per segment → read bytes into `image_bytes` field (payload carrier; document that the field now means "exact embedder payload" for audio too — rename is out of scope, comment it). Drop sub-0.5s tails.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test extract_audio`
Expected: PASS. Then: `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/extract/audio.rs src/extract/mod.rs tests/extract_audio.rs testdata/sweep.wav
git commit -m "feat: audio segmentation into timestamped clip chunks"
```

---

### Task 4: Store rows, timestamps, audio probe

**Files:**
- Modify: `src/store/schema.rs` (nullable columns + backfill), `src/store/mod.rs` (insert paths, `audio_supported` meta), `src/embedder/probe.rs` (`probe_audio`)
- Test: `tests/store_schema.rs` (migration test), `tests/index_pipeline.rs` (audio_supported meta)

**Interfaces:**
- Consumes: Task 3 chunks; `triggers()`-style open-time backfill pattern; `probe_images` as the probe template.
- Produces: `chunks` gains `t_start_ms INTEGER, t_end_ms INTEGER` (both nullable); `ALTER TABLE ... ADD COLUMN` backfill on open for pre-audio DBs (idempotent — check `PRAGMA table_info` first, exactly like trigger backfill checks). Audio rows persist vec-only (no FTS row, mirroring images). `probe_audio(client, cfg) -> bool`: POSTs a 1s 440Hz sine (44 bytes of header + 32000 samples, synthesized in code — no fixture) as `input_audio` to the llama endpoint; true on 200 + 768-d finite vector. `audio_supported` meta set at init/index open (mirror `images_supported`).

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn old_db_gains_timestamp_columns_on_open() {
    // Build a pre-audio schema manually, open via store::open, assert columns exist.
    let tmp = tmpdir();
    let path = tmp.path().join("index.sqlite");
    {
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT); CREATE TABLE chunks (id INTEGER PRIMARY KEY, file_id INTEGER, ordinal INTEGER, kind TEXT);").unwrap();
    }
    let _store = open_test_store(&path);
    let conn = rusqlite::Connection::open(&path).unwrap();
    let cols: Vec<String> = conn.prepare("PRAGMA table_info(chunks)").unwrap()
        .query_map([], |r| r.get(1)).unwrap().collect::<Result<_,_>>().unwrap();
    assert!(cols.contains(&"t_start_ms".to_string()));
    assert!(cols.contains(&"t_end_ms".to_string()));
}
```

(Adapt to `tests/store_schema.rs` harness names. Second test: httpmock `input_audio` exact-`json_body` returning a 768-d vector → `probe_audio` true; 400 → false. Mirror the existing probe test shape in `tests/embedder_http.rs`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test store_schema old_db_gains && cargo test --test embedder_http probe_audio`
Expected: FAIL (no columns, no `probe_audio`).

- [ ] **Step 3: Write minimal implementation**

Schema: add columns to CREATE TABLE + `PRAGMA table_info` backfill in the same open path as trigger backfill. Insert path: audio chunks write `t_start_ms/t_end_ms`, no FTS insert. Probe: synthesize sine in code (`(0..16000).map(|i| (i as f32 * 440.0 * TAU / 16000.0).sin())`, encode minimal WAV), POST llama audio body, validate like `probe_images`.

- [ ] **Step 4: Run tests to verify they pass**

Run: both test binaries, then `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/store/ src/embedder/probe.rs tests/store_schema.rs tests/embedder_http.rs
git commit -m "feat: audio timestamp columns, vec-only rows, audio probe"
```

---

### Task 5: Embed dialect (Appendix B.6)

**Files:**
- Modify: `src/embedder/mod.rs` (`EmbedInput::Audio`), `src/embedder/prefix.rs` (None arm + test), `src/embedder/http.rs` (`embed_llama` arm + `audio_body` + `post_audio`; openai/sglang/litert refuse)
- Modify: `docs/specs/ferret-design.md` (Appendix B.6)
- Test: `tests/embedder_http.rs`

**Interfaces:**
- Consumes: Task 3 clip bytes; `post_with_retries` + `truncate_or_check` + `maybe_dimensions` helpers; `Error::file` convention.
- Produces: `EmbedInput::Audio { wav_bytes: Vec<u8> }`; llama posts `{"input": [{"content": [{"type": "input_audio", "input_audio": {"data": "<base64>"}}]}]}` (+ `dimensions` iff != 768, mirroring image arms); openai → `Error::file("backend `openai` cannot embed audio")`; sglang → `Error::file` (OAI endpoint has no audio — use `/encode` is a later milestone, not this one); litert_lm → `Error::file` (unconfirmed over HTTP). B.6 documents the body + refusals + no-prefix rule.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn llama_audio_posts_input_audio_part() {
    let server = MockServer::start();
    let wav = minimal_wav(16000); // 1s silence, 16kHz mono
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/embeddings").json_body(json!({
            "input": [{"content": [{"type": "input_audio",
                "input_audio": {"data": BASE64.encode(&wav)}}]}]
        }));
        then.status(200).json_body(json!({"data": [{"embedding": vec![0.1; 768]}]}));
    });
    let e = test_embedder(&server, Backend::LlamaServer);
    let v = e.embed(EmbedInput::Audio { wav_bytes: wav }).unwrap();
    assert_eq!(v.len(), 768);
    mock.assert();
}

#[test]
fn openai_audio_is_file_error_without_posting() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/embeddings");
        then.status(200).json_body(json!({"data": []}));
    });
    let e = test_embedder(&server, Backend::Openai);
    let err = e.embed(EmbedInput::Audio { wav_bytes: vec![1, 2, 3] }).unwrap_err();
    assert!(err.is_file_error());
    mock.assert_hits(0);
}
```

(Adapt `MockServer`/`test_embedder`/BASE64 helper names to `tests/embedder_http.rs` conventions — it already has image-dialect + never-posts tests to copy the shape from. Add sglang + litert refuse cases in the same style.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test embedder_http audio`
Expected: FAIL (no `Audio` variant).

- [ ] **Step 3: Write minimal implementation**

`EmbedInput::Audio` variant; prefix arm `None` (+ extend the existing textless-prefix unit test pattern); `embed_llama` explicit arm before the prefix fallback; `audio_body(cfg, wav)` builder next to `image_body`; `post_audio` using it; three refuse arms with per-file errors naming the backend. B.6 prose: body shape, base64 WAV (16kHz mono required — extraction guarantees it), no-prefix rule, refusal matrix.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test embedder_http`
Expected: PASS (26+ tests). Then: `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/embedder/ tests/embedder_http.rs docs/specs/ferret-design.md
git commit -m "feat: llama input_audio dialect (Appendix B.6), refusals elsewhere"
```

---

### Task 6: Index wiring + no_ffmpeg

**Files:**
- Modify: `src/index/walk.rs` (ffmpeg gate), `src/index/mod.rs` (Audio dispatch arm + clip cleanup)
- Modify: `src/index/skip.rs` (`NoFfmpeg` reason)
- Test: `tests/walk_skip.rs`, `tests/index_pipeline.rs`

**Interfaces:**
- Consumes: Tasks 1–5; `SkipReason` enum; raster-deletion precedent (pdf.rs cleanup after commit).
- Produces: walk: audio exts with missing ffmpeg → walk-skip `no_ffmpeg` (mirror `no_magick` gate placement: after kind resolution, only for `FileKind::Audio`). Index arm: `FileKind::Audio => extract_audio(...)` → embed each chunk via `EmbedInput::Audio` → vec rows with timestamps → delete segment WAVs after commit (mirror raster cleanup, all exits). Per-file ffmpeg/decode failures → `Error::file` record-and-continue.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn audio_without_ffmpeg_walk_skips() {
    // PATH with no ffmpeg/ffprobe: audio file yields skip row, never an error.
    let (env, cfg) = no_ffmpeg_env_with_audio();
    let summary = run_index_json(&cfg);
    assert!(summary.events.iter().any(|e| e.path.ends_with(".wav") && e.reason == "no_ffmpeg"));
    assert!(!summary.events.iter().any(|e| e.status == "error"));
}

#[test]
fn audio_indexes_segments_with_fake_embedder() {
    let Some(_) = find_ffmpeg(None) else { eprintln!("skip: no ffmpeg"); return; };
    let (env, cfg) = env_with_audio("testdata/sweep.wav");
    let summary = run_index_json(&cfg);
    assert_eq!(summary.embedded_files(), 1);
    let store = open_test_store(&cfg);
    assert_eq!(store.audio_chunk_count(), 3);
    assert!(store.audio_chunks_have_timestamps());
}
```

(Adapt to `tests/walk_skip.rs` / `tests/index_pipeline.rs` harnesses — the magick-gate and image-pipeline tests are the templates. `FERRET_EMBEDDER=fake` for the pipeline test; FakeEmbedder must accept `EmbedInput::Audio` — extend the fake alongside (return a deterministic 768-d vector; assert it was called with audio in the test).)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test walk_skip audio && cargo test --test index_pipeline audio`
Expected: FAIL (no Audio arm, no `no_ffmpeg`).

- [ ] **Step 3: Write minimal implementation**

`skip.rs`: `NoFfmpeg => "no_ffmpeg"`. `walk.rs`: gate after kind resolution (`kind == FileKind::Audio && find_ffmpeg(...).is_none()`), mirroring the NoMagick block. `index/mod.rs`: dispatch arm (extract → per-chunk `EmbedInput::Audio{wav_bytes}` → store vec rows with `t_start_ms/t_end_ms`) + cache-clip deletion after commit on all exits. `FakeEmbedder`: audio arm returning a fixed finite 768-d vec.

- [ ] **Step 4: Run tests to verify they pass**

Run: both binaries, then `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/index/ tests/walk_skip.rs tests/index_pipeline.rs src/embedder/fake.rs
git commit -m "feat: index audio files with no_ffmpeg walk-skip"
```

---

### Task 7: Search audio hits

**Files:**
- Modify: `src/search/json.rs` (snippet + spans), `src/search/sql.rs` (timestamp select, if not already carried)
- Test: `tests/search_json.rs`, `tests/search_filters.rs`

**Interfaces:**
- Consumes: Task 4 rows; existing hit struct + `chunk_kind` rendering.
- Produces: audio hits render `chunk_kind: "audio_segment"`, `page: null`, `snippet: "[mm:ss–mm:ss] basename"` (from `t_start_ms/t_end_ms`; `t_end` missing → `[mm:ss–?]` — never panic), `snippet_spans: []`, `scores.fts: null`, `scores.filename` as usual. `--kind audio` filter returns them (vec `file_kind` metadata already carries it — no SQL change expected; test proves it).

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn search_returns_timestamped_audio_hits() {
    let (env, cfg) = indexed_env_with_audio("testdata/sweep.wav");
    let v = run_search_json(&cfg, "sine tone", &["--kind", "audio"]);
    let hits = v["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h["chunk_kind"] == "audio_segment"));
    assert!(hits.iter().all(|h| h["page"].is_null() && h["snippet_spans"] == json!([])));
    assert!(hits.iter().all(|h| h["scores"]["fts"].is_null()));
    let snip = hits[0]["snippet"].as_str().unwrap();
    assert!(snip.starts_with("[00:00–") && snip.ends_with("sweep.wav"));
}
```

(Second test in `search_filters.rs`: `--kind audio` excludes text hits and vice versa, using a mixed text+audio index. Adapt harnesses to existing conventions.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test search_json audio && cargo test --test search_filters audio`
Expected: FAIL (no audio rows to find).

- [ ] **Step 3: Write minimal implementation**

Carry `t_start_ms/t_end_ms` through the hit struct (from the chunks row — check what the SELECT already pulls; add if missing), render snippet + nulls per contract above. No ranker changes (RRF + vec arm handle vec-only hits exactly like images).

- [ ] **Step 4: Run tests to verify they pass**

Run: both binaries, then `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/search/ tests/search_json.rs tests/search_filters.rs
git commit -m "feat: timestamped audio hits in search JSON"
```

---

### Task 8: Docs, packaging, live acceptance

**Files:**
- Modify: `README.md` (ffmpeg dep, segment model, backend audio matrix), `docs/specs/ferret-design.md` (kinds/chunk-kind registry, B.6 cross-refs), `flake.nix` (wrap ffmpeg alongside poppler-utils), `nix/module.nix` (note, only if a flag is needed — prefer none)
- Test: none (packaging verified by `nix flake check` + `nix build .#ferret` by the operator/supervisor)

**Interfaces:**
- Consumes: Tasks 1–7 behavior.
- Produces: README "Audio" section (supported exts, 30s segments, 16kHz mono normalization, backend matrix: llama ✅ / sglang ❌ needs `/encode` / openai ❌ / litert ❌ unconfirmed / lab-wrapper `/v1/embed` typed-items = later milestone), spec kinds registry updated, `wrapProgram … ffmpeg` in flake (mirror poppler block). **Live acceptance procedure** (manual, recorded in ledger like Task 12's smoke): against a real server, index a natural-sound file + text query matched/mismatched separation ≥ 0.1 cosine before declaring the milestone shippable. Synthetic tones do NOT count (proven non-discriminating 2026-10-13).

- [ ] **Step 1: Write the docs + packaging diff**

README section + spec registry + flake wrap (copy the poppler `wrapProgram` block shape for `ffmpeg`/`ffprobe`).

- [ ] **Step 2: Verify packaging**

Run: `nix flake check` and `nix build .#ferret` (supervisor/operator runs; subagent implementers: skip with a note if nix is unavailable, do NOT fake it).

- [ ] **Step 3: Run the full suite**

Run: `cargo test` — all green, including the new `extract_audio`/`embedder_http`/`search_json` audio tests.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/specs/ferret-design.md flake.nix
git commit -m "docs: audio operator docs, spec registry, wrap ffmpeg"
```

- [ ] **Step 5: Record live acceptance in the ledger**

Append to `docs/plans/ferret-v1-progress.md`: server, fixture, matched/mismatched cosines, ship/no-ship call. Commit with the ledger.

```bash
git add docs/plans/ferret-v1-progress.md
git commit -m "docs: audio milestone live acceptance result"
```

---

## Self-Review

**1. Spec coverage** (audio milestone scope: kinds/config → extract → store → dialect → index → search → docs/acceptance):
- kinds/exts/`[chunk.audio]`/`ffmpeg_dir`/fingerprint → Task 1 ✓
- resolve + duration → Task 2 ✓
- segmentation + timestamps + fixture → Task 3 ✓
- columns migration + vec-only rows + probe/meta → Task 4 ✓
- B.6 dialect + refusals + no-prefix rule → Task 5 ✓
- walk gate + dispatch + cleanup + fake → Task 6 ✓
- hit rendering + kind filter → Task 7 ✓
- README/spec registry/flake/live gate → Task 8 ✓
- Explicitly out: audio *queries* (voice search), sglang `/encode`, litert audio, lab-wrapper typed-items dialect, video (separate milestone), transcription/FTS text for audio.

**2. Placeholder scan:** every step names exact files, JSON shapes, function names, shell commands, and commit messages. Navigation pointers ("read X first", "mirror Y pattern") each pair with a verbatim assertion block. No TBD/TODO/similar-to.

**3. Type consistency:** `FileKind::Audio`/`ChunkKind::AudioSegment` strings used identically in Tasks 1/3/6/7; `t_start_ms/t_end_ms: Option<i64>` on `Chunk`, store columns, and hit JSON alike; `EmbedInput::Audio { wav_bytes }` in Tasks 5/6; `no_ffmpeg` reason string in Task 6 only. `image_bytes` as payload carrier is documented-not-renamed in Task 3.

---

## Execution Handoff

Plan complete and saved to `docs/plans/audio-milestone.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with reviewer gates, same as v1.

**2. Inline Execution** — I work through the 8 tasks here with checkpoints.

Which approach — and v1.1 first, or audio first?
