# Video Milestone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index video files as time-segmented vector rows (no transcription, no frame-captioning) via EmbeddingGemma 2's native video encoder, searchable with plain text queries.

**Architecture:** Mirror the audio shape: ffmpeg cuts ~10s H.264 clips (like audio segments), each clip embeds through a new `input_video` llama-dialect arm (proven live 2026-10-13: 0.73 matched vs 0.61 unrelated), rows persist vec-only with the EXISTING `t_start_ms`/`t_end_ms` columns. No new crates, no schema migration (columns already exist), no flake change (ffmpeg already wrapped).

**Tech Stack:** Rust, existing rusqlite + sqlite-vec, ffmpeg/ffprobe system binaries (reuse `find_ffmpeg`/`audio_duration` + `extract.ffmpeg_dir`), httpmock + FakeEmbedder test harnesses.

## Global Constraints

- Proven wire shape only: llama_server `{"input": [{"content": [{"type": "input_video", "input_video": {"url": "data:video/mp4;base64,<b64>"}}]}]}` on `/v1/embeddings`. sglang OAI video unconfirmed, openai text-only, litert_lm unconfirmed: all refuse video inputs as per-file errors (same as audio refusals). Lab wrapper rejects video outright (`unknown input type 'video'`, probed 2026-10-13).
- Video takes NO prefix (model rule: prefixes are text-only). `apply_prefix` returns `None` for video; `embed_llama` gets an explicit arm like `Audio`.
- Contract extensions (minor-milestone-legal, document each): new `FileKind::Video` (`"video"`), new `ChunkKind::VideoSegment` (`"video_segment"`), new `video_pipeline` fingerprint key, new `no_video_backend` skip reason, new Appendix B.7. REUSED unchanged: `t_start_ms`/`t_end_ms` columns, `[mm:ss–mm:ss]` snippet shape, `no_ffmpeg` reason, `extract.ffmpeg_dir`.
- Model math (do not re-derive): ~140 tokens/frame, ~58 frames max per input at the server's `--video-fps` (default 4fps → ~14.5s budget). `segment_secs = 10` default → ~40 frames, comfortable headroom under the cap. Longer segments would silently lose tail frames to server-side truncation — hence 10, not 30.
- A video file's muxed audio travels INSIDE the clip sent to the server; ferret creates NO separate audio chunks for video files (no double-counting, no audio-track demux).
- Test conventions: hermetic tempdirs (`tests/common.rs`), real-ffmpeg tests gate on `find_ffmpeg(None).is_ok()` and skip otherwise (magick-gate pattern), `#[serial]` on env-touching tests, exact `json_body` httpmock asserts for the dialect.
- Each task keeps `cargo test` green and ends in its own commit; spec + README updated in the task that changes behavior.

---

## File Structure

- `src/types.rs` — `spec_enum!` additions: `FileKind::Video`, `ChunkKind::VideoSegment`.
- `src/config.rs` — default video exts (`mp4 mkv webm mov avi m4v`), `[chunk.video] segment_secs = 10`, fingerprint `video_pipeline` key. NO new `extract` field (`ffmpeg_dir` reused).
- `src/extract/video.rs` (new) — `extract_video` (mp4 clips under cache); REUSES `audio.rs::find_ffmpeg` + `audio_duration` (container-agnostic ffprobe duration — do NOT duplicate).
- `src/extract/mod.rs` — re-export; `Chunk` UNCHANGED (`t_start_ms/t_end_ms` already exist; video chunks carry clips in the same payload field audio uses).
- `src/embedder/mod.rs` — `EmbedInput::Video { mp4_bytes: Vec<u8> }` variant.
- `src/embedder/prefix.rs` — video arm returns `None` (no prefix) + unit test.
- `src/embedder/http.rs` — `embed_llama` video arm (`post_video`), openai/sglang/litert refuse; `video_body` builder (Appendix B.7).
- `src/embedder/probe.rs` — `probe_video` (tiny synthetic-bytes POST) mirroring `probe_audio`; `video_supported` meta.
- `src/store/mod.rs` — `video_supported` meta accessors (NO schema change; timestamp columns exist).
- `src/index/walk.rs` + `src/index/mod.rs` — Video dispatch arm, `no_ffmpeg` walk-skip REUSED (gate covers `FileKind::Video` too), NEW `no_video_backend` capability skip (mirror `NoAudioBackend`), clip cleanup after commit (mirror audio).
- `src/search/mod.rs` — generalize the `[mm:ss–mm:ss]` snippet guard from `AudioSegment` to both segment kinds.
- `tests/extract_video.rs` (new), `tests/embedder_http.rs`, `tests/index_pipeline.rs`, `tests/search_json.rs`, `testdata/clip.mp4` (generated lavfi fixture, see Task 2).
- `docs/specs/ferret-design.md` (Appendix A + B.7 + kinds), `README.md` (Video section, backend matrix).

---

### Task 1: Types, config, fingerprint

**Files:**
- Modify: `src/types.rs`, `src/config.rs` (defaults, merge, fingerprint value fn), `src/fingerprint.rs`
- Modify: `docs/specs/ferret-design.md` (Appendix A: `video_pipeline` key)
- Test: `tests/config_merge.rs`

**Interfaces:**
- Consumes: existing `spec_enum!`, `default_extension_kind`, pipeline fingerprint builders, Task-1-audio as the line-by-line template.
- Produces: `FileKind::Video => "video"`, `ChunkKind::VideoSegment => "video_segment"`; default ext map gains `mp4 mkv webm mov avi m4v → Video`; `[chunk.video] segment_secs: u64 = 10` (fatal if 0, same overlap-style check as audio); fingerprint set gains `video_pipeline = {segment_secs}` — forces one re-index prompt; document it.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn video_kinds_parse_and_display() {
    assert_eq!("video".parse::<FileKind>().unwrap(), FileKind::Video);
    assert_eq!(ChunkKind::VideoSegment.to_string(), "video_segment");
}

#[test]
fn compiled_defaults_include_video() {
    let c = compiled_defaults(&dirs());
    for ext in ["mp4", "mkv", "webm", "mov", "avi", "m4v"] {
        assert_eq!(c.files.extension_kind.get(ext), Some(&FileKind::Video));
    }
    assert_eq!(c.chunk.video.segment_secs, 10);
}

#[test]
fn zero_video_segment_secs_is_fatal() {
    let files: HashMap<PathBuf, String> = HashMap::from([(
        PathBuf::from("/tmp/ferret-cfg/config.toml"),
        "[chunk.video]\nsegment_secs = 0\n".into(),
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

(Adapt harness names to `tests/config_merge.rs`. The `c.chunk.video` field path is the contract this task creates — implement exactly this name.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test config_merge video`
Expected: FAIL (unknown kind `video`, no such field).

- [ ] **Step 3: Write minimal implementation**

`src/types.rs`: add both variants. `src/config.rs`: ext list, `ChunkVideo { segment_secs: u64 }` + `!= 0` fatal check, fingerprint value gains `video_pipeline`. Spec Appendix A: add the key.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test config_merge`
Expected: PASS. Then: `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/types.rs src/config.rs src/fingerprint.rs tests/config_merge.rs docs/specs/ferret-design.md
git commit -m "feat: video kinds, extensions, chunk.video config, fingerprint key"
```

---

### Task 2: Clip extraction + fixture

**Files:**
- Create: `src/extract/video.rs` (`extract_video`)
- Modify: `src/extract/mod.rs` (re-export)
- Test: `tests/extract_video.rs` (new file), `testdata/clip.mp4` (new fixture, generated once, committed)

**Interfaces:**
- Consumes: `audio.rs::find_ffmpeg` + `audio_duration` (REUSE — ffprobe `format=duration` is container-agnostic; do not write video-specific resolve/duration); `segment_secs` + cache dir; the audio `Chunk` payload-carrier field.
- Produces: `extract_video(path, segment_secs, cache_dir, ffmpeg_dir) -> Result<ExtractOutcome>` — one chunk per `ceil(duration/segment_secs)` segments (`kind: VideoSegment`, `text: None`, `t_start_ms/t_end_ms`, clip bytes loaded in the payload field). Clip encode: `ffmpeg -ss {start} -t {len} -i {in} -map 0:v:0 -map 0:a? -c:v libx264 -preset veryfast -crf 28 -vf scale=640:-2 -c:a aac -b:a 64k {out}.mp4` under `cache_dir/video/{hash}/`. Clips shorter than 0.5s are dropped (mirror audio tail rule). Undecodable file → `Error::file`.

**Fixture:** generate once with system ffmpeg and commit: `ffmpeg -f lavfi -i 'testsrc=duration=25:size=320x240:rate=10' -f lavfi -i 'sine=frequency=440:duration=25' -ar 16000 -ac 1 -c:v libx264 -pix_fmt yuv420p -shortest testdata/clip.mp4` (25s → 3 segments at 10s: [0,10,20] starts, last 5s). Real-ffmpeg tests gate on `find_ffmpeg(None).is_ok()` and skip otherwise.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn clip_mp4_yields_three_timestamped_chunks() {
    let Some(_) = find_ffmpeg(None) else { eprintln!("skip: no ffmpeg"); return; };
    let tmp = tmpdir();
    let chunks = extract_video(
        Path::new("testdata/clip.mp4"), 10, &tmp.path().join("cache"), None,
    )
    .unwrap();
    assert_eq!(chunks.len(), 3);
    assert!(chunks.iter().all(|c| c.kind == ChunkKind::VideoSegment && c.text.is_none()));
    assert_eq!(
        chunks.iter().map(|c| (c.t_start_ms, c.t_end_ms)).collect::<Vec<_>>(),
        vec![(Some(0), Some(10000)), (Some(10000), Some(20000)), (Some(20000), Some(25000))]
    );
    assert!(chunks.iter().all(|c| c.payload_bytes_len() > 1000));
}

#[test]
fn corrupt_mp4_is_file_error() {
    let tmp = tmpdir();
    let bad = tmp.path().join("bad.mp4");
    std::fs::write(&bad, b"\x00\x00\x00ftyp-not-really-mp4").unwrap();
    let err = extract_video(&bad, 10, &tmp.path().join("cache"), None).unwrap_err();
    assert!(err.is_file_error());
}
```

(Use the actual payload-field accessor name from `src/extract/mod.rs` — `image_bytes` or its successor — in place of `payload_bytes_len()`. The `> 1000` asserts a real encoded clip, not an empty file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test extract_video`
Expected: FAIL (no such module/function).

- [ ] **Step 3: Write minimal implementation**

`src/extract/video.rs` mirroring `audio.rs::extract_audio` structure (duration → segment starts → ffmpeg per segment → read bytes into payload field → drop sub-0.5s tails → cleanup partials on error via the same `clean_partials` pattern). Re-export from `mod.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test extract_video`
Expected: PASS. Then: `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/extract/video.rs src/extract/mod.rs tests/extract_video.rs testdata/clip.mp4
git commit -m "feat: video segmentation into timestamped mp4 clip chunks"
```

---

### Task 3: Probe + video_supported meta (no schema change)

**Files:**
- Modify: `src/embedder/probe.rs` (`probe_video`), `src/store/mod.rs` (`video_supported` meta accessors)
- Test: `tests/embedder_http.rs` (probe tests), `tests/index_pipeline.rs` (meta test)

**Interfaces:**
- Consumes: `probe_audio` as the template; `images_supported` meta as the accessor template; EXISTING `t_start_ms/t_end_ms` columns (verify with a comment, no migration code).
- Produces: `probe_video(client, cfg) -> bool`: POSTs a minimal synthetic payload (arbitrary bytes — httpmock never decodes; live decoding is the Task 8 gate) as `input_video` to the llama endpoint; true on 200 + 768-d finite vector, false otherwise; non-llama backends false without dialing (mirror the probe gate). `video_supported` meta set per run (mirror `audio_supported`).

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn probe_video_true_on_768() {
    // httpmock exact-json_body on the B.7 video shape returning vec![0.1; 768] → true.
}

#[test]
fn probe_video_false_on_400() {
    // httpmock 400 → false, no panic.
}

#[test]
fn video_supported_meta_recorded_with_fake() {
    // Mirror audio_supported_meta_recorded_with_fake: index run with FakeEmbedder
    // records video_supported=true in meta.
}
```

(Adapt to `tests/embedder_http.rs` / `tests/index_pipeline.rs` harness names — copy the `probe_audio` + `audio_supported` test shapes. FakeEmbedder video arm lands in Task 5; this task's meta test needs it — implement the fake arm HERE if Task 5 hasn't, or order Task 5 first. Supervisor: resolve the ordering when scheduling.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test embedder_http probe_video && cargo test --test index_pipeline video_supported`
Expected: FAIL (no `probe_video`, no meta key).

- [ ] **Step 3: Write minimal implementation**

`probe_video` next to `probe_audio` (shared 768+finite validation helper if one exists — read first; do not duplicate). Meta accessors next to `audio_supported`. Comment at the insert path: timestamp columns reused as-is, no backfill needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: both binaries, then `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/embedder/probe.rs src/store/mod.rs tests/embedder_http.rs tests/index_pipeline.rs
git commit -m "feat: video probe and video_supported meta"
```

---

### Task 4: Embed dialect (Appendix B.7)

**Files:**
- Modify: `src/embedder/mod.rs` (`EmbedInput::Video`), `src/embedder/prefix.rs` (None arm + test), `src/embedder/http.rs` (`embed_llama` arm + `video_body` + `post_video`; openai/sglang/litert refuse), `src/embedder/fake.rs` (deterministic video arm — needed by Task 3's meta test if Task 4 runs second; coordinate ordering)
- Modify: `docs/specs/ferret-design.md` (Appendix B.7)
- Test: `tests/embedder_http.rs`

**Interfaces:**
- Consumes: Task 2 clip bytes; `post_with_retries` + `truncate_or_check` helpers; `Error::file` convention.
- Produces: `EmbedInput::Video { mp4_bytes: Vec<u8> }`; llama posts `{"input": [{"content": [{"type": "input_video", "input_video": {"url": "data:video/mp4;base64,<b64>"}}]}]}` (+ `dimensions` iff != 768, mirroring audio/image arms); openai → `Error::file("backend `openai` cannot embed video")`; sglang → `Error::file` (unconfirmed over OAI HTTP); litert_lm → `Error::file` (unconfirmed over HTTP). B.7 documents the body + refusals + no-prefix rule + the frame-budget math that justifies `segment_secs = 10`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn llama_video_posts_input_video_part() {
    let server = MockServer::start();
    let mp4 = vec![0u8; 64]; // opaque bytes; the mock never decodes
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/embeddings").json_body(json!({
            "input": [{"content": [{"type": "input_video",
                "input_video": {"url": format!("data:video/mp4;base64,{}", BASE64.encode(&mp4))}}]}]
        }));
        then.status(200).json_body(json!({"data": [{"embedding": vec![0.1; 768]}]}));
    });
    let e = test_embedder(&server, Backend::LlamaServer);
    let v = e.embed(EmbedInput::Video { mp4_bytes: mp4 }).unwrap();
    assert_eq!(v.len(), 768);
    mock.assert();
}

#[test]
fn openai_video_is_file_error_without_posting() {
    // Mirror openai_audio_is_file_error_without_posting: per-file error + assert_hits(0).
}
```

(Add sglang + litert refuse cases in the same style as the audio ones.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test embedder_http video`
Expected: FAIL (no `Video` variant).

- [ ] **Step 3: Write minimal implementation**

`EmbedInput::Video` variant; prefix arm `None` (+ unit test); `embed_llama` explicit arm; `video_body(cfg, mp4)` builder next to `audio_body`; `post_video` using it; three refuse arms with per-file errors naming the backend. `FakeEmbedder`: deterministic finite 768-d vec for video (hash bytes like the audio arm). B.7 prose: body shape, base64 mp4 (clips carry muxed audio; no separate audio chunks), no-prefix rule, refusal matrix, frame budget.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test embedder_http`
Expected: PASS. Then: `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/embedder/ tests/embedder_http.rs docs/specs/ferret-design.md
git commit -m "feat: llama input_video dialect (Appendix B.7), refusals elsewhere"
```

---

### Task 5: Index wiring + no_video_backend

**Files:**
- Modify: `src/index/walk.rs` (ffmpeg gate for Video), `src/index/mod.rs` (Video dispatch arm + clip cleanup), `src/index/skip.rs` (`NoVideoBackend` reason)
- Test: `tests/walk_skip.rs`, `tests/index_pipeline.rs`

**Interfaces:**
- Consumes: Tasks 1–4; `SkipReason` enum; `NoAudioBackend` as the capability-skip template; audio clip-cleanup as the deletion template.
- Produces: walk: video exts with missing ffmpeg → walk-skip `no_ffmpeg` (EXTEND the existing audio gate to `FileKind::Video` — one gate, two kinds; do not add a second gate). Index arm: `FileKind::Video => extract_video(...)` → embed each chunk via `EmbedInput::Video` → vec rows with timestamps → delete clips after commit on all exits. Missing video support (`video_supported == false`) → persist `no_video_backend` capability-skip row (mirror `NoAudioBackend`, re-processes on probe flip). Per-file failures → `Error::file` record-and-continue.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn video_without_ffmpeg_is_no_ffmpeg_skip() {
    // PATH with no ffmpeg/ffprobe: video file yields skip row reason=no_ffmpeg, never an error.
}

#[test]
fn video_indexes_segments_with_fake_embedder() {
    let Some(_) = find_ffmpeg(None) else { eprintln!("skip: no ffmpeg"); return; };
    // testdata/clip.mp4 → 3 segments, timestamps [(0,10000),(10000,20000),(20000,25000)], 3 vec rows.
}

#[test]
fn video_without_video_backend_is_capability_skip() {
    // Mirror audio_without_audio_backend_is_capability_skip: support=false → complete row
    // with skip_reason no_video_backend.
}
```

(Adapt to `tests/walk_skip.rs` / `tests/index_pipeline.rs` harnesses — copy the audio test shapes.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test walk_skip video && cargo test --test index_pipeline video`
Expected: FAIL (no Video arm, no `no_video_backend`).

- [ ] **Step 3: Write minimal implementation**

`skip.rs`: `NoVideoBackend => "no_video_backend"`. `walk.rs`: extend the ffmpeg gate condition to video (single gate). `index/mod.rs`: dispatch arm + cleanup. Capability check before the fast path (mirror audio placement).

- [ ] **Step 4: Run tests to verify they pass**

Run: both binaries, then `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/index/ tests/walk_skip.rs tests/index_pipeline.rs
git commit -m "feat: index video files with no_video_backend capability skip"
```

---

### Task 6: Search video hits

**Files:**
- Modify: `src/search/mod.rs` (generalize snippet guard), `src/search/json.rs` (passthrough — verify, likely no change)
- Test: `tests/search_json.rs`, `tests/search_filters.rs`

**Interfaces:**
- Consumes: Task 5 rows; the audio snippet arm (`None if AudioSegment`).
- Produces: video hits render `chunk_kind: "video_segment"`, `page: null`, `snippet: "[mm:ss–mm:ss] basename"` (same shape, same `?` rule), `snippet_spans: []`, `scores.fts: null`. `--kind video` filter returns them (no SQL change expected; test proves it).

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn search_returns_timestamped_video_hits() {
    // Mirror search_returns_timestamped_audio_hits with a Video/VideoSegment fixture
    // (t 0/10000 + a t_end-missing segment → "[00:10–?]").
}

#[test]
fn kind_video_returns_only_video() {
    // Mirror kind_audio_returns_only_audio on a mixed text+video index, both directions.
}
```

(Order-free snippet asserts — identical fake vectors rank arbitrarily, proven in Task 7-audio.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test search_json video && cargo test --test search_filters video`
Expected: FAIL (no video rows to find).

- [ ] **Step 3: Write minimal implementation**

Generalize the `None if AudioSegment` guard to `None if AudioSegment | VideoSegment` (single arm — do not duplicate the format block). No ranker changes.

- [ ] **Step 4: Run tests to verify they pass**

Run: both binaries, then `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/search/ tests/search_json.rs tests/search_filters.rs
git commit -m "feat: timestamped video hits in search JSON"
```

---

### Task 7: Docs + packaging (no flake change)

**Files:**
- Modify: `README.md` (Video section, backend matrix audio→video column), `docs/specs/ferret-design.md` (kinds/chunk-kind registry, B.7 cross-refs)
- Test: none (packaging verified by `nix flake check` + `nix build .#ferret` by the operator/supervisor)

**Interfaces:**
- Consumes: Tasks 1–6 behavior.
- Produces: README "Video" section (supported exts, 10s segments + frame-budget rationale, H.264 clip normalization, backend matrix: llama ✅ / sglang ❌ / openai ❌ / litert ❌ unconfirmed / lab-wrapper ❌ rejects; no separate audio chunks for video files), spec kinds registry updated. **NO flake.nix change** (ffmpeg already wrapped in the audio milestone — verify with `grep ffmpeg flake.nix` and note it in the commit message).

- [ ] **Step 1: Write the docs diff**

README section + spec registry + B.7 cross-refs (mirror the audio Task 8 shape).

- [ ] **Step 2: Verify packaging**

Run: `nix flake check` and `nix build .#ferret` (supervisor/operator runs).

- [ ] **Step 3: Run the full suite**

Run: `cargo test` — all green.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/specs/ferret-design.md
git commit -m "docs: video operator docs, spec registry (ffmpeg already wrapped)"
```

---

### Task 8: Live acceptance (natural-video gate)

**Files:**
- Modify: `docs/plans/ferret-v1-progress.md` (ledger entry only)
- Test: none (manual procedure)

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: ledger entry with server, fixture, matched/mismatched cosines, ship/no-ship call.

**Procedure** (manual, supervisor/operator runs — subagent implementers: skip with a note):
1. Rebuild llama.cpp master CPU-only if needed (`-b/-ub 2048`; standing instruction: check system binary arch first). Weights: `/tmp/eg2/` if kept, else re-fetch official ggml-org Q8_0 GGUFs.
2. Serve with `--mmproj`, confirm `/health`.
3. Index a NATURAL video file (operator-supplied: game capture, CC0 stock clip — synthetic `testsrc` does NOT count, per the audio lesson) with `segment_secs = 10`.
4. Compute text-query matched/mismatched cosine separation (direct POSTs, same method as audio acceptance): matched segment best vs mismatched best.
5. Gate: separation ≥ 0.1 cosine → SHIP; else NO-SHIP on the semantic claim (pipeline still ships if all unit/integration gates green + end-to-end proven). Prior datum: +0.12 on the 2026-10-13 probe (matched 0.73 vs unrelated 0.61) — the gate is attainable.
6. Clean the build tree + corpus; keep weights.

- [ ] **Step 1: Run the procedure, record in the ledger**

Append to `docs/plans/ferret-v1-progress.md`: server, fixture, cosines, ship/no-ship call.

```bash
git add docs/plans/ferret-v1-progress.md
git commit -m "docs: video milestone live acceptance result"
```

---

## Self-Review

**1. Spec coverage** (video milestone scope: kinds/config → extract → probe/meta → dialect → index → search → docs/acceptance):
- kinds/exts/`[chunk.video]`/fingerprint → Task 1 ✓
- clip extraction + timestamps + fixture → Task 2 ✓ (resolve/duration REUSED, not rebuilt)
- probe + meta, zero schema change → Task 3 ✓
- B.7 dialect + refusals + no-prefix + fake → Task 4 ✓
- walk gate (extended) + dispatch + cleanup + capability skip → Task 5 ✓
- hit rendering (generalized guard) + kind filter → Task 6 ✓
- README/spec registry, flake correctly untouched → Task 7 ✓
- live natural-video gate → Task 8 ✓
- Explicitly out: video *queries* (search-by-clip), sglang `/encode`-style native endpoints, litert video, lab-wrapper video dialect, transcription/captions/OCR of video content, separate audio chunks for video files, per-request fps control.

**2. Placeholder scan:** every step names exact files, JSON shapes, ffmpeg argv, function names, shell commands, and commit messages. Reuse directives name the exact source (`audio.rs::find_ffmpeg`, `NoAudioBackend`, the audio snippet arm). No TBD/TODO/similar-to. One ordering note flagged (Task 3 meta test vs Task 4 fake arm) for the scheduler to resolve.

**3. Type consistency:** `FileKind::Video`/`ChunkKind::VideoSegment` strings identical across Tasks 1/2/5/6; `t_start_ms/t_end_ms: Option<i64>` reused, never re-declared; `EmbedInput::Video { mp4_bytes }` in Tasks 4/5; `no_video_backend` in Task 5 only; `video_pipeline = {segment_secs}` in Task 1 only; `segment_secs = 10` default in Tasks 1/2/8 with the frame-budget rationale in B.7.

---

## Execution Handoff

Plan complete and saved to `docs/plans/video-milestone.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with reviewer gates, same as v1/audio.

**2. Inline Execution** — I work through the 8 tasks here with checkpoints.

Which approach — begin Task 1 now?
