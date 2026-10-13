# ferret v1.1 ("conversation") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach ferret to talk to its callers: progress/plan events, JSON run summaries, roots in status, search warnings, path-only indexing, snippet highlight spans, and a pinned offline-keyword guarantee — all additive, zero breaking changes.

**Architecture:** Extend the existing NDJSON/JSON emitters and CLI flags in place; no new modules, no schema migration (new keys only), no new dependencies. Each task is one contract addition with its own tests.

**Tech Stack:** Rust (single binary), rusqlite + sqlite-vec + FTS5, serde_json, existing `FakeEmbedder`/httpmock test harnesses.

## Global Constraints

- Additive-only contract changes: new JSON keys, new NDJSON `event` types, new flags. Never rename, remove, or retype an existing field; `search --json` shape from spec §8 stays intact.
- GUI/CLI parsers must ignore unknown NDJSON `event` types — document this rule in README + spec amendment (Task 1).
- Exit codes unchanged: 0 ok (including zero hits and per-file index errors), 1 fatal/file-escalated, 2 usage.
- Every task keeps the full suite green (`cargo test`, 201+ tests) and ends in its own commit; spec doc (`docs/specs/ferret-design.md`) and README updated in the same task that changes the behavior.
- Test conventions: hermetic tempdirs via `tests/common.rs` (`test_env()`, `EnvGuard`), `FERRET_EMBEDDER=fake` for index-building phases, `#[serial]` where process env is touched, real assertions on parsed JSON (never on eyeballed text).

---

## File Structure

No new files. All work lands in existing emitters/parsers:

- `src/index/mod.rs` — NDJSON per-file emission (~line 698), stderr summary, `index::run(cfg, embedder, extra, json, strict, out, err)` signature (unchanged).
- `src/index/walk.rs` — `walk(roots, extra)` root loop + extra-path handling (~lines 89–160).
- `src/search/mod.rs` — mode branching (`want_vec`/`want_fts`), warning-condition detection site.
- `src/search/json.rs` — search JSON rendering (hits, filters); snippet builder lives here or in `sql.rs` (implementer: read both first).
- `src/search/sql.rs` — FTS5 arm + `vec_blob` query-side blob builder.
- `src/cli.rs` — flag definitions (`--only`), `status --json` rendering, `SYSTEM_CONFIG_PATH` unaffected.
- `src/store/mod.rs` — status counts source (`f32_blob` is LE-unified; untouched).
- `tests/index_pipeline.rs`, `tests/cli_maintenance.rs`, `tests/search_json.rs`, `tests/search_filters.rs` — task tests live with their subsystem.
- `docs/specs/ferret-design.md`, `README.md` — amended per task.

---

### Task 1: NDJSON `plan` event (progress totals)

**Files:**
- Modify: `src/index/mod.rs` (emit after walk, before first file processed)
- Modify: `docs/specs/ferret-design.md` (§ on NDJSON events + unknown-event rule), `README.md` (index --json section)
- Test: `tests/index_pipeline.rs` (append new test)

**Interfaces:**
- Consumes: existing walk result (file list + sizes already in hand where per-file events are emitted); the `out: &mut dyn Write` + `json: bool` params of `index::run`.
- Produces: first stdout line `{"event":"plan","files":<u64>,"bytes":<u64>}` when `json=true`, emitted after the walk completes and before any `file` event. `bytes` = sum of file sizes as walked (u64, no truncation). Non-JSON mode unchanged. Unknown-event-tolerance rule documented for parsers.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn index_json_emits_plan_event_first() {
    let env = common::test_env();
    let root = env.root();
    std::fs::write(root.join("a.txt"), "alpha beta gamma").unwrap();
    std::fs::write(root.join("b.txt"), "delta epsilon").unwrap();
    let cfg = test_config(&env, root);
    let mut out = Vec::new();
    let mut err = Vec::new();
    ferret::index::run(&cfg, &FakeEmbedder, &[], true, false, &mut out, &mut err).unwrap();
    let stdout = String::from_utf8(out).unwrap();
    let first: serde_json::Value = serde_json::from_str(stdout.lines().next().unwrap()).unwrap();
    assert_eq!(first["event"], "plan");
    assert_eq!(first["files"], 2);
    let expected_bytes =
        std::fs::metadata(root.join("a.txt")).unwrap().len()
        + std::fs::metadata(root.join("b.txt")).unwrap().len();
    assert_eq!(first["bytes"], expected_bytes);
}
```

(Adapt harness names — `test_env()`, config builder, `FakeEmbedder` import — to the exact helpers already used in `tests/index_pipeline.rs`; keep the assertions above verbatim.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test index_pipeline index_json_emits_plan_event_first`
Expected: FAIL (first line is a `file` event, has no `event == "plan"`).

- [ ] **Step 3: Write minimal implementation**

In `src/index/mod.rs`, at the point where the walk has produced its file list and `json` is true, write the plan line to `out` before the processing loop starts:

```rust
if json {
    let bytes: u64 = files.iter().map(|f| f.size).sum();
    writeln!(out, "{}", serde_json::json!({
        "event": "plan",
        "files": files.len(),
        "bytes": bytes,
    }))?;
}
```

(Use the actual walk-result binding names in that function; `size` is whatever the walked entry already carries — do not add a second stat pass.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test index_pipeline index_json_emits_plan_event_first`
Expected: PASS. Then: `cargo test` (full suite green).

- [ ] **Step 5: Amend docs + commit**

Add to `docs/specs/ferret-design.md` NDJSON section: the `plan` event schema + the rule "parsers MUST ignore unknown `event` values". Add one line to README's `index --json` docs. Then:

```bash
git add src/index/mod.rs tests/index_pipeline.rs docs/specs/ferret-design.md README.md
git commit -m "feat: NDJSON plan event with file/byte totals"
```

---

### Task 2: JSON `summary` event on stdout

**Files:**
- Modify: `src/index/mod.rs` (same emitter area as Task 1)
- Modify: `README.md` (index --json section)
- Test: `tests/index_pipeline.rs`

**Interfaces:**
- Consumes: Task 1's `plan` line position (summary is the LAST stdout line); the existing `seen/skipped/embedded/errors` counters already computed for the stderr human summary.
- Produces: last stdout line `{"event":"summary","seen":N,"skipped":N,"embedded":N,"errors":N}` when `json=true`, with values identical to the stderr text summary. stderr human summary unchanged.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn index_json_emits_summary_event_last() {
    let env = common::test_env();
    let root = env.root();
    std::fs::write(root.join("a.txt"), "alpha beta gamma").unwrap();
    std::fs::write(root.join("skip.bin"), [0u8, 1, 2, 3]).unwrap();
    let cfg = test_config(&env, root);
    let mut out = Vec::new();
    let mut err = Vec::new();
    ferret::index::run(&cfg, &FakeEmbedder, &[], true, false, &mut out, &mut err).unwrap();
    let stdout = String::from_utf8(out).unwrap();
    let last: serde_json::Value = serde_json::from_str(stdout.lines().last().unwrap()).unwrap();
    assert_eq!(last["event"], "summary");
    assert_eq!(last["seen"], 2);
    assert_eq!(last["embedded"], 1);
    assert_eq!(last["errors"], 0);
    assert_eq!(last["skipped"], stdout.lines().filter(|l| l.contains("\"status\":\"skip\"")).count() as u64);
}
```

(`skip.bin` must be an extension the test config leaves unmapped so it walk-skips; check `files.extension_kind` defaults in `src/config.rs` and pick a truly unmapped ext. Adapt harness names to `tests/index_pipeline.rs`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test index_pipeline index_json_emits_summary_event_last`
Expected: FAIL (last line is a `file` event).

- [ ] **Step 3: Write minimal implementation**

After the processing loop in `src/index/mod.rs`, reusing the exact counter variables fed to the stderr summary:

```rust
if json {
    writeln!(out, "{}", serde_json::json!({
        "event": "summary",
        "seen": seen,
        "skipped": skipped,
        "embedded": embedded,
        "errors": errors,
    }))?;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test index_pipeline index_json_emits_summary_event_last`
Expected: PASS. Then: `cargo test` (full suite green).

- [ ] **Step 5: Amend docs + commit**

README `index --json` section: document the `summary` event with field meanings. Then:

```bash
git add src/index/mod.rs tests/index_pipeline.rs README.md
git commit -m "feat: JSON summary event on index stdout"
```

---

### Task 3: `roots` in `status --json`

**Files:**
- Modify: `src/cli.rs` (status JSON rendering — read the status handler first, it builds JSON from store counts + `cfg`)
- Modify: `README.md` (status section)
- Test: `tests/cli_maintenance.rs`

**Interfaces:**
- Consumes: `cfg.index.roots: Vec<RootEntry>` (`{path: PathBuf, collection: String}`) — no store changes.
- Produces: `status --json` gains `"roots": [{"path": <string>, "collection": <string>}]` (path as display string, collection verbatim, order = config order; `[]` when no roots). All existing keys byte-identical.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn status_json_lists_roots() {
    let env = common::test_env();
    let root = env.root();
    std::fs::create_dir_all(root.join("notes")).unwrap();
    let cfg = test_config_with_roots(&env, &[root.join("notes")]);
    let out = run_status_json(&cfg);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let roots = v["roots"].as_array().unwrap();
    assert_eq!(roots.len(), 1);
    assert!(roots[0]["path"].as_str().unwrap().ends_with("notes"));
    assert_eq!(roots[0]["collection"], "notes");
}
```

(Adapt to the existing status-test harness in `tests/cli_maintenance.rs` — same config builder + invocation path it already uses; keep assertions verbatim. Collection defaults to the dir file name per `RootEntry` construction.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test cli_maintenance status_json_lists_roots`
Expected: FAIL (`roots` key missing).

- [ ] **Step 3: Write minimal implementation**

In the `status --json` builder in `src/cli.rs`, add alongside the existing keys:

```rust
let roots: Vec<serde_json::Value> = cfg.index.roots.iter().map(|r| {
    serde_json::json!({
        "path": r.path.to_string_lossy(),
        "collection": r.collection,
    })
}).collect();
```

and insert `"roots": roots` into the emitted object.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test cli_maintenance status_json_lists_roots`
Expected: PASS. Then: `cargo test` (full suite green).

- [ ] **Step 5: Amend docs + commit**

README status section: document `roots`. Spec § on status output: add the key. Then:

```bash
git add src/cli.rs tests/cli_maintenance.rs README.md docs/specs/ferret-design.md
git commit -m "feat: list configured roots in status --json"
```

---

### Task 4: `warnings[]` in search JSON

**Files:**
- Modify: `src/search/mod.rs` (detect conditions where filters/mode are decided)
- Modify: `src/search/json.rs` (render `"warnings": [...]`)
- Modify: `docs/specs/ferret-design.md` (§8: warnings array + codes), `README.md` (search --json field notes)
- Test: `tests/search_json.rs`

**Interfaces:**
- Consumes: parsed request (which of `--under`/`--name` present), per-arm hit contribution (did the vec arm return any hit with non-null `scores.vec`?), total hit count.
- Produces: top-level `"warnings": []` (always present, possibly empty) containing zero or more of exactly these string codes:
  - `"vec_starved"` — a path-scoping filter (`--under` or `--name`) was present AND at least one hit was returned AND zero returned hits have non-null `scores.vec`.
  - `"browse_no_vec"` — only the FTS arm ran: the query carried no searchable text (existing filter-only browse behavior, now labeled), or explicit `--mode keyword`.
  - No other codes in v1.1. Codes are stable strings; parsers match on exact equality.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn search_json_warns_when_under_starves_vec() {
    let (env, cfg) = indexed_env_with_text(&["the quick brown fox"]);
    // --under narrows to a path the vec arm cannot reach: FTS still hits.
    let v = run_search_json(&cfg, "quick", &["--under", "nowhere-dir"]);
    assert!(v["hits"].as_array().unwrap().is_empty());
    // empty hits => no vec_starved (rule needs >= 1 hit); warnings is present but empty
    assert_eq!(v["warnings"], serde_json::json!([]));
}

#[test]
fn search_json_vec_starved_when_only_fts_hits() {
    let (env, cfg) = indexed_env_with_text(&["the quick brown fox"]);
    // Restrict --under to the real file's dir but use a name glob matching nothing semantic:
    // craft so FTS hits while vec contributes zero (see implementation note below).
    let v = run_search_json(&cfg, "quick", &["--name", "*.txt"]);
    let hits = v["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h["scores"]["vec"].is_null()));
    assert!(v["warnings"].as_array().unwrap().iter().any(|w| w == "vec_starved"));
}

#[test]
fn search_json_browse_labels_no_vec() {
    let (env, cfg) = indexed_env_with_text(&["the quick brown fox"]);
    let v = run_search_json(&cfg, "", &["--kind", "text"]);
    assert!(v["warnings"].as_array().unwrap().iter().any(|w| w == "browse_no_vec"));
}
```

(Adapt harness names to `tests/search_json.rs` conventions — it already builds fake-embedder indexes and parses search JSON. Implementation note for test 2: `--name` inflates the vec-arm `k` cap without a second KNN pass per spec; with `*.txt` matching, FTS hits while vec returns nothing — if the fixture instead yields vec hits, narrow with `--under <file-parent>` + a query term absent from embeddings but present as literal text. The rule is deterministic; make the fixture obey it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test search_json warnings && cargo test --test search_json browse_labels`
Expected: FAIL (`warnings` key missing).

- [ ] **Step 3: Write minimal implementation**

In `src/search/mod.rs`, after arms complete and before rendering, compute:

```rust
let mut warnings: Vec<&'static str> = Vec::new();
let scoped = filters.under.is_some() || filters.name.is_some();
let any_vec = hits.iter().any(|h| h.scores.vec.is_some());
if scoped && !hits.is_empty() && !any_vec {
    warnings.push("vec_starved");
}
if !want_vec {
    warnings.push("browse_no_vec");
}
```

(Field names per actual structs — `filters`, hit `scores.vec`, and the existing `want_vec` bool in `search/mod.rs`. Pass `warnings` into the `json.rs` renderer, which emits `"warnings": [...]` top-level, always present.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test search_json`
Expected: PASS (all, including the 5 frozen-contract tests). Then: `cargo test` full suite green.

- [ ] **Step 5: Amend docs + commit**

Spec §8 + README field notes: document `warnings`, both codes, exact trigger rules. Then:

```bash
git add src/search/mod.rs src/search/json.rs tests/search_json.rs docs/specs/ferret-design.md README.md
git commit -m "feat: search warnings array (vec_starved, browse_no_vec)"
```

---

### Task 5: `index --only PATH…` (skip the roots walk)

**Files:**
- Modify: `src/cli.rs` (`index` subcommand: `--only` flag; `--only` with zero positional PATHs = usage error exit 2)
- Modify: `src/index/walk.rs` (when `only=true`, skip the roots loop entirely; process `extra` paths exactly as today, including gitignore + collection inheritance)
- Modify: `README.md` (index section: watcher recipe using `--only`), `docs/specs/ferret-design.md` (index semantics)
- Test: `tests/index_pipeline.rs`

**Interfaces:**
- Consumes: existing `extra: &[PathBuf]` handling (unchanged semantics); a new `only: bool` threaded from CLI through `index::run` — add it as a parameter on `index::run` (signature change is internal, not contract) OR a field on the existing config/opts struct if one exists (implementer: check `index::run`'s current params; prefer the smallest diff).
- Produces: `ferret index --only PATH…` indexes exactly those paths, never walks roots. `ferret index --only` with no PATHs → exit 2 usage error ("--only requires at least one PATH"). Plain `index [PATH…]` behavior unchanged.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn index_only_skips_roots_walk() {
    let env = common::test_env();
    let root_a = env.root().join("a");
    let root_b = env.root().join("b");
    std::fs::create_dir_all(&root_a).unwrap();
    std::fs::create_dir_all(&root_b).unwrap();
    std::fs::write(root_a.join("in_a.txt"), "alpha alpha alpha").unwrap();
    std::fs::write(root_b.join("in_b.txt"), "beta beta beta").unwrap();
    let cfg = test_config_with_roots(&env, &[&root_a, &root_b]);
    let mut out = Vec::new();
    let mut err = Vec::new();
    // --only path in B: A must never appear in NDJSON.
    ferret::index::run_only(&cfg, &FakeEmbedder, &[root_b.join("in_b.txt")], true, false, &mut out, &mut err).unwrap();
    let stdout = String::from_utf8(out).unwrap();
    assert!(stdout.contains("in_b.txt"));
    assert!(!stdout.contains("in_a.txt"));
}

#[test]
fn index_only_without_paths_is_usage_error() {
    let status = run_cli(&["index", "--only", "--json"]);
    assert_eq!(status.code, 2);
}
```

(First test calls whatever entry point the implementation exposes — if `only` becomes a param on `index::run`, call it with `only=true`; keep the two assertions verbatim. Second test uses the existing CLI-invocation harness — check how usage-error tests invoke the CLI in `tests/cli_init.rs` and reuse that path.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test index_pipeline index_only`
Expected: FAIL (no `--only` entry point / roots always walked).

- [ ] **Step 3: Write minimal implementation**

CLI (`src/cli.rs`): add `#[arg(long)] only: bool` to the index args; if `only && paths.is_empty()` return `Error::usage("--only requires at least one PATH")`. Thread `only` into `index::run`. Walk (`src/index/walk.rs`): guard the `for root in roots` loop with `if !only`. Nothing else changes — extra-path collection inheritance, gitignore, and per-file processing are untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test index_pipeline index_only`
Expected: PASS. Then: `cargo test` full suite green.

- [ ] **Step 5: Amend docs + commit**

README index section + spec: document `--only`, its watcher use (batch changed paths, chunk ~500 per argv limits client-side), and the exit-2 case. Then:

```bash
git add src/cli.rs src/index/walk.rs src/index/mod.rs tests/index_pipeline.rs tests/cli_init.rs README.md docs/specs/ferret-design.md
git commit -m "feat: index --only skips the roots walk"
```

---

### Task 6: `snippet_spans` (highlight offsets in search hits)

**Files:**
- Modify: `src/search/sql.rs` (read FTS5 `offsets()` for the text column in the FTS arm)
- Modify: `src/search/json.rs` (render per-hit `"snippet_spans": [{"start": c, "end": c}]`)
- Modify: `docs/specs/ferret-design.md` (§8 hit schema), `README.md` (field notes)
- Test: `tests/search_json.rs`

**Interfaces:**
- Consumes: the existing snippet builder (implementer: read it first in `json.rs`/`sql.rs` — snippet is a ≤200-char window of chunk text); FTS5 `offsets(chunks_fts)` auxiliary function (returns phrase/column/byte-offset/byte-length quadruples for the matched row).
- Produces: every hit gains `"snippet_spans": []` (always present). For text-chunk hits whose snippet derives from the FTS-matched column: up to 5 `{start,end}` entries, CHAR offsets into the returned `snippet` string, derived from `offsets()` byte offsets mapped through the snippet window (byte→char conversion, clamp to snippet bounds, drop spans outside the window). Image/pdf_page hits and keyword-absent arms: `[]`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn search_json_snippet_spans_highlight_match() {
    let (env, cfg) = indexed_env_with_text(&["the quick brown fox jumps"]);
    let v = run_search_json(&cfg, "quick", &[]);
    let hit = &v["hits"].as_array().unwrap()[0];
    let snippet = hit["snippet"].as_str().unwrap();
    let spans = hit["snippet_spans"].as_array().unwrap();
    assert!(!spans.is_empty());
    assert!(spans.len() <= 5);
    for s in spans {
        let (start, end) = (s["start"].as_u64().unwrap() as usize, s["end"].as_u64().unwrap() as usize);
        let chars: Vec<char> = snippet.chars().collect();
        assert!(start < end && end <= chars.len());
    }
    // At least one span covers the matched term.
    let chars: Vec<char> = snippet.chars().collect();
    assert!(spans.iter().any(|s| {
        let (a, b) = (s["start"].as_u64().unwrap() as usize, s["end"].as_u64().unwrap() as usize);
        chars[a..b].iter().collect::<String>().to_lowercase() == "quick"
    }));
}

#[test]
fn search_json_spans_empty_for_image_hits() {
    let (env, cfg) = indexed_env_with_image();
    let v = run_search_json(&cfg, "photo", &["--kind", "image"]);
    for h in v["hits"].as_array().unwrap() {
        assert_eq!(h["snippet_spans"], serde_json::json!([]));
    }
}
```

(Adapt harnesses to `tests/search_json.rs` conventions. If the image search returns zero hits in the fixture, seed via the image-index test fixture used in `tests/index_images.rs` and assert on whatever hits return — spans must be `[]` regardless.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test search_json snippet_spans && cargo test --test search_json spans_empty`
Expected: FAIL (`snippet_spans` key missing).

- [ ] **Step 3: Write minimal implementation**

In the FTS arm (`src/search/sql.rs`), select `offsets(chunks_fts)` alongside existing columns; parse the quadruple string, keep entries whose column index is the chunk-text column, take byte (offset,length), map to char indices in the full chunk text, then intersect with the snippet window the existing builder produced (implementer: reuse the builder's window variables — do not rebuild the snippet). Clamp, cap at 5, attach to the hit struct; renderer emits them verbatim. Non-text hits: empty vec.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test search_json`
Expected: PASS (frozen-contract tests unaffected — new key only). Then: `cargo test` full suite green.

- [ ] **Step 5: Amend docs + commit**

Spec §8 + README: `snippet_spans` schema (char offsets into `snippet`, ≤5, `[]` when unsupported), FTS-sourced. Then:

```bash
git add src/search/sql.rs src/search/json.rs src/search/mod.rs tests/search_json.rs docs/specs/ferret-design.md README.md
git commit -m "feat: snippet highlight spans in search hits"
```

---

### Task 7: Offline-keyword guarantee + cancellation contract

**Files:**
- Modify: `README.md` (new "Offline & cancellation" section), `docs/specs/ferret-design.md` (mode semantics note)
- Test: `tests/search_filters.rs` (or `tests/cli_maintenance.rs` — wherever a dead-endpoint search test fits; prefer CLI level via `try_run`)

**Interfaces:**
- Consumes: NO code change expected — `SearchMode::Keyword` already skips `embed()` (`want_vec=false`). This task pins the behavior and documents it. If the test exposes a CLI pre-flight health check that hits the network on `search` (implementer: check `src/cli.rs` search path for `ensure_up`/`check_health` before dispatch), remove that pre-flight for `search` (health is proven per-call by embed errors) as part of this task.
- Produces: (a) passing test: keyword search with a dead endpoint returns exit 0 + hits; hybrid with a dead endpoint returns exit 1. (b) README "Offline & cancellation" section: keyword mode needs no embedder; killing `search` is always safe (read-only); killing `index` then rerunning resumes via `complete` rows and PDF `next_page` cursors (partial files re-embed, committed pages stay); never run two `index` concurrently.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
#[serial]
fn keyword_search_works_with_dead_endpoint() {
    let (env, cfg) = indexed_env_with_text(&["the quick brown fox"]);
    // Point the endpoint at a closed port; keyword must never dial it.
    let mut dead = cfg;
    dead.embedder.endpoint = "http://127.0.0.1:9".into();
    let v = run_search_json_with_config(&dead, "quick", &["--mode", "keyword"]);
    assert!(!v["hits"].as_array().unwrap().is_empty());
}

#[test]
#[serial]
fn hybrid_search_fails_with_dead_endpoint() {
    let (env, cfg) = indexed_env_with_text(&["the quick brown fox"]);
    let mut dead = cfg;
    dead.embedder.endpoint = "http://127.0.0.1:9".into();
    let err = run_search_json_err(&dead, "quick", &["--mode", "hybrid"]);
    assert_eq!(err.exit_code(), 1);
}
```

(Adapt to existing harnesses: build the index with `FakeEmbedder` first, then swap the endpoint. `127.0.0.1:9` is the discard port — connection refused, no hang. If the harness drives `try_run`, assert on its `Error::exit_code()`. `#[serial]` guards any process-env manipulation the harness needs.)

- [ ] **Step 2: Run tests to verify behavior**

Run: `cargo test keyword_search_works_with_dead_endpoint hybrid_search_fails_with_dead_endpoint`
Expected: keyword test PASSES already (behavior exists — this pins it); hybrid test passes. If the keyword test FAILS (CLI pre-flights health), proceed to Step 3; otherwise skip to Step 4.

- [ ] **Step 3: Remove search pre-flight (only if Step 2 failed)**

In `src/cli.rs` search path, remove the `ensure_up`/`check_health` call before dispatch (embed-time errors already surface as exit 1 with the endpoint named). Minimal diff, no other behavior change.

- [ ] **Step 4: Write docs + run full suite**

README "Offline & cancellation" section + spec mode-semantics note (exact wording from Interfaces/Produces above). Run: `cargo test` full suite green.

- [ ] **Step 5: Commit**

```bash
git add tests/ README.md docs/specs/ferret-design.md src/cli.rs
git commit -m "docs: pin offline keyword search, document cancellation contract"
```

(If Step 3 was skipped, `src/cli.rs` won't be in the diff — commit what's changed.)

---

## Self-Review

**1. Spec coverage** (v1.1 scope from the evaluation: plan event, summary event, roots in status, warnings array, `--only`, snippet spans, keyword pin + cancellation docs):
- plan event → Task 1 ✓
- summary event → Task 2 ✓
- roots in status → Task 3 ✓
- warnings (`vec_starved`, `browse_no_vec`) → Task 4 ✓
- `index --only` → Task 5 ✓
- snippet spans → Task 6 ✓
- keyword pin + cancellation contract → Task 7 ✓
- Unknown-event tolerance rule → Task 1 docs ✓
- Out of scope (v1.2+ candidates, NOT in this plan): `doctor`, `init --with-server`, public library API, move-heuristic for renames, thumbnail API (won't-fix by design).
- Audio/video milestones (specced separately; runtime-proven 2026-10-13, see progress ledger): audio = new typed-items dialect + audio kinds + ~30s segment chunking + ffmpeg wrap, acceptance gated on natural-sound discrimination (synthetic tones do NOT discriminate); video = native `input_video` via llama dialect (proven discriminating over `/v1/embeddings`), frames-as-images fallback needs no server support. Lab wrapper still needs `video` added to `/v1/embed` types; SGLang audio needs native `/encode` alongside `/v1/embeddings`.

**2. Placeholder scan:** every step carries exact JSON shapes, exact warning codes + trigger rules, exact commands, real test bodies, real commit messages. Task 5/6 name the files/functions to read where the implementer must meet existing code (`index::run` params, snippet builder window) — these are navigation pointers, not placeholders, and each is paired with a verbatim assertion block.

**3. Type consistency:** NDJSON events share `{"event": ...}` string-discriminated shape (Tasks 1–2); search additions are top-level `warnings: string[]` (Task 4) and per-hit `snippet_spans: {start,end}[]` with `usize`-as-JSON-number (Task 6); status `roots: {path,collection}[]` mirrors `RootEntry` (Task 3). No cross-task renames.

---

## Execution Handoff

Plan complete and saved to `docs/plans/ferret-v1.1.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session with checkpoints for review.

Which approach?
