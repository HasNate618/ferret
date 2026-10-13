# Task 4 Report: Appendix A config fingerprint

**Status:** DONE
**Commit:** `5aa339f` (`feat: hash split text/image pipeline fingerprints`)

## Implemented behavior

`src/fingerprint.rs` implements the Appendix A config fingerprint (spec
`docs/specs/ferret-design.md` §13 / Appendix A):

- `config_fingerprint_json(cfg, images_supported) -> String` — canonical
  JSON: UTF-8, object keys sorted lexicographically, no insignificant
  whitespace (compact `serde_json`), built by routing every object through a
  `BTreeMap` into `serde_json::Map`, so output is deterministic even if a
  `preserve_order` feature ever lands. Exact field set:
  - Top level: `backend`, `dimensions`, `embed_recipe`, `extension_kind`
    (full merged no-dot lower map, every compiled default), `image_pipeline`,
    `model_id` (= `embedder.model`), `prefix_profile`, `prefix_version`,
    `text_pipeline`.
  - `text_pipeline`: `max_chars`, `mode`, `numeric_heavy`, `overlap`,
    `pdf: { embed_text, first_pass_pages, max_pages }`, `skip_letterless`.
  - `image_pipeline`: `heic`, `images_supported`, `max_edge_px`,
    `pdf: { embed_page_image, render_dpi }`.
  - Deliberately absent (walk filters / status knobs): `include_globs`,
    `exclude_globs`, `include_hidden`, `respect_gitignore`, `max_size`,
    `follow_symlinks`, all `search.*` / `extract.*` / endpoint / batching.
- `config_fingerprint(cfg, images_supported) -> String` — hex blake3 of the
  canonical JSON (64 chars).
- `text_pipeline_json(cfg)` / `image_pipeline_json(cfg, images_supported)`
  — standalone canonical-JSON halves for store meta.
- `file_fingerprint(canonical_path, mtime, size, config_fingerprint) ->
  String` — blake3 over `path || 0x1f || mtime || 0x1f || size || 0x1f ||
  config_fingerprint` (streamed `Hasher`, ASCII numbers), hex.
- Constants `EMBED_RECIPE = "v1"` and `PREFIX_VERSION = "v1"`.

Verified byte-for-byte against the Appendix A example (modulo the one-key
stub → full merged `extension_kind` map) via a throwaway example binary
since removed.

## Files changed

- `Cargo.toml` — added `blake3 = "1"`.
- `src/fingerprint.rs` — new module (implementation + `#[cfg(test)]` unit
  tests, brief's test code verbatim).
- `src/lib.rs` — registered `pub mod fingerprint;`.
- `Cargo.lock` — blake3 1.8.7 (+ transitive `constant_time_eq`, `cpufeatures`).
- `docs/plans/tasks/task-4-report.md` — this report (untracked, like the
  other docs/plans files).

## Test results

TDD followed: brief's tests written first, run to failure, then implemented.

- `cargo test --lib fingerprint` (pre-implementation): FAIL — `cannot find
  function config_fingerprint_json in this scope` (expected failure step).
- `cargo test --lib fingerprint` (post-implementation):
  ```
  running 3 tests
  test fingerprint::tests::file_fingerprint_includes_unit_separator ... ok
  test fingerprint::tests::canonical_json_key_order_and_split_pipelines ... ok
  test fingerprint::tests::fingerprint_is_stable_hex ... ok
  test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
  ```
- `cargo test` (full suite): all pass — lib 12, config_merge 16, paths 3,
  cli_init 5, doc-tests 0; 0 failed.
- `cargo clippy --all-targets`: clean, no warnings.
- Canonical JSON spot-check (defaults, images_supported=true) begins:
  `{"backend":"llama_server","dimensions":768,"embed_recipe":"v1",...`
  with `image_pipeline.pdf = {"embed_page_image":true,"render_dpi":120}` and
  `text_pipeline.pdf = {"embed_text":true,"first_pass_pages":30,"max_pages":5000}`.

## Self-review findings and fixes

- **Commit omitted `Cargo.lock`**: the brief's `git add` list did not include
  it, but every prior task commit did (repo convention for a binary crate).
  Fixed by amending my own commit (`9782758` → `5aa339f`) to include the
  updated lockfile, leaving a clean tracked tree.
- **`cargo fmt --check` fails repo-wide**, including files from Tasks 1-3
  (import ordering, line width). The committed style predates rustfmt's
  conventions, so I did not reformat anything (out of scope) and matched the
  existing hand-written style in the new file. Flagging for a possible
  one-off repo-wide `cargo fmt` decision by the orchestrator.
- **Test helper**: the brief's test calls `compiled_defaults(&test_dirs())`;
  `tests/common.rs::test_dirs` takes a `&Path` arg and integration-only. I
  added a zero-arg local `test_dirs()` in the `#[cfg(test)]` module building
  `FerretDirs` from static paths — `compiled_defaults` only reads
  `dirs.data_dir`, so no filesystem access is needed.

## Commit hashes

- `5aa339f` — feat: hash split text/image pipeline fingerprints
  (amended from `9782758` to include `Cargo.lock`, per repo convention)
