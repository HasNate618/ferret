# Task 6 Report: Embedder trait, prefix profiles, fake embedder

**Status:** DONE_WITH_CONCERNS (behavior complete, tested, and committed; three
brief-internal gaps were resolved with the supervisor's explicit approval
before any edit — details below)
**Commit:** `ffc3cf2` (`feat: add embedder prefixes and fake unit-vector backend`)

## Implemented behavior

- `src/embedder/mod.rs` (new):
  - `EmbedInput` (Query / Document / Image / PdfPage) and the `Embedder`
    trait (`embed`, default `embed_batch` iterating `embed`, and
    `images_supported`) transcribed verbatim from the brief.
  - `build_embedder(cfg, env)`: `env("FERRET_EMBEDDER") == Some("fake")`
    returns the fake embedder **regardless of `embedder.backend`** (fake
    reports `images_supported = true`). Without the fake hook it returns
    `Error::fatal("http embedder not implemented")` until Task 12.
  - `l2_normalize(v)`: errors on any non-finite component or a zero norm;
    otherwise divides by the L2 norm (only finite `f32`, unit length).
  - `truncate_or_check(v, backend, dimensions)`: `openai`/`sglang` must be
    exactly `dimensions` (wrong length or non-finite → fatal). `llama_server`
    / `litert_lm`: longer than `dimensions` → truncated then L2-renormalized;
    shorter, or any non-finite → fatal; exact length passes through.
- `src/embedder/prefix.rs` (new): `apply_prefix(profile, input)` with the
  exact strings from the brief —
  - embeddinggemma2: query search `task: search result | query: {text}`,
    query code `task: code retrieval | query: {text}`, document
    `title: {title or none} | text: {text}`;
  - litert_lm: query search `task: search query | text: {text}`, query code
    `task: code retrieval | query: {text}`, document
    `task: search result | text: {text}`;
  - Image-only → `None` (no text prefix).
  - Unit tests: the brief's four verbatim plus code-query (both profiles),
    document-with-`none`-title, litert document, PdfPage-with-text (both
    profiles), textless PdfPage → `None`.
- `src/embedder/fake.rs` (new): `FakeEmbedder { profile, dimensions }`.
  - `embed`: vector = first `dimensions` bytes of
    `blake3(prefix || 0x1f || payload)` as `u8 as f32`, then L2-normalized
    (always finite, unit length). blake3's **extended output** is used
    because `dimensions` is 128–768, far beyond the 32-byte digest.
  - Payload per kind (approved resolution): Query/Document → UTF-8 text;
    Image → raw bytes; PdfPage → `image_png || 0x1f || text-or-empty`, so a
    textless PdfPage never collides with an Image of the same bytes and every
    input kind yields a distinct vector.
  - `FakeEmbedder::new(8)` (pinned test signature) defaults the profile to
    `EmbeddingGemma2`; `new_with_profile(profile, dimensions)` is what
    `build_embedder` wires from `cfg.embedder.prefix_profile`, so fake vectors
    follow the active prefix strings (and search-vs-code queries differ).
  - `images_supported() = true` (test hook is multimodal by definition).
- `src/lib.rs`: `pub mod embedder;` (mod.rs re-exports `apply_prefix` and
  `FakeEmbedder`).
- `src/config.rs`: comment-only touch — the header now says `FERRET_EMBEDDER`
  is read by `embedder::build_embedder` via its `env` parameter (it was
  already declared "never here"; the brief's "Modify: src/config.rs (read
  FERRET_EMBEDDER)" is satisfied by the embedder module, approved).

## Supervisor-approved ambiguity resolutions (pre-edit)

1. `apply_prefix` on `PdfPage`: `text: Some(t)` → the profile's **Document**
   prefix (title `"none"`); `text: None` → `None`. Grounded in
   `docs/specs/ferret-design.md` §3.2 ("Text part uses the Document prefix of
   the active profile").
2. Fake payload composition per kind (see above), guaranteeing text vs image
   vs pdf-page inputs always differ deterministically.
3. `FakeEmbedder::new(8)` defaulting to `EmbeddingGemma2` +
   `new_with_profile` wired from the merged config; config.rs comment-only.

## Files changed

- `src/embedder/mod.rs` (new) — trait, `EmbedInput`, `build_embedder`,
  `l2_normalize`, `truncate_or_check`.
- `src/embedder/prefix.rs` (new) — `apply_prefix` + 12 unit tests.
- `src/embedder/fake.rs` (new) — `FakeEmbedder`.
- `src/lib.rs` — `pub mod embedder;`.
- `src/config.rs` — doc-comment touch only (no behavior change).
- `tests/embedder_fake.rs` (new) — 9 integration tests.

## Test results

Red phase verified first (`cargo test --lib embedder` → FAILED, 8 failed /
2 passed, with stub `apply_prefix`/`embed`), then:

```
$ cargo test
test result: ok. 16 passed; 0 failed  (src lib: types + config + prefix)
test result: ok. 15 passed; 0 failed  (tests/store_schema.rs)
test result: ok. 12 passed; 0 failed  (tests/config_merge.rs)
test result: ok.  9 passed; 0 failed  (tests/embedder_fake.rs)
test result: ok.  9 passed; 0 failed  (tests/cli_init.rs + others: paths 3,
                                      fingerprint, …)
all suites: ok, 0 failed
$ cargo clippy --all-targets
Finished `dev` profile — no warnings
```

(Tail: every `test result:` line is `ok`; 0 failed across all targets.)

## Self-review findings and fixes

- Compile fix during red phase: prefix.rs tests needed
  `use super::super::Embedder` for the `embed` call (trait method scope).
- `tests/embedder_fake.rs`: `unwrap_err()` on `Box<dyn Embedder>` needs
  `Debug` on the `Ok` type — replaced with an explicit `match` + `panic!`.
- Confirmed blake3's plain 32-byte digest cannot cover `dimensions` 128–768;
  used `finalize_xof()` (extended output), whose first bytes equal the
  standard hash, matching the brief's "first `dimensions` bytes".
- Formatting: the repo is not `cargo fmt`-clean from prior tasks (config.rs,
  fingerprint.rs, store, existing tests all show diffs), and the brief's test
  code is pinned verbatim (long lines). Left formatting consistent with
  Tasks 1–5 rather than reformatting unrelated files. `cargo clippy` is clean.

## Concerns / notes for later tasks

- `build_embedder` is fatal without `FERRET_EMBEDDER=fake` until Task 12
  (HTTP backends) — by design, per the brief.
- Mixing prefix profiles in one index is a hard error, but that check lives
  in store/meta (Task 5 already implements it) — this task only provides the
  prefix functions.
- Capability probe (`embedder.multimodal` omitted → probe once) lands with
  the HTTP backends in Task 12, per the brief; not implemented here.
