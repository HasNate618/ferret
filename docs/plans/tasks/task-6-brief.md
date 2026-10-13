### Task 6: Embedder trait, prefixes, fake embedder

**Files:**
- Create: `src/embedder/mod.rs`, `src/embedder/prefix.rs`, `src/embedder/fake.rs`
- Modify: `src/lib.rs`, `src/config.rs` (read `FERRET_EMBEDDER`)
- Test: `src/embedder/prefix.rs` unit tests + `tests/embedder_fake.rs`

**Interfaces:**
- Consumes: `Config`, `QueryTask`, `PrefixProfile`
- Produces:

```rust
pub enum EmbedInput {
    Query { text: String, task: QueryTask },
    Document { title: Option<String>, text: String },
    Image { bytes: Vec<u8>, mime: String },
    PdfPage { image_png: Vec<u8>, text: Option<String> },
}

pub trait Embedder: Send + Sync {
    fn embed(&self, input: EmbedInput) -> Result<Vec<f32>>;
    fn embed_batch(&self, inputs: &[EmbedInput]) -> Result<Vec<Vec<f32>>> {
        inputs.iter().cloned().map(|i| self.embed(i)).collect()
    }
    fn images_supported(&self) -> bool;
}

pub fn build_embedder(cfg: &Config, env: &dyn Fn(&str) -> Option<String>) -> Result<Box<dyn Embedder>>;
pub fn apply_prefix(profile: PrefixProfile, input: &EmbedInput) -> Option<String>;
pub fn l2_normalize(v: Vec<f32>) -> Result<Vec<f32>>;
pub fn truncate_or_check(v: Vec<f32>, backend: Backend, dimensions: u32) -> Result<Vec<f32>>;
```

`apply_prefix` returns `None` for image-only (no text prefix). Document prefix uses `"none"` when title is `None`.

**embeddinggemma2:**
- Query search: `task: search result | query: {text}`
- Query code: `task: code retrieval | query: {text}`
- Document: `title: {title or none} | text: {text}`

**litert_lm:**
- Query search: `task: search query | text: {text}`
- Query code: `task: code retrieval | query: {text}`
- Document: `task: search result | text: {text}`

Fake embedder: `images_supported = true`. Vector = first `dimensions` bytes of `blake3(prefix || 0x1f || payload)` interpreted as `u8 as f32`, then L2-normalize. Payload is UTF-8 text or raw image bytes. Always finite. `build_embedder` if `env("FERRET_EMBEDDER") == Some("fake")` returns fake **regardless of backend**.

HTTP embedder in this task: **not yet**. `build_embedder` without fake may return `Error::fatal("http embedder not implemented")` until Task 12.

`truncate_or_check`: openai/sglang must already be `dimensions` (caller sends `dimensions` in JSON later). llama_server/litert_lm: if `len > dimensions`, truncate then L2-renormalize; if `len < dimensions` or any non-finite → error.

- [ ] **Step 1: Write failing prefix tests**

```rust
#[test]
fn gemma2_search_query_prefix() {
    let s = apply_prefix(
        PrefixProfile::EmbeddingGemma2,
        &EmbedInput::Query { text: "pets".into(), task: QueryTask::Search },
    );
    assert_eq!(s.as_deref(), Some("task: search result | query: pets"));
}

#[test]
fn litert_search_query_prefix() {
    let s = apply_prefix(
        PrefixProfile::LitertLm,
        &EmbedInput::Query { text: "pets".into(), task: QueryTask::Search },
    );
    assert_eq!(s.as_deref(), Some("task: search query | text: pets"));
}

#[test]
fn image_only_has_no_text_prefix() {
    let s = apply_prefix(
        PrefixProfile::EmbeddingGemma2,
        &EmbedInput::Image { bytes: vec![1], mime: "image/png".into() },
    );
    assert!(s.is_none());
}

#[test]
fn fake_is_unit_length_and_deterministic() {
    let e = FakeEmbedder::new(8);
    let a = e.embed(EmbedInput::Document { title: Some("a.txt".into()), text: "hi".into() }).unwrap();
    let b = e.embed(EmbedInput::Document { title: Some("a.txt".into()), text: "hi".into() }).unwrap();
    assert_eq!(a, b);
    let n = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((n - 1.0).abs() < 1e-5);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib embedder`
Expected: FAIL

- [ ] **Step 3: Implement prefix + fake + normalize**

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedder tests/embedder_fake.rs src/lib.rs
git commit -m "$(cat <<'EOF'
feat: add embedder prefixes and fake unit-vector backend

EOF
)"
```

---

