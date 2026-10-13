### Task 9: Text extraction and chunking

**Files:**
- Create: `src/extract/mod.rs`, `src/extract/text.rs`
- Test: `tests/extract_text.rs`, `testdata/two_sentences.txt`, `testdata/numeric.txt`, `testdata/letterless.txt`, `testdata/sample.html`
- Modify: `Cargo.toml` (`unicode-segmentation`, `html-escape`)

**Interfaces:**
- Consumes: `Config.chunk.text`, `FileKind`
- Produces:

```rust
pub struct Chunk {
    pub kind: ChunkKind,
    pub text: Option<String>,
    pub start_byte: Option<i64>,
    pub end_byte: Option<i64>,
    pub page: Option<i64>,
    pub embed: EmbedDecision,
}

pub enum EmbedDecision {
    Vec,      // write chunks + fts + vec
    FtsOnly,  // numeric_heavy: chunks + fts, no vec
    Drop,     // skip_letterless: do not emit
}

pub fn extract_text(path: &Path, bytes: &[u8], cfg: &TextChunkConfig) -> Result<Vec<Chunk>>;
```

HTML (`ext` html/htm): strip tags, decode entities, then same windows. JSON/YAML/TOML: raw UTF-8, no pretty-print. `svg` is text.

`mode=chars`: sliding windows of `max_chars` Unicode scalars, overlap `overlap`, prefer break on newline then space. Store UTF-8 byte offsets.

`mode=sentences`: `unicode_sentences()`, pack until `max_chars`; one over-long sentence falls back to char windows (with overlap) for that sentence only.

`skip_letterless`: if no `char::is_alphabetic`, **Drop**.

`numeric_heavy`: let `n` = non-whitespace scalars, `d` = ASCII digits among those; if `n>0` and `d as f64 / n as f64 >= numeric_heavy` (and numeric_heavy != 0) → `FtsOnly`.

If `overlap >= max_chars`, config already rejected.

Empty/whitespace file → empty vec (caller still writes `files` complete=1).

Title for document prefix is basename — that is index's job, not extract.

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn sentences_split_when_chars_would_not() {
    let cfg = TextChunkConfig { mode: TextChunkMode::Sentences, max_chars: 2000, overlap: 200, skip_letterless: true, numeric_heavy: 0.7 };
    let chunks = extract_text(Path::new("x.txt"), b"Hello world. Goodbye moon.", &cfg).unwrap();
    assert_eq!(chunks.len(), 2);
}

#[test]
fn numeric_heavy_is_fts_only() {
    let cfg = TextChunkConfig { mode: TextChunkMode::Chars, max_chars: 2000, overlap: 200, skip_letterless: true, numeric_heavy: 0.7 };
    let chunks = extract_text(Path::new("n.txt"), b"1234567890 12345", &cfg).unwrap();
    assert!(matches!(chunks[0].embed, EmbedDecision::FtsOnly));
}

#[test]
fn letterless_dropped() {
    let cfg = TextChunkConfig { mode: TextChunkMode::Chars, max_chars: 2000, overlap: 200, skip_letterless: true, numeric_heavy: 0.7 };
    let chunks = extract_text(Path::new("x.txt"), b"***\n---\n", &cfg).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn html_strips_tags() {
    let cfg = TextChunkConfig { mode: TextChunkMode::Chars, max_chars: 2000, overlap: 200, skip_letterless: true, numeric_heavy: 0.7 };
    let chunks = extract_text(Path::new("a.html"), b"<p>Hi&amp;bye</p>", &cfg).unwrap();
    assert_eq!(chunks[0].text.as_deref(), Some("Hi&bye"));
}
```

Wait: `&amp;` decodes to `&`, so text is `Hi&bye`. Offsets are into the **original file** bytes, not the stripped string — spec: `start_byte` / `end_byte` in the original file (UTF-8 byte offsets). For HTML, offsets into the original file covering the extracted region are best-effort: use the span of the remaining text mapped back if cheap; if mapping is messy, store offsets into the stripped buffer **only if** we document that — spec says original file. Implement a simple tag stripper that records byte ranges of kept scalars in the original buffer.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test extract_text`
Expected: FAIL

- [ ] **Step 3: Implement**

Do not use a full browser HTML parser. Sequential scan: drop `<...>` including comments/`<script>`/`<style>` inner text; `html_escape::decode_html_entities`.

- [ ] **Step 4: Run tests**

Run: `cargo test --test extract_text`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/extract testdata tests/extract_text.rs Cargo.toml
git commit -m "$(cat <<'EOF'
feat: extract and chunk text with sentence and numeric-heavy modes

EOF
)"
```

---

