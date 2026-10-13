### Task 4: Appendix A config fingerprint

**Files:**
- Create: `src/fingerprint.rs`
- Test: unit tests in `src/fingerprint.rs` under `#[cfg(test)]`
- Modify: `Cargo.toml` (`blake3 = "1"`), `src/lib.rs`

**Interfaces:**
- Consumes: `Config`, `images_supported: bool`
- Produces:
  - `fingerprint::config_fingerprint(cfg: &Config, images_supported: bool) -> String` (hex blake3)
  - `fingerprint::config_fingerprint_json(cfg: &Config, images_supported: bool) -> String` (canonical JSON)
  - `fingerprint::file_fingerprint(canonical_path: &str, mtime: i64, size: u64, config_fingerprint: &str) -> String`
  - `fingerprint::text_pipeline_json` / `image_pipeline_json` helpers used by store meta
  - Constants `EMBED_RECIPE: &str = "v1";` `PREFIX_VERSION: &str = "v1";`

Canonical JSON: UTF-8, object keys sorted lexicographically, no insignificant whitespace. Implement by building `serde_json::Value::Object(Map)` from a `BTreeMap`, then `to_string()` (serde_json compact).

Field set **exactly** as Appendix A. `extension_kind` is the **full** merged map (no-dot, lower). `files.include_globs` / `include_hidden` / `respect_gitignore` / `exclude_globs` are **not** in this JSON.

`file_fingerprint = blake3(canonical_path || 0x1f || mtime ascii || 0x1f || size ascii || 0x1f || config_fingerprint)` hex.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn canonical_json_key_order_and_split_pipelines() {
    let cfg = compiled_defaults(&test_dirs());
    let json = config_fingerprint_json(&cfg, true);
    assert!(json.contains("\"image_pipeline\":{"));
    assert!(json.contains("\"text_pipeline\":{"));
    assert!(!json.contains("include_globs"));
    assert!(json.contains("\"prefix_profile\":\"embeddinggemma2\""));
    // image_pipeline.pdf has embed_page_image + render_dpi only
    assert!(json.contains("\"embed_page_image\":true"));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let img_pdf = &v["image_pipeline"]["pdf"];
    assert!(img_pdf.get("embed_text").is_none());
    assert!(img_pdf.get("max_pages").is_none());
    let txt_pdf = &v["text_pipeline"]["pdf"];
    assert!(txt_pdf.get("embed_page_image").is_none());
    assert_eq!(txt_pdf["max_pages"], 5000);
}

#[test]
fn fingerprint_is_stable_hex() {
    let cfg = compiled_defaults(&test_dirs());
    let a = config_fingerprint(&cfg, true);
    let b = config_fingerprint(&cfg, true);
    assert_eq!(a, b);
    assert_eq!(a.len(), 64);
    assert_ne!(a, config_fingerprint(&cfg, false));
}

#[test]
fn file_fingerprint_includes_unit_separator() {
    let fp = file_fingerprint("/a/b.txt", 1, 2, "abcd");
    let expected = blake3::hash(b"/a/b.txt\x1f1\x1f2\x1fabcd").to_hex().to_string();
    assert_eq!(fp, expected);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib fingerprint`
Expected: FAIL

- [ ] **Step 3: Implement `src/fingerprint.rs`**

Build the JSON object with keys: `backend`, `dimensions`, `embed_recipe`, `extension_kind`, `image_pipeline`, `model_id`, `prefix_profile`, `prefix_version`, `text_pipeline`.

`image_pipeline`: `{ heic, images_supported, max_edge_px, pdf: { embed_page_image, render_dpi } }`

`text_pipeline`: `{ max_chars, mode, numeric_heavy, overlap, pdf: { embed_text, first_pass_pages, max_pages }, skip_letterless }`

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/fingerprint.rs src/lib.rs
git commit -m "$(cat <<'EOF'
feat: hash split text/image pipeline fingerprints

EOF
)"
```

---

