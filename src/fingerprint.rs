//! Appendix A config fingerprint: canonical JSON of exactly the config that
//! changes how a given file is chunked and embedded, hashed with blake3.
//!
//! Walk filters (`files.include_globs`, `exclude_globs`, `include_hidden`,
//! `respect_gitignore`) and status knobs like `files.max_size` are
//! deliberately absent: they change *which* paths are walked, not how a
//! walked file is processed (spec §13, Appendix A).
//!
//! Canonical JSON: UTF-8, keys sorted lexicographically, no insignificant
//! whitespace — built via `BTreeMap` into a compact `serde_json` object.

use crate::config::Config;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Embedding-recipe revision (Appendix B). Bump when frozen HTTP bodies change.
pub const EMBED_RECIPE: &str = "v1";
/// Prefix-template revision (spec §3.2). Bump when prefixes change.
pub const PREFIX_VERSION: &str = "v1";

/// Hex blake3 of the canonical Appendix A JSON.
pub fn config_fingerprint(cfg: &Config, images_supported: bool) -> String {
    blake3::hash(config_fingerprint_json(cfg, images_supported).as_bytes())
        .to_hex()
        .to_string()
}

/// Canonical Appendix A JSON (compact, sorted keys, exact field set).
pub fn config_fingerprint_json(cfg: &Config, images_supported: bool) -> String {
    to_canonical_json(config_fingerprint_value(cfg, images_supported))
}

/// The text half of the fingerprint, standalone for store meta: everything
/// that changes text chunking or PDF text embedding.
pub fn text_pipeline_json(cfg: &Config) -> String {
    to_canonical_json(text_pipeline_value(cfg))
}

/// The image half of the fingerprint, standalone for store meta: everything
/// that changes image handling or PDF page-image rendering.
pub fn image_pipeline_json(cfg: &Config, images_supported: bool) -> String {
    to_canonical_json(image_pipeline_value(cfg, images_supported))
}

/// `blake3(canonical_path || 0x1f || mtime || 0x1f || size || 0x1f ||
/// config_fingerprint)` as hex; separators make the concatenation unambiguous.
pub fn file_fingerprint(
    canonical_path: &str,
    mtime: i64,
    size: u64,
    config_fingerprint: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(canonical_path.as_bytes());
    hasher.update(&[0x1f]);
    hasher.update(mtime.to_string().as_bytes());
    hasher.update(&[0x1f]);
    hasher.update(size.to_string().as_bytes());
    hasher.update(&[0x1f]);
    hasher.update(config_fingerprint.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn config_fingerprint_value(cfg: &Config, images_supported: bool) -> Value {
    let extension_kind = object(
        cfg.files
            .extension_kind
            .iter()
            .map(|(ext, kind)| (ext.clone(), Value::String(kind.to_string())))
            .collect(),
    );
    object(vec![
        ("backend".into(), Value::String(cfg.embedder.backend.to_string())),
        ("dimensions".into(), Value::from(cfg.embedder.dimensions)),
        ("embed_recipe".into(), Value::String(EMBED_RECIPE.to_string())),
        ("extension_kind".into(), extension_kind),
        (
            "image_pipeline".into(),
            image_pipeline_value(cfg, images_supported),
        ),
        ("model_id".into(), Value::String(cfg.embedder.model.clone())),
        (
            "prefix_profile".into(),
            Value::String(cfg.embedder.prefix_profile.to_string()),
        ),
        ("prefix_version".into(), Value::String(PREFIX_VERSION.to_string())),
        ("text_pipeline".into(), text_pipeline_value(cfg)),
    ])
}

/// Text pipeline: text chunking plus PDF text embedding knobs.
fn text_pipeline_value(cfg: &Config) -> Value {
    let pdf = object(vec![
        ("embed_text".into(), Value::Bool(cfg.chunk.pdf.embed_text)),
        ("first_pass_pages".into(), Value::from(cfg.chunk.pdf.first_pass_pages)),
        ("max_pages".into(), Value::from(cfg.chunk.pdf.max_pages)),
    ]);
    object(vec![
        ("max_chars".into(), Value::from(cfg.chunk.text.max_chars)),
        ("mode".into(), Value::String(cfg.chunk.text.mode.to_string())),
        ("numeric_heavy".into(), Value::from(cfg.chunk.text.numeric_heavy)),
        ("overlap".into(), Value::from(cfg.chunk.text.overlap)),
        ("pdf".into(), pdf),
        ("skip_letterless".into(), Value::Bool(cfg.chunk.text.skip_letterless)),
    ])
}

/// Image pipeline: image handling plus PDF page-image rendering knobs.
/// `render_dpi` lives here (not in the text pipeline) because changing only
/// `max_chars` must not re-render page images (spec §13).
fn image_pipeline_value(cfg: &Config, images_supported: bool) -> Value {
    let pdf = object(vec![
        ("embed_page_image".into(), Value::Bool(cfg.chunk.pdf.embed_page_image)),
        ("render_dpi".into(), Value::from(cfg.chunk.pdf.render_dpi)),
    ]);
    object(vec![
        ("heic".into(), Value::String(cfg.files.heic.to_string())),
        ("images_supported".into(), Value::Bool(images_supported)),
        ("max_edge_px".into(), Value::from(cfg.chunk.image.max_edge_px)),
        ("pdf".into(), pdf),
    ])
}

/// A `serde_json` object with keys sorted lexicographically: route every map
/// through a `BTreeMap` so output is deterministic regardless of feature flags.
fn object(entries: Vec<(String, Value)>) -> Value {
    let sorted: BTreeMap<String, Value> = entries.into_iter().collect();
    Value::Object(Map::from_iter(sorted))
}

fn to_canonical_json(value: Value) -> String {
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compiled_defaults;
    use crate::paths::FerretDirs;
    use std::path::PathBuf;

    fn test_dirs() -> FerretDirs {
        FerretDirs {
            config_dir: PathBuf::from("/t/cfg"),
            data_dir: PathBuf::from("/t/data"),
            cache_dir: PathBuf::from("/t/cache"),
        }
    }

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
}
