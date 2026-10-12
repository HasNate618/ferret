//! Embedder front end: typed inputs, the `Embedder` trait, L2
//! normalization / dimension enforcement, and `build_embedder` with the
//! `FERRET_EMBEDDER=fake` test hook. HTTP backends are served by
//! [`HttpEmbedder`] (Task 12); `openai` / `sglang` construct but their
//! `embed` stays fatal until Task 16.

pub mod fake;
pub mod http;
pub mod prefix;
pub mod probe;

pub use fake::FakeEmbedder;
pub use http::HttpEmbedder;
pub use prefix::apply_prefix;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::types::{Backend, QueryTask};

/// One typed embedding input per modality (spec §3.1). Callers pass a typed
/// input so prefixes and modalities stay consistent.
#[derive(Debug, Clone)]
pub enum EmbedInput {
    /// Search/code query; `task` selects the query prefix.
    Query { text: String, task: QueryTask },
    /// Text document; `title` is the basename, `"none"` in the prefix when
    /// unknown.
    Document { title: Option<String>, text: String },
    /// Image-only input; no text prefix.
    Image { bytes: Vec<u8>, mime: String },
    /// One PDF page: image part plus optional page text, one vector.
    PdfPage { image_png: Vec<u8>, text: Option<String> },
}

/// An embedding backend. Every output is a single L2-normalized vector of
/// `embedder.dimensions` finite `f32`s.
pub trait Embedder: Send + Sync {
    fn embed(&self, input: EmbedInput) -> Result<Vec<f32>>;
    fn embed_batch(&self, inputs: &[EmbedInput]) -> Result<Vec<Vec<f32>>> {
        inputs.iter().cloned().map(|i| self.embed(i)).collect()
    }
    fn images_supported(&self) -> bool;
}

/// Build the embedder for the merged config. `FERRET_EMBEDDER=fake` wins over
/// `embedder.backend` (any backend) and the fake reports
/// `images_supported = true`. Otherwise an [`HttpEmbedder`] is built and
/// health-checked (spawning the managed server first when configured): a
/// down embedder is fatal here, before any file is touched.
pub fn build_embedder(
    cfg: &Config,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Box<dyn Embedder>> {
    if env("FERRET_EMBEDDER").as_deref() == Some("fake") {
        return Ok(Box::new(FakeEmbedder::new_with_profile(
            cfg.embedder.prefix_profile,
            cfg.embedder.dimensions,
        )));
    }
    let http = HttpEmbedder::new(cfg)?;
    http.check_health()?;
    Ok(Box::new(http))
}

/// L2-normalize `v`. Any non-finite component, or a zero norm, is an error:
/// ferret stores only finite `f32` and every stored vector is unit length.
pub fn l2_normalize(v: Vec<f32>) -> Result<Vec<f32>> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::fatal("vector contains non-finite values"));
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return Err(Error::fatal("cannot L2-normalize a zero vector"));
    }
    Ok(v.into_iter().map(|x| x / norm).collect())
}

/// Enforce the configured `dimensions` on a raw backend vector (spec §3.3).
/// `openai`/`sglang` receive `dimensions` in the request body and must return
/// exactly that length. `llama_server`/`litert_lm` do not send `dimensions`:
/// a longer vector is truncated then L2-renormalized. A shorter vector, or
/// any non-finite component, is an error.
pub fn truncate_or_check(v: Vec<f32>, backend: Backend, dimensions: u32) -> Result<Vec<f32>> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::fatal("embedder returned non-finite values"));
    }
    let n = dimensions as usize;
    match backend {
        Backend::Openai | Backend::Sglang => {
            if v.len() != n {
                return Err(Error::fatal(format!(
                    "embedder returned {} dimensions, expected {dimensions}",
                    v.len()
                )));
            }
            Ok(v)
        }
        Backend::LlamaServer | Backend::LitertLm => {
            if v.len() < n {
                return Err(Error::fatal(format!(
                    "embedder returned {} dimensions, expected at least {dimensions}",
                    v.len()
                )));
            }
            if v.len() > n {
                return l2_normalize(v.into_iter().take(n).collect());
            }
            Ok(v)
        }
    }
}
