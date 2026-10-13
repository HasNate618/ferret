//! OpenAI-shaped HTTP text embeddings for `llama_server` / `litert_lm`
//! (Task 12; spec Appendix B.1), B.2/B.3 image bodies (Tasks 14–15), and
//! the `openai` / `sglang` dialects (Task 16; Appendices B.4–B.5).

use std::sync::OnceLock;
use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::json;

use super::probe;
use super::{apply_prefix, l2_normalize, truncate_or_check, EmbedInput, Embedder};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::types::Backend;
use base64::Engine as _;

/// Delays before attempts 2, 3, 4 on 429 / 503 / transport failure.
const RETRY_DELAYS: [Duration; 3] = [
    Duration::from_millis(200),
    Duration::from_millis(400),
    Duration::from_millis(800),
];

/// OpenAI's native width: `dimensions` is omitted from request bodies at
/// exactly this value (Appendix B.5) and sent otherwise.
pub(crate) const NATIVE_DIMENSIONS: u32 = 768;

pub struct HttpEmbedder {
    backend: Backend,
    profile: crate::types::PrefixProfile,
    dimensions: u32,
    client: Client,
    cfg: Config,
    images: OnceLock<bool>,
}

impl HttpEmbedder {
    /// Pure constructor: builds the client, performs no I/O. Health and the
    /// image probe run lazily (`check_health`, `images_supported`) so tests
    /// and `openai` construction never touch the network here.
    pub fn new(cfg: &Config) -> Result<Self> {
        let client = Client::builder()
            .timeout(probe::client_timeout(cfg))
            .build()
            .map_err(|e| Error::fatal(format!("cannot build HTTP client: {e}")))?;
        Ok(Self {
            backend: cfg.embedder.backend,
            profile: cfg.embedder.prefix_profile,
            dimensions: cfg.embedder.dimensions,
            client,
            cfg: cfg.clone(),
            images: OnceLock::new(),
        })
    }

    /// Startup gate used by `build_embedder`: fatal when the server is
    /// down (spawning it first when `managed = true`).
    pub fn check_health(&self) -> Result<()> {
        probe::ensure_up(&self.client, &self.cfg)
    }

    /// One prefixed text embedding round-trip with retries (Appendix B.1).
    fn post_text(&self, text: &str) -> Result<Vec<f32>> {
        let body = json!({
            "model": self.cfg.embedder.model,
            "encoding_format": "float",
            "input": text,
        });
        let resp = self.post_with_retries(&probe::embeddings_url(&self.cfg), &body)?;
        parse_embedding(resp)
    }

    /// One image embedding round-trip with retries (Appendix B.2).
    fn post_image(&self, png: &[u8]) -> Result<Vec<f32>> {
        let body = image_body(&self.cfg, png);
        let resp = self.post_with_retries(&probe::embeddings_url(&self.cfg), &body)?;
        parse_embedding(resp)
    }

    /// One PDF page round-trip with retries (Appendix B.3: image part
    /// first, then the text part).
    fn post_pdf_page(&self, png: &[u8], text: Option<&str>) -> Result<Vec<f32>> {
        let body = pdf_page_body(&self.cfg, self.profile, png, text);
        let resp = self.post_with_retries(&probe::embeddings_url(&self.cfg), &body)?;
        parse_embedding(resp)
    }

    /// llama_server / litert_lm text+image dialect (Appendices B.1–B.3).
    fn embed_llama(&self, input: EmbedInput) -> Result<Vec<f32>> {
        let sized = match &input {
            // Appendix B.2: the data-URL part is the whole input; no text
            // prefix of any kind.
            EmbedInput::Image { bytes, .. } => {
                let raw = self.post_image(bytes)?;
                truncate_or_check(raw, self.backend, self.dimensions)?
            }
            // Appendix B.3: image part first, then the Document-prefixed
            // page text (empty string when the page has no text).
            EmbedInput::PdfPage { image_png, text } => {
                let raw = self.post_pdf_page(image_png, text.as_deref())?;
                truncate_or_check(raw, self.backend, self.dimensions)?
            }
            _ => match apply_prefix(self.profile, &input) {
                Some(t) => {
                    let raw = self.post_text(&t)?;
                    truncate_or_check(raw, self.backend, self.dimensions)?
                }
                None => {
                    return Err(Error::fatal(
                        "internal error: unprefixable input reached llama embed",
                    ));
                }
            },
        };
        Ok(sized)
    }

    /// `dimensions` is sent only when it differs from the native width.
    fn maybe_dimensions(&self) -> Option<u32> {
        (self.dimensions != NATIVE_DIMENSIONS).then_some(self.dimensions)
    }

    /// openai dialect (Appendix B.5): prefixed strings only, never image
    /// parts. Image inputs refuse to post (per-file); PDF pages embed as
    /// text-only Documents even when page images were rendered.
    fn embed_openai(&self, input: EmbedInput) -> Result<Vec<f32>> {
        let text = match &input {
            EmbedInput::Image { .. } => {
                return Err(Error::file(
                    "openai embeddings are text-only; image skipped",
                ));
            }
            EmbedInput::PdfPage { text, .. } => prefixed_or_empty(self.profile, text.as_deref()),
            _ => match apply_prefix(self.profile, &input) {
                Some(t) => t,
                None => {
                    return Err(Error::fatal(
                        "internal error: unprefixable input reached openai embed",
                    ));
                }
            },
        };
        let mut body = json!({
            "model": self.cfg.embedder.model,
            "encoding_format": "float",
            "input": text,
        });
        if let Some(d) = self.maybe_dimensions() {
            body["dimensions"] = d.into();
        }
        let resp = self.post_with_retries(&probe::embeddings_url(&self.cfg), &body)?;
        let raw = parse_embedding(resp)?;
        truncate_or_check(raw, self.backend, self.dimensions)
    }

    /// sglang dialect (Appendix B.4): `{text, image}` items, one image per
    /// item. The PDF text carries the `<|image|>` marker first, then the
    /// Document-prefixed page text (marker alone when textless).
    fn embed_sglang(&self, input: EmbedInput) -> Result<Vec<f32>> {
        let item = match &input {
            EmbedInput::PdfPage { image_png, text } => {
                let prefixed = prefixed_or_empty(self.profile, text.as_deref());
                let text = if prefixed.is_empty() {
                    "<|image|>".to_string()
                } else {
                    format!("<|image|>\n{prefixed}")
                };
                json!({
                    "text": text,
                    "image": data_uri(image_png),
                })
            }
            EmbedInput::Image { bytes, .. } => {
                // Standalone image: the marker locates the image, same
                // shape the sglang probe uses (no live sglang server
                // exists to freeze this further).
                json!(sglang_image_item(bytes))
            }
            _ => match apply_prefix(self.profile, &input) {
                Some(t) => json!({ "text": t }),
                None => {
                    return Err(Error::fatal(
                        "internal error: unprefixable input reached sglang embed",
                    ));
                }
            },
        };
        let mut body = json!({
            "model": self.cfg.embedder.model,
            "encoding_format": "float",
            "input": [item],
        });
        if let Some(d) = self.maybe_dimensions() {
            body["dimensions"] = d.into();
        }
        let resp = self.post_with_retries(&probe::embeddings_url(&self.cfg), &body)?;
        let raw = parse_embedding(resp)?;
        truncate_or_check(raw, self.backend, self.dimensions)
    }

    /// POST with the frozen retry policy: 429 / 503 / transport failure
    /// retry after 200/400/800ms (at most 4 attempts); other 4xx return a
    /// per-file error immediately; other 5xx are fatal without retry.
    fn post_with_retries(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<reqwest::blocking::Response> {
        let mut last_err = String::new();
        // Attempt 0 goes out immediately; attempts 1..=3 sleep first.
        for attempt in 0..4usize {
            if attempt > 0 {
                std::thread::sleep(RETRY_DELAYS[attempt - 1]);
            }
            let send = probe::authed(self.client.post(url), &self.cfg)
                .json(body)
                .send();
            match send {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return Ok(resp);
                    } else if status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                    {
                        last_err = format!("{url} -> {status}");
                        continue;
                    } else if status.is_client_error() {
                        return Err(Error::file(format!("embedder rejected the request ({status} for {url}); file skipped")));
                    } else {
                        return Err(Error::fatal(format!(
                            "embedder failed ({status} for {url})"
                        )));
                    }
                }
                Err(e) => {
                    last_err = e.to_string();
                    continue;
                }
            }
        }
        Err(Error::fatal(format!(
            "embedder at {} did not recover after retries (last: {last_err})",
            self.cfg.embedder.endpoint,
        )))
    }
}

impl Embedder for HttpEmbedder {
    fn embed(&self, input: EmbedInput) -> Result<Vec<f32>> {
        // One shared tail: every backend returns exactly `dimensions`
        // finite f32s (llama/litert truncate long tails; openai/sglang
        // must match exactly or the call fails — never store a short
        // vector in a wide index), then ferret L2-normalizes unconditionally.
        let sized = match self.backend {
            Backend::Openai => self.embed_openai(input)?,
            Backend::Sglang => self.embed_sglang(input)?,
            Backend::LlamaServer | Backend::LitertLm => self.embed_llama(input)?,
        };
        // The server may not L2-normalize: always normalize what we store.
        l2_normalize(sized)
    }

    fn images_supported(&self) -> bool {
        *self.images.get_or_init(|| {
            if self.backend == Backend::Openai {
                return false;
            }
            if let Some(v) = self.cfg.embedder.multimodal {
                return v;
            }
            let ok = probe::probe_images(&self.client, &self.cfg);
            if !ok {
                eprintln!(
                    "warning: embedder at {} shows no image support; image files will be skipped",
                    self.cfg.embedder.endpoint,
                );
            }
            ok
        })
    }
}

/// Frozen Appendix B.2 image body, shared by the image embed path and
/// the capability probe: the data-URL part is the whole input, no text
/// prefix of any kind.
pub(crate) fn image_body(cfg: &Config, png: &[u8]) -> serde_json::Value {
    let url = data_uri(png);
    json!({
        "model": cfg.embedder.model,
        "encoding_format": "float",
        "input": [{ "content": [{ "type": "image_url", "image_url": { "url": url } }] }],
    })
}

/// Frozen Appendix B.3 PDF page body: the image part first, then the
/// text part carrying the profile's Document-prefixed page text (empty
/// string when the page has no text or `embed_text` is off — the part is
/// still present). No `<|image|>` marker anywhere: llama.cpp substitutes
/// the image part itself. (Spec prose governs over the B.3 example JSON,
/// which shows the text abbreviated.)
pub(crate) fn pdf_page_body(
    cfg: &Config,
    profile: crate::types::PrefixProfile,
    png: &[u8],
    text: Option<&str>,
) -> serde_json::Value {
    let url = data_uri(png);
    let prefixed = prefixed_or_empty(profile, text);
    json!({
        "model": cfg.embedder.model,
        "encoding_format": "float",
        "input": [{ "content": [
            { "type": "image_url", "image_url": { "url": url } },
            { "type": "text", "text": prefixed },
        ] }],
    })
}

/// `data:image/png;base64,…` URI for one PNG's bytes.
fn data_uri(png: &[u8]) -> String {
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    )
}

/// The profile's Document-prefixed text, or `""` when there is none.
fn prefixed_or_empty(profile: crate::types::PrefixProfile, text: Option<&str>) -> String {
    text.and_then(|t| {
        apply_prefix(
            profile,
            &EmbedInput::Document {
                title: None,
                text: t.to_string(),
            },
        )
    })
    .unwrap_or_default()
}

/// Frozen sglang image item (Appendix B.4 family): the `<|image|>` marker
/// locates the image. Shared by standalone-image embeds and the sglang
/// capability probe so both speak one shape.
pub(crate) fn sglang_image_item(png: &[u8]) -> serde_json::Value {
    json!({
        "text": "<|image|>",
        "image": data_uri(png),
    })
}

/// True when a response value carries a `data[0].embedding` array.
pub(crate) fn has_embedding(v: &serde_json::Value) -> bool {
    v.pointer("/data/0/embedding").is_some_and(|e| e.is_array())
}

/// `data[0].embedding` as `Vec<f32>`; anything else is a fatal protocol
/// error (the server spoke, but not embeddings).
fn parse_embedding(resp: reqwest::blocking::Response) -> Result<Vec<f32>> {
    let v: serde_json::Value = resp.json().map_err(|e| {
        Error::fatal(format!("embedder returned a non-JSON response: {e}"))
    })?;
    v.pointer("/data/0/embedding")
        .and_then(|e| e.as_array())
        .ok_or_else(|| Error::fatal("embedder response has no data[0].embedding"))
        .and_then(|arr| {
            arr.iter()
                .map(|n| {
                    n.as_f64()
                        .map(|f| f as f32)
                        .ok_or_else(|| Error::fatal("embedder returned a non-numeric embedding value"))
                })
                .collect()
        })
}
