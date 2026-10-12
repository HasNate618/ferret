//! OpenAI-shaped HTTP text embeddings for `llama_server` / `litert_lm`
//! (Task 12; spec Appendix B.1). `openai` / `sglang` construct fine but
//! their `embed` is fatal until Task 16 wires their dialects.

use std::sync::OnceLock;
use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::json;

use super::probe;
use super::{apply_prefix, l2_normalize, truncate_or_check, EmbedInput, Embedder};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::types::Backend;

/// Delays before attempts 2, 3, 4 on 429 / 503 / transport failure.
const RETRY_DELAYS: [Duration; 3] = [
    Duration::from_millis(200),
    Duration::from_millis(400),
    Duration::from_millis(800),
];

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
        match self.backend {
            Backend::Openai | Backend::Sglang => {
                return Err(Error::fatal(format!(
                    "embedder.backend `{}` needs Task 16 (openai/sglang client)",
                    self.backend,
                )));
            }
            Backend::LlamaServer | Backend::LitertLm => {}
        }
        let text = match apply_prefix(self.profile, &input) {
            Some(t) => t,
            None => {
                return Err(Error::file(
                    "image-only input needs Task 14 (image embeddings)",
                ));
            }
        };
        let raw = self.post_text(&text)?;
        let sized = truncate_or_check(raw, self.backend, self.dimensions)?;
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
