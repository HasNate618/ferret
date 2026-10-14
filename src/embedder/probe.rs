//! Health checks, managed-server spawn, and the image capability probe
//! for the HTTP embedders (Task 12). The text POST itself lives in
//! `super::http`; this module answers "is the server up?" and "does it do
//! images?" so `HttpEmbedder` stays small.

use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::json;

use crate::config::Config;
use crate::error::{Error, Result};

/// 1x1 transparent PNG (RFC 2083 minimal IDAT), base64. Sent as a data URL
/// by [`probe_images`]; the server never sees a real corpus image here.
pub const PROBE_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

/// Verify the server is up. `GET {endpoint}/health` 200 wins; 404 falls
/// back to a probe `POST {endpoint}/v1/embeddings` with input `"ok"` for
/// servers without the health route. Anything else (other 4xx/5xx or a
/// transport failure) is `Error::fatal` naming the endpoint.
pub fn check_health(client: &Client, cfg: &Config) -> Result<()> {
    let base = base_url(cfg);
    let health_url = format!("{base}/health");
    match client.get(health_url).send() {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(resp) if resp.status() == reqwest::StatusCode::NOT_FOUND => probe_post(client, cfg),
        Ok(resp) => Err(health_down(
            cfg,
            &format!("GET /health -> {}", resp.status()),
        )),
        Err(e) => Err(health_down(cfg, &e.to_string())),
    }
}

/// `managed = true` startup: check health, and if the server is down, spawn
/// it and check once more. `Ok` means a later request may proceed; `Err`
/// names the endpoint.
pub fn ensure_up(client: &Client, cfg: &Config) -> Result<()> {
    match check_health(client, cfg) {
        Ok(()) => Ok(()),
        Err(first) => {
            if !cfg.embedder.managed {
                return Err(first);
            }
            spawn_managed(cfg)?;
            check_health(client, cfg)
        }
    }
}

/// Ask the server to embed a 1x1 PNG in its own dialect (B.2 for
/// llama/litert, B.4 marker item for sglang). 200 with a parseable
/// embedding means image support; HTTP 4xx or a transport failure means
/// text-only. Other 5xx are treated as "no" too — a 500 on a 67-byte PNG
/// is not a healthy multimodal server.
pub fn probe_images(client: &Client, cfg: &Config) -> bool {
    use base64::Engine as _;
    let Ok(png) = base64::engine::general_purpose::STANDARD.decode(PROBE_PNG_BASE64) else {
        return false;
    };
    // The probe speaks the backend's own image dialect: an sglang-only
    // server would 4xx the B.2 shape, so sglang probes with a B.4 item.
    // openai never reaches here (short-circuited by the caller).
    let body = match cfg.embedder.backend {
        crate::types::Backend::Sglang => serde_json::json!({
            "model": cfg.embedder.model,
            "encoding_format": "float",
            "input": [super::http::sglang_image_item(&png)],
        }),
        _ => super::http::image_body(cfg, &png),
    };
    let resp = authed(client.post(embeddings_url(cfg)), cfg)
        .json(&body)
        .send();
    match resp {
        Ok(r) if r.status().is_success() => r
            .json::<serde_json::Value>()
            .is_ok_and(|v| super::http::has_embedding(&v)),
        _ => false,
    }
}

/// Deterministic 1s 440Hz sine, 16kHz mono 16-bit PCM with a 44-byte WAV
/// header (32044 bytes total), synthesized in code — no fixture. Shared
/// with tests so the probe-body mock can assert the exact base64.
pub fn probe_wav() -> Vec<u8> {
    let samples: Vec<i16> = (0..16000)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 16000.0).sin() * (i16::MAX as f32))
        .map(|s| s as i16)
        .collect();
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + samples.len() * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&32000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        wav.extend_from_slice(&s.to_le_bytes());
    }
    wav
}

/// Audio capability probe: POSTs the 1s sine as a llama `input_audio`
/// part (Appendix B.6 body, shared with the embed path). True only on
/// 200 + a 768-d finite embedding. Non-llama backends refuse audio
/// (Task 5) and return false without dialing.
pub fn probe_audio(client: &Client, cfg: &Config) -> bool {
    if cfg.embedder.backend != crate::types::Backend::LlamaServer {
        return false;
    }
    let body = super::http::audio_body(cfg, &probe_wav());
    let resp = authed(client.post(embeddings_url(cfg)), cfg)
        .json(&body)
        .send();
    match resp {
        Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().is_ok_and(|v| {
            v.pointer("/data/0/embedding")
                .and_then(|e| e.as_array())
                .is_some_and(|a| {
                    a.len() == 768 && a.iter().all(|v| v.as_f64().is_some_and(|f| f.is_finite()))
                })
        }),
        _ => false,
    }
}

/// Minimal `ftyp` box (24 bytes), opaque probe payload: mocks never
/// decode it, and live decoding is the Task 8 gate. Shared with tests
/// so the probe-body mock can assert the exact base64.
pub fn probe_mp4() -> Vec<u8> {
    let mut mp4 = Vec::with_capacity(24);
    mp4.extend_from_slice(&20u32.to_be_bytes());
    mp4.extend_from_slice(b"ftyp");
    mp4.extend_from_slice(b"isom");
    mp4.extend_from_slice(&0u32.to_be_bytes());
    mp4.extend_from_slice(b"isom");
    mp4
}

/// Video capability probe: POSTs the [`probe_mp4`] bytes as a llama
/// `input_video` part (Appendix B.7 body, shared with the embed path
/// via `super::http::video_body`). True only on 200 + a 768-d finite
/// embedding. Non-llama backends refuse video (Task 4) and return
/// false without dialing.
pub fn probe_video(client: &Client, cfg: &Config) -> bool {
    if cfg.embedder.backend != crate::types::Backend::LlamaServer {
        return false;
    }
    let body = super::http::video_body(cfg, &probe_mp4());
    let resp = authed(client.post(embeddings_url(cfg)), cfg)
        .json(&body)
        .send();
    match resp {
        Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().is_ok_and(|v| {
            v.pointer("/data/0/embedding")
                .and_then(|e| e.as_array())
                .is_some_and(|a| {
                    a.len() == 768 && a.iter().all(|v| v.as_f64().is_some_and(|f| f.is_finite()))
                })
        }),
        _ => false,
    }
}
pub fn embeddings_url(cfg: &Config) -> String {
    format!("{}/v1/embeddings", base_url(cfg))
}

/// Base URL with a trailing slash stripped.
pub fn base_url(cfg: &Config) -> String {
    cfg.embedder.endpoint.trim_end_matches('/').to_string()
}

/// Attach `Authorization: Bearer {api_key}` when the key is non-empty.
pub fn authed(
    req: reqwest::blocking::RequestBuilder,
    cfg: &Config,
) -> reqwest::blocking::RequestBuilder {
    if cfg.embedder.api_key.is_empty() {
        req
    } else {
        req.bearer_auth(&cfg.embedder.api_key)
    }
}

/// Probe POST used when `GET /health` 404s. The frozen body is Appendix B.1
/// with input `"ok"`; only the status matters.
fn probe_post(client: &Client, cfg: &Config) -> Result<()> {
    let body = json!({
        "model": cfg.embedder.model,
        "encoding_format": "float",
        "input": "ok",
    });
    match authed(client.post(embeddings_url(cfg)), cfg)
        .json(&body)
        .send()
    {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(resp) => Err(health_down(
            cfg,
            &format!("probe POST -> {}", resp.status()),
        )),
        Err(e) => Err(health_down(cfg, &e.to_string())),
    }
}

fn health_down(cfg: &Config, detail: &str) -> Error {
    Error::fatal(format!(
        "embedder is down at {} ({detail}); start it or set embedder.managed = true",
        cfg.embedder.endpoint,
    ))
}

/// Spawn the managed server and return immediately: ferret never shuts the
/// child down. Uses `managed_command` verbatim when non-empty, else the
/// frozen `llama-server` default (Appendix: `--mmproj` omitted when empty).
fn spawn_managed(cfg: &Config) -> Result<()> {
    let argv: Vec<String> = if cfg.embedder.managed_command.is_empty() {
        default_server_argv(cfg)?
    } else {
        cfg.embedder.managed_command.clone()
    };
    let (prog, args) = argv
        .split_first()
        .ok_or_else(|| Error::fatal("embedder.managed_command is empty"))?;
    std::process::Command::new(prog)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| Error::fatal(format!("could not spawn managed embedder ({prog}): {e}")))?;
    // The follow-up health check (with its own timeout) is the readiness
    // wait; no sleep here.
    Ok(())
}

fn default_server_argv(cfg: &Config) -> Result<Vec<String>> {
    let port = reqwest::Url::parse(&cfg.embedder.endpoint)
        .ok()
        .and_then(|u| u.port())
        .ok_or_else(|| {
            Error::fatal(format!(
                "cannot take a port from embedder.endpoint `{}` for the managed llama-server command",
                cfg.embedder.endpoint,
            ))
        })?;
    let mut argv = vec![
        "llama-server".to_string(),
        "-m".to_string(),
        cfg.embedder.gguf.clone(),
    ];
    if !cfg.embedder.mmproj.is_empty() {
        argv.push("--mmproj".to_string());
        argv.push(cfg.embedder.mmproj.clone());
    }
    argv.extend(
        [
            "--embeddings",
            "--pooling",
            "mean",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "-c",
            "4096",
            "-b",
            "4096",
            "-ub",
            "4096",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    Ok(argv)
}

/// Client timeout from config (blocking client; per-request ceiling).
pub fn client_timeout(cfg: &Config) -> Duration {
    Duration::from_millis(cfg.embedder.timeout_ms.max(1))
}
