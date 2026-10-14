//! Task 12: HTTP text embeddings against llama_server / litert_lm
//! (OpenAI-shaped `POST {endpoint}/v1/embeddings`). All servers are
//! `httpmock`; no real network.

mod common;

use ferret::config::{Config, compiled_defaults};
use ferret::embedder::{EmbedInput, Embedder, HttpEmbedder};
use ferret::types::{Backend, QueryTask};

fn test_cfg(endpoint: &str) -> Config {
    let mut cfg = compiled_defaults(&common::test_dirs(std::path::Path::new("/ferret-t12")));
    cfg.embedder.endpoint = endpoint.to_string();
    cfg.embedder.dimensions = 4;
    cfg
}

fn doc_input(text: &str) -> EmbedInput {
    EmbedInput::Document {
        title: None,
        text: text.to_string(),
    }
}

#[test]
fn llama_server_query_body_is_appendix_b1() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": "task: search result | query: pets in the lease"
            }));
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": [1.0, 0.0, 0.0, 0.0] }]
        }));
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    let _ = e
        .embed(EmbedInput::Query {
            text: "pets in the lease".into(),
            task: QueryTask::Search,
        })
        .unwrap();
    mock.assert();
}

#[test]
fn llama_server_omits_dimensions_key() {
    // `json_body` is an EXACT structural match: if the client sent a
    // `dimensions` key the mock would not match, the server would 404, and
    // the unwrap below would fail. A successful embed therefore proves the
    // key is absent (llama_server / litert_lm must not receive it).
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": "title: none | text: pets"
            }));
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": [0.0, 1.0, 0.0, 0.0] }]
        }));
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    let _ = e.embed(doc_input("pets")).unwrap();
    mock.assert();
}

#[test]
fn retries_503_up_to_attempt_limit() {
    // Persistent 503: initial attempt + 3 retries, then a fatal error.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(503).body("overloaded");
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    let err = e.embed(doc_input("retry me")).unwrap_err();
    assert!(err.to_string().contains("503"), "unexpected error: {err}");
    assert_eq!(mock.calls(), 4, "expected initial attempt + 3 retries");
}

#[test]
fn succeeds_after_503_cleared() {
    // httpmock has no response sequences, so recovery is tested by
    // deleting the 503 mock and re-mocking 200: the next embed succeeds.
    let server = httpmock::MockServer::start();
    let mut down = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(503).body("overloaded");
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    let _ = e.embed(doc_input("first")).unwrap_err();
    assert_eq!(down.calls(), 4);
    down.delete();
    let up = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": [0.0, 0.0, 1.0, 0.0] }]
        }));
    });
    let v = e.embed(doc_input("second")).unwrap();
    assert_eq!(v.len(), 4);
    up.assert();
}

#[test]
fn does_not_retry_400() {
    let server = httpmock::MockServer::start();
    let bad = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(400).body("bad request");
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    let err = e.embed(doc_input("bad")).unwrap_err();
    assert!(err.to_string().contains("400"), "unexpected error: {err}");
    assert!(err.is_file_error(), "4xx must be a per-file error: {err}");
    assert_eq!(bad.calls(), 1, "4xx must not be retried");
}

#[test]
fn bearer_header_when_api_key_set() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .header("Authorization", "Bearer secret");
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": [0.0, 0.0, 0.0, 1.0] }]
        }));
    });
    let mut cfg = test_cfg(&server.base_url());
    cfg.embedder.api_key = "secret".to_string();
    let e = HttpEmbedder::new(&cfg).unwrap();
    let _ = e.embed(doc_input("authed")).unwrap();
    mock.assert();
}

#[test]
fn stored_vectors_are_unit_length() {
    let server = httpmock::MockServer::start();
    let _mock = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        // Server returns an unnormalized vector; ferret must L2-normalize.
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": [3.0, 4.0, 0.0, 0.0] }]
        }));
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    let v = e.embed(doc_input("normalize me")).unwrap();
    assert_eq!(v, vec![0.6f32, 0.8, 0.0, 0.0]);
}

#[test]
fn health_404_falls_back_to_probe_post() {
    // llama.cpp llama-server exposes GET /health; builds that 404 it must
    // fall back to a probe POST of input "ok".
    let server = httpmock::MockServer::start();
    let health = server.mock(|when, then| {
        when.method("GET").path("/health");
        then.status(404);
    });
    let probe = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": "ok"
            }));
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": [1.0, 0.0, 0.0, 0.0] }]
        }));
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    e.check_health().unwrap();
    health.assert();
    probe.assert();
}

#[test]
fn health_failure_is_fatal_and_names_endpoint() {
    let server = httpmock::MockServer::start();
    let health = server.mock(|when, then| {
        when.method("GET").path("/health");
        then.status(500).body("down");
    });
    // Probe POST also fails so there is no fallback success.
    let _probe = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(500).body("down");
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    let err = e.check_health().unwrap_err();
    assert!(
        err.to_string().contains(&server.base_url()),
        "health error must name the endpoint: {err}"
    );
    health.assert();
}

#[test]
fn images_supported_false_for_openai_backend() {
    // Unreachable endpoint: the constructor must do no I/O, and openai
    // short-circuits to false without probing.
    let mut cfg = test_cfg("http://127.0.0.1:9");
    cfg.embedder.backend = Backend::Openai;
    let e = HttpEmbedder::new(&cfg).unwrap();
    assert!(!e.images_supported());
}

#[test]
fn multimodal_config_overrides_probe() {
    // Unreachable endpoint: an explicit `multimodal = false` must win
    // without any network probe.
    let mut cfg = test_cfg("http://127.0.0.1:9");
    cfg.embedder.multimodal = Some(false);
    let e = HttpEmbedder::new(&cfg).unwrap();
    assert!(!e.images_supported());
}

#[test]
fn llama_server_image_body_is_appendix_b2() {
    // Frozen B.2: input:[{content:[{type:image_url,...}]}]. `json_body`
    // is an EXACT structural match, so any extra key (dimensions, text
    // prefix) fails the mock and the unwrap below.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": [{
                    "content": [{
                        "type": "image_url",
                        "image_url": { "url": "data:image/png;base64,QUJD" }
                    }]
                }]
            }));
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": [1.0, 0.0, 0.0, 0.0] }]
        }));
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    // "ABC" is not a valid PNG; the body shape is what matters here (the
    // server is mocked). base64("ABC") == "QUJD".
    let v = e
        .embed(EmbedInput::Image {
            bytes: b"ABC".to_vec(),
            mime: "image/png".into(),
        })
        .unwrap();
    assert_eq!(v.len(), 4);
    mock.assert();
}

#[test]
fn llama_server_pdf_body_is_appendix_b3() {
    // Frozen B.3: image part first, then the Document-prefixed text part.
    // Exact structural match: a `<|image|>` marker or a missing text part
    // fails the mock and the unwrap below.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": [{
                    "content": [
                        { "type": "image_url",
                          "image_url": { "url": "data:image/png;base64,QUJD" } },
                        { "type": "text",
                          "text": "title: none | text: page one pets" }
                    ]
                }]
            }));
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": [1.0, 0.0, 0.0, 0.0] }]
        }));
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    let v = e
        .embed(EmbedInput::PdfPage {
            image_png: b"ABC".to_vec(),
            text: Some("page one pets".to_string()),
        })
        .unwrap();
    assert_eq!(v.len(), 4);
    mock.assert();
}

#[test]
fn llama_server_pdf_body_textless_page_has_empty_text_part() {
    // No page text (or embed_text=false): the text part is still present
    // with an empty string — never a `<|image|>` marker.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": [{
                    "content": [
                        { "type": "image_url",
                          "image_url": { "url": "data:image/png;base64,QUJD" } },
                        { "type": "text", "text": "" }
                    ]
                }]
            }));
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": [0.0, 1.0, 0.0, 0.0] }]
        }));
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    let _ = e
        .embed(EmbedInput::PdfPage {
            image_png: b"ABC".to_vec(),
            text: None,
        })
        .unwrap();
    mock.assert();
}

fn openai_cfg(endpoint: &str, dimensions: u32) -> Config {
    let mut cfg = test_cfg(endpoint);
    cfg.embedder.backend = Backend::Openai;
    cfg.embedder.dimensions = dimensions;
    cfg
}

fn sglang_cfg(endpoint: &str, dimensions: u32) -> Config {
    let mut cfg = test_cfg(endpoint);
    cfg.embedder.backend = Backend::Sglang;
    cfg.embedder.dimensions = dimensions;
    cfg
}

fn embedding_json(n: u32) -> serde_json::Value {
    let mut v = vec![0.0f64; n as usize];
    v[0] = 1.0;
    serde_json::json!({ "data": [{ "embedding": v }] })
}

#[test]
fn openai_query_sends_dimensions_when_not_768() {
    // B.5 with dimensions: input is a plain prefixed string.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "dimensions": 4,
                "input": "task: search result | query: pets in the lease"
            }));
        then.status(200).json_body(embedding_json(4));
    });
    let cfg = openai_cfg(&server.base_url(), 4);
    let e = HttpEmbedder::new(&cfg).unwrap();
    let v = e
        .embed(EmbedInput::Query {
            text: "pets in the lease".into(),
            task: QueryTask::Search,
        })
        .unwrap();
    assert_eq!(v.len(), 4);
    mock.assert();
}

#[test]
fn openai_omits_dimensions_at_768() {
    // Exact match without the key: sending it at 768 fails the mock.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": "title: none | text: pets"
            }));
        then.status(200).json_body(embedding_json(768));
    });
    let cfg = openai_cfg(&server.base_url(), 768);
    let e = HttpEmbedder::new(&cfg).unwrap();
    let v = e.embed(doc_input("pets")).unwrap();
    assert_eq!(v.len(), 768);
    mock.assert();
}

#[test]
fn openai_wrong_length_is_fatal() {
    // A 256-d response in a 768-d index must fail, never store truncated.
    let server = httpmock::MockServer::start();
    let _mock = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(200).json_body(embedding_json(256));
    });
    let cfg = openai_cfg(&server.base_url(), 768);
    let e = HttpEmbedder::new(&cfg).unwrap();
    let err = e.embed(doc_input("pets")).unwrap_err();
    assert!(!err.is_file_error(), "dimension mismatch is fatal: {err}");
}

#[test]
fn openai_image_file_never_posts() {
    // backend=openai: images capability-skip before any embed, so no POST
    // may contain an image part.
    let server = httpmock::MockServer::start();
    let image_posts = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .body_includes("image_url");
        then.status(200).json_body(embedding_json(4));
    });
    let any_posts = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(200).json_body(embedding_json(4));
    });
    let mut cfg = openai_cfg(&server.base_url(), 4);
    cfg.embedder.multimodal = None;
    let e = HttpEmbedder::new(&cfg).unwrap();
    assert!(!e.images_supported(), "openai is text-only");
    // Direct embed of an image input must also refuse to post.
    let err = e
        .embed(EmbedInput::Image {
            bytes: b"PNG".to_vec(),
            mime: "image/png".into(),
        })
        .unwrap_err();
    assert!(err.is_file_error(), "openai image is per-file: {err}");
    assert_eq!(image_posts.calls(), 0, "no image part may post to openai");
    assert_eq!(any_posts.calls(), 0, "openai Image must not POST at all");
}

#[test]
fn openai_pdf_page_embeds_as_text_only_document() {
    // Even with embed_page_image on: no image part, Document text only.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "dimensions": 4,
                "input": "title: none | text: page one pets"
            }));
        then.status(200).json_body(embedding_json(4));
    });
    let cfg = openai_cfg(&server.base_url(), 4);
    let e = HttpEmbedder::new(&cfg).unwrap();
    let v = e
        .embed(EmbedInput::PdfPage {
            image_png: b"PNG".to_vec(),
            text: Some("page one pets".to_string()),
        })
        .unwrap();
    assert_eq!(v.len(), 4);
    mock.assert();
}

#[test]
fn sglang_pdf_uses_image_marker() {
    // Frozen B.4: text starts with "<|image|>\n", image is a data URI.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "dimensions": 4,
                "input": [{
                    "text": "<|image|>\ntitle: none | text: page one pets",
                    "image": "data:image/png;base64,QUJD"
                }]
            }));
        then.status(200).json_body(embedding_json(4));
    });
    let cfg = sglang_cfg(&server.base_url(), 4);
    let e = HttpEmbedder::new(&cfg).unwrap();
    let v = e
        .embed(EmbedInput::PdfPage {
            image_png: b"ABC".to_vec(),
            text: Some("page one pets".to_string()),
        })
        .unwrap();
    assert_eq!(v.len(), 4);
    mock.assert();
}

#[test]
fn sglang_text_item_sends_dimensions_when_not_768() {
    // Text items carry {text}; dimensions only when != 768.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "dimensions": 4,
                "input": [{ "text": "title: none | text: pets" }]
            }));
        then.status(200).json_body(embedding_json(4));
    });
    let cfg = sglang_cfg(&server.base_url(), 4);
    let e = HttpEmbedder::new(&cfg).unwrap();
    let _ = e.embed(doc_input("pets")).unwrap();
    mock.assert();
}

#[test]
fn sglang_textless_pdf_sends_marker_alone() {
    // Interface bullet: marker alone when textless.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "dimensions": 4,
                "input": [{
                    "text": "<|image|>",
                    "image": "data:image/png;base64,QUJD"
                }]
            }));
        then.status(200).json_body(embedding_json(4));
    });
    let cfg = sglang_cfg(&server.base_url(), 4);
    let e = HttpEmbedder::new(&cfg).unwrap();
    let v = e
        .embed(EmbedInput::PdfPage {
            image_png: b"ABC".to_vec(),
            text: None,
        })
        .unwrap();
    assert_eq!(v.len(), 4);
    mock.assert();
}

#[test]
fn sglang_omits_dimensions_at_768() {
    // Counterpart to the openai omission case: no key at native width.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": [{ "text": "title: none | text: pets" }]
            }));
        then.status(200).json_body(embedding_json(768));
    });
    let cfg = sglang_cfg(&server.base_url(), 768);
    let e = HttpEmbedder::new(&cfg).unwrap();
    let v = e.embed(doc_input("pets")).unwrap();
    assert_eq!(v.len(), 768);
    mock.assert();
}

#[test]
fn sglang_standalone_image_uses_marker_item() {
    // Provisional shape (no live server): same marker item as the probe.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "dimensions": 4,
                "input": [{
                    "text": "<|image|>",
                    "image": "data:image/png;base64,QUJD"
                }]
            }));
        then.status(200).json_body(embedding_json(4));
    });
    let cfg = sglang_cfg(&server.base_url(), 4);
    let e = HttpEmbedder::new(&cfg).unwrap();
    let v = e
        .embed(EmbedInput::Image {
            bytes: b"ABC".to_vec(),
            mime: "image/png".into(),
        })
        .unwrap();
    assert_eq!(v.len(), 4);
    mock.assert();
}

#[test]
fn sglang_probe_posts_b4_shape() {
    // images_supported with multimodal=None must probe in sglang's own
    // dialect (B.4 marker item), not the B.2 shape — and report true.
    let server = httpmock::MockServer::start();
    let png_b64 = ferret::embedder::probe::PROBE_PNG_BASE64;
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": [{
                    "text": "<|image|>",
                    "image": format!("data:image/png;base64,{png_b64}")
                }]
            }));
        then.status(200).json_body(embedding_json(4));
    });
    let cfg = sglang_cfg(&server.base_url(), 4);
    let e = HttpEmbedder::new(&cfg).unwrap();
    assert!(e.images_supported());
    mock.assert();
}

#[test]
fn openai_textless_pdf_posts_empty_string() {
    // Freezes current behavior: text-only Document with empty input
    // (likely a per-file 400 downstream, never an image part).
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "dimensions": 4,
                "input": ""
            }));
        then.status(200).json_body(embedding_json(4));
    });
    let cfg = openai_cfg(&server.base_url(), 4);
    let e = HttpEmbedder::new(&cfg).unwrap();
    let _ = e
        .embed(EmbedInput::PdfPage {
            image_png: b"PNG".to_vec(),
            text: None,
        })
        .unwrap();
    mock.assert();
}

#[test]
fn probe_audio_true_on_768d_response() {
    use base64::Engine as _;
    use ferret::embedder::probe::{probe_audio, probe_wav};
    let server = httpmock::MockServer::start();
    let data = base64::engine::general_purpose::STANDARD.encode(probe_wav());
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": [{ "content": [{ "type": "input_audio",
                    "input_audio": { "data": data } }] }]
            }));
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": vec![0.1; 768] }]
        }));
    });
    let cfg = test_cfg(&server.base_url());
    let client = reqwest::blocking::Client::new();
    assert!(probe_audio(&client, &cfg));
    mock.assert();
}

#[test]
fn probe_audio_false_on_400() {
    use ferret::embedder::probe::probe_audio;
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(400).body("no audio here");
    });
    let cfg = test_cfg(&server.base_url());
    let client = reqwest::blocking::Client::new();
    assert!(!probe_audio(&client, &cfg));
    mock.assert();
}

#[test]
fn probe_audio_false_without_posting_for_openai() {
    use ferret::embedder::probe::probe_audio;
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": vec![0.1; 768] }]
        }));
    });
    let mut cfg = test_cfg(&server.base_url());
    cfg.embedder.backend = Backend::Openai;
    let client = reqwest::blocking::Client::new();
    assert!(!probe_audio(&client, &cfg));
    mock.assert_hits(0);
}

#[test]
fn llama_audio_posts_input_audio_part() {
    use base64::Engine as _;
    use ferret::embedder::{EmbedInput, probe};
    let server = httpmock::MockServer::start();
    let wav = probe::probe_wav();
    let data = base64::engine::general_purpose::STANDARD.encode(&wav);
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": [{ "content": [{ "type": "input_audio",
                    "input_audio": { "data": data } }] }]
            }));
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": vec![0.1; 768] }]
        }));
    });
    let cfg = test_cfg(&server.base_url());
    let e = HttpEmbedder::new(&cfg).unwrap();
    let v = e.embed(EmbedInput::Audio { wav_bytes: wav }).unwrap();
    assert_eq!(v.len(), cfg.embedder.dimensions as usize);
    mock.assert();
}

#[test]
fn openai_audio_is_file_error_without_posting() {
    use ferret::embedder::{EmbedInput, Embedder};
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(200).json_body(serde_json::json!({"data": []}));
    });
    let mut cfg = test_cfg(&server.base_url());
    cfg.embedder.backend = Backend::Openai;
    let e = HttpEmbedder::new(&cfg).unwrap();
    let err = e
        .embed(EmbedInput::Audio {
            wav_bytes: vec![1, 2, 3],
        })
        .unwrap_err();
    assert!(err.is_file_error());
    mock.assert_hits(0);
}

#[test]
fn sglang_and_litert_audio_are_file_errors_without_posting() {
    use ferret::embedder::{EmbedInput, Embedder};
    for backend in [Backend::Sglang, Backend::LitertLm] {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("POST").path("/v1/embeddings");
            then.status(200).json_body(serde_json::json!({"data": []}));
        });
        let mut cfg = test_cfg(&server.base_url());
        cfg.embedder.backend = backend;
        let e = HttpEmbedder::new(&cfg).unwrap();
        let err = e
            .embed(EmbedInput::Audio {
                wav_bytes: vec![1, 2, 3],
            })
            .unwrap_err();
        assert!(
            err.is_file_error(),
            "backend {backend:?} should refuse audio"
        );
        mock.assert_hits(0);
    }
}

#[test]
fn probe_video_true_on_768d_response() {
    use base64::Engine as _;
    use ferret::embedder::probe::{probe_mp4, probe_video};
    let server = httpmock::MockServer::start();
    let data = base64::engine::general_purpose::STANDARD.encode(probe_mp4());
    let mock = server.mock(|when, then| {
        when.method("POST")
            .path("/v1/embeddings")
            .json_body(serde_json::json!({
                "model": "embeddinggemma-2",
                "encoding_format": "float",
                "input": [{ "content": [{ "type": "input_video",
                    "input_video": { "url": format!("data:video/mp4;base64,{data}") } }] }]
            }));
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": vec![0.1; 768] }]
        }));
    });
    let cfg = test_cfg(&server.base_url());
    let client = reqwest::blocking::Client::new();
    assert!(probe_video(&client, &cfg));
    mock.assert();
}

#[test]
fn probe_video_false_on_400() {
    use ferret::embedder::probe::probe_video;
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(400).body("no video here");
    });
    let cfg = test_cfg(&server.base_url());
    let client = reqwest::blocking::Client::new();
    assert!(!probe_video(&client, &cfg));
    mock.assert();
}

#[test]
fn probe_video_false_without_posting_for_openai() {
    use ferret::embedder::probe::probe_video;
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST").path("/v1/embeddings");
        then.status(200).json_body(serde_json::json!({
            "data": [{ "embedding": vec![0.1; 768] }]
        }));
    });
    let mut cfg = test_cfg(&server.base_url());
    cfg.embedder.backend = ferret::types::Backend::Openai;
    let client = reqwest::blocking::Client::new();
    assert!(!probe_video(&client, &cfg));
    mock.assert_hits(0);
}
