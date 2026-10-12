//! Fake embedder: determinism, unit length, per-kind distinctness, and the
//! `FERRET_EMBEDDER=fake` hook in `build_embedder`.

mod common;

use common::test_dirs;
use ferret::config::compiled_defaults;
use ferret::embedder::{
    build_embedder, l2_normalize, truncate_or_check, EmbedInput, Embedder, FakeEmbedder,
};
use ferret::types::{Backend, PrefixProfile, QueryTask};

fn no_env(_: &str) -> Option<String> {
    None
}

fn fake_env(key: &str) -> Option<String> {
    (key == "FERRET_EMBEDDER").then(|| "fake".to_string())
}

fn unit_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn build_embedder_fake_env_wins_over_any_backend() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = compiled_defaults(&test_dirs(tmp.path()));
    let e = build_embedder(&cfg, &fake_env).unwrap();
    assert!(e.images_supported());
    let v = e
        .embed(EmbedInput::Query { text: "pets".into(), task: QueryTask::Search })
        .unwrap();
    assert_eq!(v.len(), cfg.embedder.dimensions as usize);
    assert!(v.iter().all(|x| x.is_finite()));
    assert!((unit_norm(&v) - 1.0).abs() < 1e-5);
}

#[test]
fn build_embedder_without_fake_is_fatal_until_task_12() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = compiled_defaults(&test_dirs(tmp.path()));
    let err = match build_embedder(&cfg, &no_env) {
        Err(err) => err,
        Ok(_) => panic!("expected fatal error without the fake hook"),
    };
    assert_eq!(err.to_string(), "http embedder not implemented");
}

#[test]
fn fake_vectors_differ_per_input_kind() {
    let e = FakeEmbedder::new(8);
    let query = e
        .embed(EmbedInput::Query { text: "hi".into(), task: QueryTask::Search })
        .unwrap();
    let doc = e
        .embed(EmbedInput::Document { title: None, text: "hi".into() })
        .unwrap();
    let image = e.embed(EmbedInput::Image { bytes: b"hi".to_vec(), mime: "image/png".into() }).unwrap();
    let page_textless =
        e.embed(EmbedInput::PdfPage { image_png: b"hi".to_vec(), text: None }).unwrap();
    let page_with_text =
        e.embed(EmbedInput::PdfPage { image_png: b"hi".to_vec(), text: Some("hi".into()) }).unwrap();
    // Same payload bytes, different kinds: all five vectors are distinct.
    for (name, v) in [
        ("query", &query),
        ("document", &doc),
        ("image", &image),
        ("page_textless", &page_textless),
        ("page_with_text", &page_with_text),
    ] {
        for (other_name, other) in [
            ("query", &query),
            ("document", &doc),
            ("image", &image),
            ("page_textless", &page_textless),
            ("page_with_text", &page_with_text),
        ] {
            if name != other_name {
                assert_ne!(v, other, "{name} vs {other_name} must differ");
            }
        }
    }
}

#[test]
fn fake_query_task_changes_vector() {
    let e = FakeEmbedder::new(8);
    let search = e
        .embed(EmbedInput::Query { text: "parse".into(), task: QueryTask::Search })
        .unwrap();
    let code = e
        .embed(EmbedInput::Query { text: "parse".into(), task: QueryTask::Code })
        .unwrap();
    assert_ne!(search, code);
}

#[test]
fn fake_respects_profile_and_dimensions() {
    let gemma2 = FakeEmbedder::new_with_profile(PrefixProfile::EmbeddingGemma2, 128);
    let litert = FakeEmbedder::new_with_profile(PrefixProfile::LitertLm, 128);
    let input = EmbedInput::Query { text: "pets".into(), task: QueryTask::Search };
    let a = gemma2.embed(input.clone()).unwrap();
    let b = litert.embed(input).unwrap();
    assert_eq!(a.len(), 128);
    assert_ne!(a, b, "profiles must produce distinct vectors");
}

#[test]
fn fake_embed_batch_matches_embed() {
    let e = FakeEmbedder::new(8);
    let inputs = vec![
        EmbedInput::Document { title: Some("a.txt".into()), text: "hi".into() },
        EmbedInput::Query { text: "pets".into(), task: QueryTask::Search },
    ];
    let batch = e.embed_batch(&inputs).unwrap();
    assert_eq!(batch[0], e.embed(inputs[0].clone()).unwrap());
    assert_eq!(batch[1], e.embed(inputs[1].clone()).unwrap());
}

#[test]
fn l2_normalize_rejects_nonfinite_and_zero() {
    assert!(l2_normalize(vec![f32::NAN, 1.0]).is_err());
    assert!(l2_normalize(vec![f32::INFINITY, 1.0]).is_err());
    assert!(l2_normalize(vec![0.0, 0.0]).is_err());
    let v = l2_normalize(vec![3.0, 4.0]).unwrap();
    assert_eq!(v, vec![0.6, 0.8]);
}

#[test]
fn truncate_or_check_openai_requires_exact_dimensions() {
    let v: Vec<f32> = (0..4).map(|i| i as f32 + 1.0).collect();
    let out = truncate_or_check(v.clone(), Backend::Openai, 4).unwrap();
    assert_eq!(out, v);
    assert!(truncate_or_check(v.clone(), Backend::Openai, 3).is_err());
    assert!(truncate_or_check(v.clone(), Backend::Sglang, 5).is_err());
    assert!(truncate_or_check(vec![f32::NAN, 1.0, 2.0, 3.0], Backend::Openai, 4).is_err());
}

#[test]
fn truncate_or_check_llama_truncates_and_renorms() {
    // 3-4-5 right triangle over 6 values: truncating to 3 must renormalize
    // the prefix back to unit length.
    let v = vec![3.0, 4.0, 0.0, 5.0, 12.0, 13.0];
    let out = truncate_or_check(v, Backend::LlamaServer, 3).unwrap();
    assert_eq!(out, vec![0.6, 0.8, 0.0]);
    assert!((unit_norm(&out) - 1.0).abs() < 1e-6);
    // Exact length passes through unchanged.
    let exact = vec![0.6, 0.8, 0.0];
    assert_eq!(
        truncate_or_check(exact.clone(), Backend::LitertLm, 3).unwrap(),
        exact
    );
    // Shorter is an error.
    assert!(truncate_or_check(vec![1.0, 2.0], Backend::LlamaServer, 3).is_err());
    assert!(truncate_or_check(vec![1.0, f32::NAN], Backend::LitertLm, 3).is_err());
}
