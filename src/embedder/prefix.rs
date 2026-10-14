//! Prompt prefixes per profile (spec §3.2). The strings are the model-card /
//! LiteRT-LM docs spellings verbatim; test vectors pin them, so do not
//! "improve" them. Prefixes apply to text only: image-only inputs (and a
//! PDF page with no text) get no text prefix.

use super::EmbedInput;
use crate::types::{PrefixProfile, QueryTask};

/// Text prompt for `input` under `profile`, or `None` when there is no text
/// part. A PDF page with text uses the profile's Document prefix (title
/// `"none"`); a textless page has no text part.
pub fn apply_prefix(profile: PrefixProfile, input: &EmbedInput) -> Option<String> {
    match (profile, input) {
        (
            PrefixProfile::EmbeddingGemma2,
            EmbedInput::Query {
                text,
                task: QueryTask::Search,
            },
        ) => Some(format!("task: search result | query: {text}")),
        (
            PrefixProfile::EmbeddingGemma2,
            EmbedInput::Query {
                text,
                task: QueryTask::Code,
            },
        ) => Some(format!("task: code retrieval | query: {text}")),
        (PrefixProfile::EmbeddingGemma2, EmbedInput::Document { title, text }) => Some(format!(
            "title: {} | text: {text}",
            title.as_deref().unwrap_or("none")
        )),
        (
            PrefixProfile::EmbeddingGemma2,
            EmbedInput::PdfPage {
                text: Some(text), ..
            },
        ) => Some(format!("title: none | text: {text}")),
        (
            PrefixProfile::LitertLm,
            EmbedInput::Query {
                text,
                task: QueryTask::Search,
            },
        ) => Some(format!("task: search query | text: {text}")),
        (
            PrefixProfile::LitertLm,
            EmbedInput::Query {
                text,
                task: QueryTask::Code,
            },
        ) => Some(format!("task: code retrieval | query: {text}")),
        (PrefixProfile::LitertLm, EmbedInput::Document { text, .. }) => {
            Some(format!("task: search result | text: {text}"))
        }
        (
            PrefixProfile::LitertLm,
            EmbedInput::PdfPage {
                text: Some(text), ..
            },
        ) => Some(format!("task: search result | text: {text}")),
        // Image-only inputs, audio/video segments, and a PDF page with no text
        // have no text part (model rule: prefixes are text-only).
        (_, EmbedInput::Audio { .. }) => None,
        (_, EmbedInput::Video { .. }) => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Embedder, FakeEmbedder};
    use super::{EmbedInput, apply_prefix};
    use crate::types::{PrefixProfile, QueryTask};

    #[test]
    fn gemma2_search_query_prefix() {
        let s = apply_prefix(
            PrefixProfile::EmbeddingGemma2,
            &EmbedInput::Query {
                text: "pets".into(),
                task: QueryTask::Search,
            },
        );
        assert_eq!(s.as_deref(), Some("task: search result | query: pets"));
    }

    #[test]
    fn litert_search_query_prefix() {
        let s = apply_prefix(
            PrefixProfile::LitertLm,
            &EmbedInput::Query {
                text: "pets".into(),
                task: QueryTask::Search,
            },
        );
        assert_eq!(s.as_deref(), Some("task: search query | text: pets"));
    }

    #[test]
    fn image_only_has_no_text_prefix() {
        let s = apply_prefix(
            PrefixProfile::EmbeddingGemma2,
            &EmbedInput::Image {
                bytes: vec![1],
                mime: "image/png".into(),
            },
        );
        assert!(s.is_none());
    }

    #[test]
    fn audio_has_no_text_prefix() {
        for profile in [PrefixProfile::EmbeddingGemma2, PrefixProfile::LitertLm] {
            let s = apply_prefix(profile, &EmbedInput::Audio { wav_bytes: vec![1] });
            assert!(s.is_none());
        }
    }

    #[test]
    fn video_has_no_text_prefix() {
        for profile in [PrefixProfile::EmbeddingGemma2, PrefixProfile::LitertLm] {
            let s = apply_prefix(profile, &EmbedInput::Video { mp4_bytes: vec![1] });
            assert!(s.is_none());
        }
    }

    #[test]
    fn fake_is_unit_length_and_deterministic() {
        let e = FakeEmbedder::new(8);
        let a = e
            .embed(EmbedInput::Document {
                title: Some("a.txt".into()),
                text: "hi".into(),
            })
            .unwrap();
        let b = e
            .embed(EmbedInput::Document {
                title: Some("a.txt".into()),
                text: "hi".into(),
            })
            .unwrap();
        assert_eq!(a, b);
        let n = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gemma2_code_query_prefix() {
        let s = apply_prefix(
            PrefixProfile::EmbeddingGemma2,
            &EmbedInput::Query {
                text: "parse".into(),
                task: QueryTask::Code,
            },
        );
        assert_eq!(s.as_deref(), Some("task: code retrieval | query: parse"));
    }

    #[test]
    fn litert_code_query_prefix() {
        let s = apply_prefix(
            PrefixProfile::LitertLm,
            &EmbedInput::Query {
                text: "parse".into(),
                task: QueryTask::Code,
            },
        );
        assert_eq!(s.as_deref(), Some("task: code retrieval | query: parse"));
    }

    #[test]
    fn gemma2_document_prefix_uses_none_for_missing_title() {
        let s = apply_prefix(
            PrefixProfile::EmbeddingGemma2,
            &EmbedInput::Document {
                title: None,
                text: "pets".into(),
            },
        );
        assert_eq!(s.as_deref(), Some("title: none | text: pets"));
    }

    #[test]
    fn litert_document_prefix() {
        let s = apply_prefix(
            PrefixProfile::LitertLm,
            &EmbedInput::Document {
                title: Some("a.txt".into()),
                text: "pets".into(),
            },
        );
        assert_eq!(s.as_deref(), Some("task: search result | text: pets"));
    }

    #[test]
    fn pdf_page_with_text_uses_document_prefix() {
        let s = apply_prefix(
            PrefixProfile::EmbeddingGemma2,
            &EmbedInput::PdfPage {
                image_png: vec![1],
                text: Some("pets".into()),
            },
        );
        assert_eq!(s.as_deref(), Some("title: none | text: pets"));
        let s = apply_prefix(
            PrefixProfile::LitertLm,
            &EmbedInput::PdfPage {
                image_png: vec![1],
                text: Some("pets".into()),
            },
        );
        assert_eq!(s.as_deref(), Some("task: search result | text: pets"));
    }

    #[test]
    fn textless_pdf_page_has_no_text_prefix() {
        for profile in [PrefixProfile::EmbeddingGemma2, PrefixProfile::LitertLm] {
            let s = apply_prefix(
                profile,
                &EmbedInput::PdfPage {
                    image_png: vec![1],
                    text: None,
                },
            );
            assert!(s.is_none());
        }
    }
}
