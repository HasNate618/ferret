//! Deterministic fake embedder (`FERRET_EMBEDDER=fake`): blake3-derived unit
//! vectors, independent of any server. Test hook only.

use super::{EmbedInput, Embedder, l2_normalize};
use crate::error::Result;
use crate::types::PrefixProfile;

pub struct FakeEmbedder {
    profile: PrefixProfile,
    dimensions: u32,
}

impl FakeEmbedder {
    /// Test hook with the model-card profile default.
    pub fn new(dimensions: u32) -> Self {
        Self::new_with_profile(PrefixProfile::EmbeddingGemma2, dimensions)
    }

    /// Fake embedder carrying the merged-config profile so vectors follow
    /// the active prefix strings.
    pub fn new_with_profile(profile: PrefixProfile, dimensions: u32) -> Self {
        Self {
            profile,
            dimensions,
        }
    }

    /// `blake3(prefix || 0x1f || payload)`: the prefix is the profile's text
    /// prompt (empty for image-only inputs), the payload is the UTF-8 text or
    /// the raw image bytes. A PDF page hashes `image_png || 0x1f || text` so
    /// it never collides with a plain `Image` of the same bytes.
    fn hash_input(&self, input: &EmbedInput) -> Vec<u8> {
        let mut buf = Vec::new();
        if let Some(prefix) = super::apply_prefix(self.profile, input) {
            buf.extend_from_slice(prefix.as_bytes());
        }
        buf.push(0x1f);
        match input {
            EmbedInput::Query { text, .. } | EmbedInput::Document { text, .. } => {
                buf.extend_from_slice(text.as_bytes());
            }
            EmbedInput::Image { bytes, .. } => buf.extend_from_slice(bytes),
            EmbedInput::Audio { wav_bytes } => buf.extend_from_slice(wav_bytes),
            EmbedInput::PdfPage { image_png, text } => {
                buf.extend_from_slice(image_png);
                buf.push(0x1f);
                if let Some(text) = text {
                    buf.extend_from_slice(text.as_bytes());
                }
            }
        }
        buf
    }
}

impl Embedder for FakeEmbedder {
    fn embed(&self, input: EmbedInput) -> Result<Vec<f32>> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.hash_input(&input));
        // First `dimensions` output bytes as `u8 as f32`, always finite; the
        // extended output covers `dimensions` beyond blake3's 32-byte digest.
        let mut bytes = vec![0u8; self.dimensions as usize];
        hasher.finalize_xof().fill(&mut bytes);
        let raw: Vec<f32> = bytes.iter().map(|&b| b as f32).collect();
        l2_normalize(raw)
    }

    fn images_supported(&self) -> bool {
        true
    }
}
