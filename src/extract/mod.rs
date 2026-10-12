//! Extraction: turn file bytes into embeddable chunks (spec §4).
//!
//! v1 kinds: text extraction here; images and PDF pages land in later
//! tasks and get their own modules.

use crate::types::ChunkKind;

pub mod text;

/// What the indexer does with an extracted chunk (spec §4.1–4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedDecision {
    /// Write the chunks row, FTS row, and vec row.
    Vec,
    /// `numeric_heavy`: chunks row + FTS row, but no vec row.
    FtsOnly,
    /// `skip_letterless`: do not emit at all.
    Drop,
}

/// One extracted chunk. `start_byte` / `end_byte` are UTF-8 byte offsets
/// into the ORIGINAL file bytes (spec §4.1); `page` is None for text.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub kind: ChunkKind,
    pub text: Option<String>,
    pub start_byte: Option<i64>,
    pub end_byte: Option<i64>,
    pub page: Option<i64>,
    pub embed: EmbedDecision,
}

pub use text::extract_text;
