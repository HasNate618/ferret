//! Extraction: turn file bytes into embeddable chunks (spec §4).
//!
//! v1 kinds: text extraction here; images and PDF pages land in later
//! tasks and get their own modules.

use crate::types::ChunkKind;

pub mod audio;
pub mod image;
pub mod pdf;
pub mod text;
pub mod video;

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
    /// Segment timestamps in milliseconds (audio/video segments;
    /// `None` otherwise).
    pub t_start_ms: Option<i64>,
    pub t_end_ms: Option<i64>,
    /// PNG bytes for image chunks (and Task 15 PDF pages), WAV bytes for
    /// audio segments, MP4 bytes for video segments: the exact payload
    /// sent to the embedder (the field name is historical; renaming it
    /// is out of scope). `None` for text.
    pub image_bytes: Option<Vec<u8>>,
}

pub use text::extract_text;

/// One file's extraction result: zero or more chunks. Text files emit many;
/// an image emits exactly one; a PDF page emits one per page (Task 15).
/// (`Vec<Chunk>` today; the name locks the plan's `ExtractOutcome`.)
pub type ExtractOutcome = Vec<Chunk>;
