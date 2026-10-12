//! Indexing: walk the corpus, fingerprint config, and write rows.
//!
//! `walk` enumerates the corpus and classifies each path; the indexer
//! pipeline (`run`) lands in Task 11.

pub mod skip;
pub mod walk;

pub use skip::SkipReason;
pub use walk::{kind_for_ext, walk, WalkEvent};
