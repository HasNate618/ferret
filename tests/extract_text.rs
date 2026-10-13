//! Task 9: text extraction and chunking.

use ferret::config::TextChunkConfig;
use ferret::extract::{Chunk, EmbedDecision, extract_text};
use ferret::types::{ChunkKind, TextChunkMode};
use std::path::Path;

fn chars_cfg() -> TextChunkConfig {
    TextChunkConfig {
        mode: TextChunkMode::Chars,
        max_chars: 2000,
        overlap: 200,
        skip_letterless: true,
        numeric_heavy: 0.7,
    }
}

#[test]
fn sentences_split_when_chars_would_not() {
    let cfg = TextChunkConfig {
        mode: TextChunkMode::Sentences,
        max_chars: 2000,
        overlap: 200,
        skip_letterless: true,
        numeric_heavy: 0.7,
    };
    let chunks = extract_text(Path::new("x.txt"), b"Hello world. Goodbye moon.", &cfg).unwrap();
    assert_eq!(chunks.len(), 2);
}

#[test]
fn numeric_heavy_is_fts_only() {
    let cfg = TextChunkConfig {
        mode: TextChunkMode::Chars,
        max_chars: 2000,
        overlap: 200,
        skip_letterless: true,
        numeric_heavy: 0.7,
    };
    let chunks = extract_text(Path::new("n.txt"), b"1234567890 12345", &cfg).unwrap();
    assert!(matches!(chunks[0].embed, EmbedDecision::FtsOnly));
}

#[test]
fn letterless_dropped() {
    let cfg = TextChunkConfig {
        mode: TextChunkMode::Chars,
        max_chars: 2000,
        overlap: 200,
        skip_letterless: true,
        numeric_heavy: 0.7,
    };
    let chunks = extract_text(Path::new("x.txt"), b"***\n---\n", &cfg).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn html_strips_tags() {
    let cfg = TextChunkConfig {
        mode: TextChunkMode::Chars,
        max_chars: 2000,
        overlap: 200,
        skip_letterless: true,
        numeric_heavy: 0.7,
    };
    let chunks = extract_text(Path::new("a.html"), b"<p>Hi&amp;bye</p>", &cfg).unwrap();
    assert_eq!(chunks[0].text.as_deref(), Some("Hi&bye"));
}

// --- controller-added coverage -------------------------------------------

#[test]
fn html_offsets_point_into_original_bytes() {
    let bytes = b"<p>Hi&amp;bye</p>";
    let chunks = extract_text(Path::new("a.html"), bytes, &chars_cfg()).unwrap();
    let chunk: &Chunk = &chunks[0];
    let start = chunk.start_byte.unwrap() as usize;
    let end = chunk.end_byte.unwrap() as usize;
    // Offsets point into the ORIGINAL bytes: "Hi&bye" spans the 'H' inside
    // <p> through the 'e' before </p>; the entity &amp; decodes to the '&'.
    assert_eq!((start, end), (3, 13));
    assert_eq!(&bytes[3..5], b"Hi");
    assert_eq!(&bytes[5..10], b"&amp;");
    assert_eq!(&bytes[10..13], b"bye");
}

#[test]
fn plain_text_offsets_span_the_window() {
    let bytes = b"abcdefghij";
    let cfg = TextChunkConfig {
        mode: TextChunkMode::Chars,
        max_chars: 4,
        overlap: 1,
        skip_letterless: true,
        numeric_heavy: 0.0,
    };
    let chunks = extract_text(Path::new("x.txt"), bytes, &cfg).unwrap();
    let first = &chunks[0];
    assert_eq!(first.text.as_deref(), Some("abcd"));
    assert_eq!(first.start_byte, Some(0));
    assert_eq!(first.end_byte, Some(4));
    // Windows advance by max_chars - overlap.
    assert_eq!(chunks[1].text.as_deref(), Some("defg"));
    assert_eq!(chunks[1].start_byte, Some(3));
}

#[test]
fn chars_prefer_breaking_on_newline() {
    let cfg = TextChunkConfig {
        mode: TextChunkMode::Chars,
        max_chars: 10,
        overlap: 0,
        skip_letterless: true,
        numeric_heavy: 0.0,
    };
    let chunks = extract_text(Path::new("x.txt"), b"aaaa\nbbbbcccccc", &cfg).unwrap();
    // The window ends at the newline rather than at exactly 10 scalars.
    assert_eq!(chunks[0].text.as_deref(), Some("aaaa\n"));
    assert_eq!(chunks[0].end_byte, Some(5));
}

#[test]
fn empty_file_yields_no_chunks() {
    let chunks = extract_text(Path::new("x.txt"), b"   \n\t\n", &chars_cfg()).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn over_long_sentence_falls_back_to_char_windows() {
    let cfg = TextChunkConfig {
        mode: TextChunkMode::Sentences,
        max_chars: 10,
        overlap: 0,
        skip_letterless: true,
        numeric_heavy: 0.0,
    };
    // One 25-scalar sentence with no interior break: falls back to windows.
    let chunks = extract_text(Path::new("x.txt"), b"aaaaaaaaaaaaaaaaaaaaaaaaa.", &cfg).unwrap();
    assert!(chunks.len() > 1);
    assert!(chunks.iter().all(|c| c.kind == ChunkKind::Text));
}

#[test]
fn all_chunks_are_text_kind() {
    let chunks = extract_text(
        Path::new("x.txt"),
        b"one two three.",
        &TextChunkConfig {
            mode: TextChunkMode::Sentences,
            ..chars_cfg()
        },
    )
    .unwrap();
    assert!(chunks.iter().all(|c| c.kind == ChunkKind::Text));
    assert!(chunks.iter().all(|c| c.page.is_none()));
}

#[test]
fn testdata_files_parse() {
    let sentences = extract_text(
        Path::new("testdata/two_sentences.txt"),
        b"Hello world. Goodbye moon.",
        &chars_cfg(),
    )
    .unwrap();
    assert_eq!(sentences.len(), 1);
    assert!(matches!(sentences[0].embed, EmbedDecision::Vec));

    let numeric = extract_text(
        Path::new("testdata/numeric.txt"),
        b"1234567890 12345",
        &chars_cfg(),
    )
    .unwrap();
    assert!(matches!(numeric[0].embed, EmbedDecision::FtsOnly));

    let letterless = extract_text(
        Path::new("testdata/letterless.txt"),
        b"***\n---\n",
        &chars_cfg(),
    )
    .unwrap();
    assert!(letterless.is_empty());

    let html = extract_text(
        Path::new("testdata/sample.html"),
        b"<p>Hi&amp;bye</p>",
        &chars_cfg(),
    )
    .unwrap();
    assert_eq!(html[0].text.as_deref(), Some("Hi&bye"));
}
