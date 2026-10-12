//! Text extraction and chunking (spec §4.1–4.2).
//!
//! Two modes: `chars` slides overlapping windows (preferring newline, then
//! space breaks); `sentences` splits on Unicode sentence boundaries and
//! emits one chunk per sentence — spec §10 pins the fixture "two sentences
//! → two chunks when `max_chars` would keep them as one", so the pack
//! limit applies only to the over-long-sentence fallback, not as a pack
//! overlap. HTML is stripped with a sequential scan (no browser parser)
//! that keeps a byte map back into the original file, so stored offsets
//! point at the original bytes.

use std::path::Path;

use unicode_segmentation::UnicodeSegmentation;

use crate::config::TextChunkConfig;
use crate::error::Result;
use crate::types::{ChunkKind, TextChunkMode};

use super::{Chunk, EmbedDecision};

/// Extract embeddable text chunks from `bytes` (the contents of `path`;
/// only the extension matters, for HTML dispatch).
pub fn extract_text(path: &Path, bytes: &[u8], cfg: &TextChunkConfig) -> Result<Vec<Chunk>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    // `text` is what chunking sees; `spans[i]` is the byte range in the
    // ORIGINAL file the i-th Unicode scalar came from (entity decodings
    // map to the whole entity's span).
    let (text, spans) = match ext.as_str() {
        "html" | "htm" => strip_html(bytes),
        // JSON / YAML / TOML / SVG / everything else: raw UTF-8, no
        // pretty-print rewriting.
        _ => {
            let lossy = String::from_utf8_lossy(bytes);
            let spans = lossy
                .char_indices()
                .map(|(b, ch)| (b as i64, (b + ch.len_utf8()) as i64))
                .collect();
            (lossy.into_owned(), spans)
        }
    };

    let ranges = match cfg.mode {
        TextChunkMode::Chars => char_windows(&text, cfg.max_chars, cfg.overlap),
        TextChunkMode::Sentences => sentence_windows(&text, cfg.max_chars, cfg.overlap),
    };

    let mut out = Vec::new();
    for (start, end) in ranges {
        let chunk_text: String = text.chars().skip(start).take(end - start).collect();
        if chunk_text.trim().is_empty() {
            continue;
        }
        let embed = embed_decision(&chunk_text, cfg);
        if embed == EmbedDecision::Drop {
            continue;
        }
        out.push(Chunk {
            kind: ChunkKind::Text,
            text: Some(chunk_text),
            start_byte: spans.get(start).map(|(s, _)| *s),
            end_byte: end.checked_sub(1).and_then(|last| spans.get(last).map(|(_, e)| *e)),
            page: None,
            embed,
            image_bytes: None,
        });
    }
    Ok(out)
}

/// `numeric_heavy` (non-zero) makes digit-dominant chunks FTS-only;
/// `skip_letterless` drops what has no letter and is not numeric (e.g.
/// punctuation separators). Numeric exports are letterless too, so the
/// numeric rule runs FIRST — otherwise `skip_letterless` would drop the
/// very chunks `numeric_heavy` exists to keep for FTS (spec §4.1).
fn embed_decision(text: &str, cfg: &TextChunkConfig) -> EmbedDecision {
    if cfg.numeric_heavy != 0.0 {
        let mut nonws = 0usize;
        let mut digits = 0usize;
        for c in text.chars() {
            if c.is_whitespace() {
                continue;
            }
            nonws += 1;
            if c.is_ascii_digit() {
                digits += 1;
            }
        }
        if nonws > 0 && (digits as f64 / nonws as f64) >= cfg.numeric_heavy {
            return EmbedDecision::FtsOnly;
        }
    }
    if cfg.skip_letterless && !text.chars().any(char::is_alphabetic) {
        return EmbedDecision::Drop;
    }
    EmbedDecision::Vec
}

/// Sliding windows of `max` scalars advancing by `max - overlap`. Windows
/// prefer to break on a newline, then any whitespace, before the hard cap.
fn char_windows(text: &str, max: usize, overlap: usize) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    if total == 0 || max == 0 {
        return Vec::new();
    }
    let step = max.saturating_sub(overlap).max(1);
    let mut out = Vec::new();
    let mut start = 0usize;
    loop {
        let hard_end = (start + max).min(total);
        let end = if hard_end >= total {
            total
        } else {
            let mut newline_end = None;
            let mut space_end = None;
            for i in (start + 1..hard_end).rev() {
                if chars[i] == '\n' {
                    newline_end = Some(i + 1);
                    break;
                }
                if space_end.is_none() && chars[i].is_whitespace() {
                    space_end = Some(i + 1);
                }
            }
            newline_end.or(space_end).unwrap_or(hard_end)
        };
        out.push((start, end));
        if end >= total {
            break;
        }
        let next = end.saturating_sub(overlap).max(start + 1);
        if next <= start {
            // Defensive: overlap >= max is rejected by config, but never
            // loop forever if it slips through.
            start += step;
        } else {
            start = next;
        }
    }
    out
}

/// One chunk per sentence (spec §10 fixture governs); a sentence longer
/// than `max` falls back to overlapping char windows for that sentence
/// only.
fn sentence_windows(text: &str, max: usize, overlap: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut cursor = 0usize; // scalar index
    for sentence in text.unicode_sentences() {
        let len = sentence.chars().count();
        let start = cursor;
        cursor += len;
        if sentence.trim().is_empty() {
            continue;
        }
        if len > max {
            for (ws, we) in char_windows(sentence, max, overlap) {
                out.push((start + ws, start + we));
            }
            continue;
        }
        out.push((start, cursor));
    }
    out
}

// --- HTML stripping --------------------------------------------------------

/// Strip HTML to plain text with a byte-span map back to the original
/// bytes. Drops tags, comments, and `<script>` / `<style>` inner text;
/// decodes entities; keeps whitespace verbatim. Every kept scalar records
/// the span of the original bytes it came from, so a chunk's `end_byte`
/// excludes trailing markup.
fn strip_html(bytes: &[u8]) -> (String, Vec<(i64, i64)>) {
    let src = String::from_utf8_lossy(bytes);
    let b = src.as_bytes();
    let mut out = String::new();
    let mut spans: Vec<(i64, i64)> = Vec::new();
    let mut i = 0usize;

    while i < b.len() {
        if b[i] == b'<' {
            // Comment: skip to `-->`.
            if src[i..].starts_with("<!--") {
                match src[i..].find("-->") {
                    Some(end) => {
                        i += end + 3;
                        continue;
                    }
                    None => break,
                }
            }
            let (name, tag_end) = scan_tag(&src, i);
            if matches!(name.as_str(), "script" | "style") {
                // Drop inner text up to the closing tag.
                match find_closing_tag(&src, tag_end, &name) {
                    Some(close) => {
                        i = close;
                        continue;
                    }
                    None => break,
                }
            }
            i = tag_end;
            continue;
        }
        if b[i] == b'&' {
            if let Some((decoded, len)) = decode_entity(&src, i) {
                // Every char of a decoded entity spans the entity's bytes.
                for ch in decoded.chars() {
                    out.push(ch);
                    spans.push((i as i64, (i + len) as i64));
                }
                i += len;
                continue;
            }
        }
        let ch_len = utf8_len(b[i]);
        out.push_str(&src[i..i + ch_len]);
        spans.push((i as i64, (i + ch_len) as i64));
        i += ch_len;
    }
    (out, spans)
}

/// Scan a tag starting at `start` (at `<`). Returns the lowercase tag name
/// and the index just past the tag's `>` (quoted attribute values may
/// contain `>`).
fn scan_tag(src: &str, start: usize) -> (String, usize) {
    let b = src.as_bytes();
    let mut name_end = start + 1;
    while name_end < b.len() && !b[name_end].is_ascii_whitespace() && b[name_end] != b'>' {
        name_end += 1;
    }
    let name = src[start + 1..name_end].to_ascii_lowercase();
    let mut k = start + 1;
    let mut quote: Option<u8> = None;
    while k < b.len() {
        match b[k] {
            q @ (b'"' | b'\'') => match quote {
                Some(open) if open == q => quote = None,
                Some(_) => {}
                None => quote = Some(q),
            },
            b'>' if quote.is_none() => return (name, k + 1),
            _ => {}
        }
        k += 1;
    }
    (name, b.len())
}

/// Index of the closing `</name>` at or after `from` (case-insensitive).
fn find_closing_tag(src: &str, from: usize, name: &str) -> Option<usize> {
    let lower = src[from..].to_ascii_lowercase();
    lower.find(&format!("</{name}")).map(|p| from + p)
}

/// Decode one HTML entity at `at` (`&amp;` → `&`). Entities without a
/// semicolon, or unknown ones, stay literal.
fn decode_entity(src: &str, at: usize) -> Option<(String, usize)> {
    let rest = &src[at..];
    let semi = rest.find(';')?;
    if semi > 32 {
        return None;
    }
    let entity = &rest[..=semi];
    if entity.len() < 3 || !entity.starts_with('&') {
        return None;
    }
    let decoded = html_escape::decode_html_entities(entity);
    if decoded == entity {
        return None;
    }
    Some((decoded.to_string(), entity.len()))
}

fn utf8_len(lead: u8) -> usize {
    match lead {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_entities_and_script_inner_text() {
        let input = b"<p>Hi&amp;bye</p><script>var x=1;</script>tail";
        let (text, spans) = strip_html(input);
        assert_eq!(text, "Hi&byetail");
        assert_eq!(spans[0], (3, 4)); // 'H' of "Hi"
        assert_eq!(spans[2], (5, 10)); // '&' decodes from the whole &amp; entity
        assert_eq!(*spans.last().unwrap(), (45, 46)); // final 'l' of tail
    }

    #[test]
    fn comment_is_dropped() {
        let (text, _) = strip_html(b"<!-- c --><p>ok</p>");
        assert_eq!(text, "ok");
    }

    #[test]
    fn unterminated_entity_stays_literal() {
        let (text, _) = strip_html(b"a&b");
        assert_eq!(text, "a&b");
    }

    #[test]
    fn quoted_attribute_may_contain_gt() {
        let (text, _) = strip_html(b"<a title=\"x>y\">z</a>");
        assert_eq!(text, "z");
    }
}
