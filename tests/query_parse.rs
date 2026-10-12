//! Task 7 integration tests: search query parsing (prefixes, hyphens, dates).

use ferret::search::{parse_query, FilterFlags};
use ferret::types::{FileKind, QueryTask};

#[test]
fn kind_pdf_and_spaced() {
    let a = parse_query("kind:pdf pets", &FilterFlags::default()).unwrap();
    let b = parse_query("kind: pdf pets", &FilterFlags::default()).unwrap();
    assert_eq!(a.filters.kind, vec![FileKind::Pdf]);
    assert_eq!(a.nl_query, "pets");
    assert_eq!(a, b);
}

#[test]
fn hyphenated_word_is_not_fts_minus() {
    let p = parse_query("left-wing", &FilterFlags::default()).unwrap();
    assert_eq!(p.keyword_query, "left-wing");
    assert_eq!(p.semantic_query, "left-wing");
}

#[test]
fn leading_minus_is_operator() {
    let p = parse_query("-wing birds", &FilterFlags::default()).unwrap();
    assert_eq!(p.keyword_query, "-wing birds");
    assert_eq!(p.semantic_query, "birds"); // standalone FTS op token dropped from semantic copy
}

#[test]
fn unknown_prefix_stays_in_query() {
    let p = parse_query("foo:bar pets", &FilterFlags::default()).unwrap();
    assert_eq!(p.nl_query, "foo:bar pets");
}

#[test]
fn flags_win_over_prefixes() {
    let flags = FilterFlags {
        kind: vec![FileKind::Text],
        ..Default::default()
    };
    let p = parse_query("kind:pdf pets", &flags).unwrap();
    assert_eq!(p.filters.kind, vec![FileKind::Text]);
}

#[test]
fn code_prefix_sets_task() {
    let p = parse_query("code: hashmap rust", &FilterFlags::default()).unwrap();
    assert_eq!(p.task, QueryTask::Code);
    assert_eq!(p.nl_query, "hashmap rust");
}

#[test]
fn invalid_date_is_usage() {
    let err = parse_query("after:2026-13-01 x", &FilterFlags::default())
        .unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn quoted_prefix_value_keeps_spaces() {
    let p = parse_query("under:\"/tmp\" pets", &FilterFlags::default()).unwrap();
    assert_eq!(p.filters.under, Some(std::fs::canonicalize("/tmp").unwrap()));
    assert_eq!(p.nl_query, "pets");
}

#[test]
fn repeated_prefixes_union() {
    let p = parse_query("kind:pdf kind:text pets", &FilterFlags::default()).unwrap();
    assert_eq!(p.filters.kind, vec![FileKind::Pdf, FileKind::Text]);
}

#[test]
fn phrase_unquotes_in_semantic_copy_only() {
    let p = parse_query("\"pets in the lease\"", &FilterFlags::default()).unwrap();
    assert_eq!(p.semantic_query, "pets in the lease");
    assert_eq!(p.keyword_query, "\"pets in the lease\"");
}

#[test]
fn boolean_operators_dropped_from_semantic_copy() {
    let p = parse_query("cats OR dogs", &FilterFlags::default()).unwrap();
    assert_eq!(p.semantic_query, "cats dogs");
    assert_eq!(p.keyword_query, "cats OR dogs");
}

#[test]
fn filters_only_query_has_empty_text_copy() {
    let p = parse_query("kind:pdf", &FilterFlags::default()).unwrap();
    assert_eq!(p.nl_query, "");
    assert_eq!(p.semantic_query, "");
    assert_eq!(p.filters.kind, vec![FileKind::Pdf]);
}

#[test]
fn plus_token_dropped_semantically() {
    let p = parse_query("+lease pets", &FilterFlags::default()).unwrap();
    assert_eq!(p.semantic_query, "pets");
    assert_eq!(p.keyword_query, "+lease pets");
}

#[test]
fn missing_under_dir_is_usage() {
    let err = parse_query("under:/definitely/not/here x", &FilterFlags::default()).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}
