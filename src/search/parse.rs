//! Search query parser.
//!
//! Grammar (spec §6): a prefix is a token at the start of the query or after
//! whitespace — `name:` then a value. `kind:pdf` and `kind: pdf` are the same.
//! A value runs until the next whitespace, or until the closing quote when
//! quoted (`path:"/home/nate/My Documents"`). Unknown prefixes stay in the
//! query text (they are search text, not errors).
//!
//! One scan produces three query copies:
//! - `nl_query` / `keyword_query`: the remainder after prefix stripping,
//!   verbatim, keeping quotes and FTS operators. The keyword arm builds its
//!   FTS5 `MATCH` from this copy (later task); JSON `query` prints it.
//! - `semantic_query`: phrases unquoted (inner text kept) and standalone FTS
//!   operator tokens dropped — `OR`, `AND`, `NOT`, and tokens an operator
//!   sign introduces (`+term` / `-term`).
//!
//! Hyphens: `+` / `-` are operators only at the START of a token (spec §6.3:
//! `left-wing` and `COVID-19` are not subtraction). Interior hyphens are
//! never rewritten.

use std::path::PathBuf;

use chrono::{Local, NaiveDate, NaiveDateTime, NaiveTime, TimeZone};

use crate::config::expand_tilde;
use crate::error::{Error, Result};
use crate::types::{FileKind, QueryTask};

/// clap-parsed filter flags. Flags and query prefixes are equivalent
/// (spec §6.3); flags win on conflict. Repeatable flags union their values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FilterFlags {
    pub kind: Vec<FileKind>,
    pub ext: Vec<String>,
    pub under: Option<PathBuf>,
    pub collection: Option<String>,
    pub name: Option<String>,
    /// `YYYY-MM-DD`, same semantics as the `after:` prefix.
    pub after: Option<String>,
    /// `YYYY-MM-DD`, same semantics as the `before:` prefix.
    pub before: Option<String>,
    pub code: bool,
}

/// SQL-able filters (spec §6.3). All AND-combined by the search query.
#[derive(Debug, Clone, PartialEq)]
pub struct Filters {
    pub kind: Vec<FileKind>,
    pub ext: Vec<String>, // no dot, lower
    pub under: Option<PathBuf>,
    pub collection: Option<String>,
    pub name: Option<String>, // glob, as given
    pub after: Option<i64>,   // unix seconds local 00:00:00
    pub before: Option<i64>,  // next day 00:00:00 local
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedQuery {
    pub nl_query: String,       // remainder after prefix strip (JSON `query`)
    pub semantic_query: String, // unquote phrases, drop standalone OR/AND/NOT
    pub keyword_query: String,  // remainder including quotes and operators
    pub task: QueryTask,
    pub filters: Filters,
}

/// Parse `raw` into filters plus the semantic/keyword query copies. Flags
/// override the equivalent prefixes.
pub fn parse_query(raw: &str, flag_overrides: &FilterFlags) -> Result<ParsedQuery> {
    let mut kind: Vec<FileKind> = Vec::new();
    let mut ext: Vec<String> = Vec::new();
    let mut under: Option<PathBuf> = None;
    let mut collection: Option<String> = None;
    let mut name: Option<String> = None;
    let mut after: Option<i64> = None;
    let mut before: Option<i64> = None;
    let mut task = QueryTask::Search;

    // Remainder tokens: (raw form with quotes, Some(inner) when quoted).
    let mut parts: Vec<(String, Option<String>)> = Vec::new();

    let mut idx = 0;
    while idx < raw.len() {
        if raw[idx..].starts_with(char::is_whitespace) {
            idx += raw[idx..].chars().next().unwrap().len_utf8();
            continue;
        }

        // A quoted token is always query text, never a prefix.
        if raw[idx..].starts_with('"') {
            let (inner, next) = read_quoted(raw, idx)?;
            parts.push((format!("\"{inner}\""), Some(inner)));
            idx = next;
            continue;
        }

        // Try a prefix head: identifier chars up to a ':' (no whitespace or
        // quote may sit between). Anything else is an ordinary word.
        let (head, has_colon, after_head) = read_prefix_head(raw, idx);
        if has_colon && is_known_prefix(head) {
            if head == "code" {
                // `code:` takes no value: `code: hashmap rust` means
                // task = code with query "hashmap rust" (spec §6.3).
                task = QueryTask::Code;
                idx = after_head;
                continue;
            }
            let (value, value_end) = take_value(raw, after_head, head)?;
            match head {
                "kind" | "type" => kind.push(parse_kind(&value)?),
                "ext" => ext.push(normalize_ext(&value)),
                "path" | "under" => under = Some(canonical_dir(&value)?),
                "collection" => collection = Some(value),
                "name" => name = Some(value),
                "after" => after = Some(after_midnight(&value)?),
                "before" => before = Some(before_midnight(&value)?),
                _ => unreachable!("is_known_prefix filtered the names"),
            }
            idx = value_end;
            continue;
        }

        let (word, next) = read_word(raw, idx);
        parts.push((word, None));
        idx = next;
    }

    if flag_overrides.code {
        task = QueryTask::Code;
    }

    // Flags win on conflict; repeatable flags union their own values.
    let kind = if flag_overrides.kind.is_empty() {
        kind
    } else {
        flag_overrides.kind.clone()
    };
    let ext = if flag_overrides.ext.is_empty() {
        ext
    } else {
        flag_overrides.ext.iter().map(|e| normalize_ext(e)).collect()
    };
    let under = match &flag_overrides.under {
        Some(p) => Some(canonical_dir(&p.to_string_lossy())?),
        None => under,
    };
    let collection = flag_overrides.collection.clone().or(collection);
    let name = flag_overrides.name.clone().or(name);
    let after = match &flag_overrides.after {
        Some(d) => Some(after_midnight(d)?),
        None => after,
    };
    let before = match &flag_overrides.before {
        Some(d) => Some(before_midnight(d)?),
        None => before,
    };

    let joined = || -> String {
        parts
            .iter()
            .map(|(raw, _)| raw.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };

    Ok(ParsedQuery {
        nl_query: joined(),
        semantic_query: semantic_copy(&parts),
        keyword_query: joined(),
        task,
        filters: Filters {
            kind,
            ext,
            under,
            collection,
            name,
            after,
            before,
        },
    })
}

/// Known prefix names (spec §6.3 filter table plus `code`).
fn is_known_prefix(head: &str) -> bool {
    matches!(
        head,
        "kind" | "type" | "ext" | "path" | "under" | "collection" | "name" | "after" | "before" | "code"
    )
}

/// Read the candidate prefix head at `start`: identifier chars up to a ':'.
/// Returns (head, whether a colon terminated it, index just past the head).
/// A whitespace or quote terminates the head with no colon.
fn read_prefix_head(raw: &str, start: usize) -> (&str, bool, usize) {
    let mut end = start;
    while end < raw.len() {
        let ch = raw[end..].chars().next().unwrap();
        if ch == ':' {
            return (&raw[start..end], true, end + 1);
        }
        if ch.is_whitespace() || ch == '"' {
            return (&raw[start..end], false, end);
        }
        end += ch.len_utf8();
    }
    (&raw[start..end], false, end)
}

/// Read the value for a prefix whose head ended at `at`.
/// Attached value: `kind:pdf` / quoted `path:"a b"`. Separated value:
/// `kind: pdf` (whitespace skipped, then quoted or bare word).
fn take_value(raw: &str, at: usize, pname: &str) -> Result<(String, usize)> {
    if at < raw.len() && raw[at..].starts_with('"') {
        return read_quoted(raw, at);
    }
    if at < raw.len() && !raw[at..].starts_with(char::is_whitespace) {
        let (word, end) = read_word(raw, at);
        return Ok((word, end));
    }
    let mut idx = at;
    while idx < raw.len() && raw[idx..].starts_with(char::is_whitespace) {
        idx += raw[idx..].chars().next().unwrap().len_utf8();
    }
    if idx >= raw.len() {
        return Err(Error::usage(format!("prefix `{pname}:` requires a value")));
    }
    if raw[idx..].starts_with('"') {
        return read_quoted(raw, idx);
    }
    let (word, end) = read_word(raw, idx);
    Ok((word, end))
}

/// Read a bare word from `start` to the next whitespace.
fn read_word(raw: &str, start: usize) -> (String, usize) {
    let mut end = start;
    while end < raw.len() && !raw[end..].starts_with(char::is_whitespace) {
        end += raw[end..].chars().next().unwrap().len_utf8();
    }
    (raw[start..end].to_string(), end)
}

/// Read a "quoted" token from `start` (which must be at the opening quote).
/// Returns the inner text and the index past the closing quote.
fn read_quoted(raw: &str, start: usize) -> Result<(String, usize)> {
    let mut idx = start + 1;
    let mut inner = String::new();
    while idx < raw.len() {
        if raw[idx..].starts_with('"') {
            return Ok((inner, idx + 1));
        }
        let ch = raw[idx..].chars().next().unwrap();
        inner.push(ch);
        idx += ch.len_utf8();
    }
    Err(Error::usage("unterminated quote in query"))
}

fn parse_kind(v: &str) -> Result<FileKind> {
    v.parse::<FileKind>()
        .map_err(|e| Error::usage(format!("invalid kind filter: {e}")))
}

/// Semantic copy: phrases unquoted, standalone FTS operators dropped.
///
/// An operator is a token that is exactly `OR` / `AND` / `NOT`, or that an
/// operator sign introduces (`+term` / `-term` at token start). `left-wing`
/// and `COVID-19` never start with an operator sign, so they survive.
fn semantic_copy(parts: &[(String, Option<String>)]) -> String {
    let mut tokens: Vec<&str> = Vec::new();
    for (raw, quoted) in parts {
        let text = quoted.as_deref().unwrap_or(raw.as_str());
        for tok in text.split_whitespace() {
            if is_semantic_operator(tok) {
                continue;
            }
            tokens.push(tok);
        }
    }
    tokens.join(" ")
}

fn is_semantic_operator(tok: &str) -> bool {
    matches!(tok, "OR" | "AND" | "NOT") || tok.starts_with('+') || tok.starts_with('-')
}

fn normalize_ext(e: &str) -> String {
    e.trim_start_matches('.').to_lowercase()
}

fn parse_date(s: &str, field: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .map_err(|_| Error::usage(format!("invalid {field} date `{s}` (expected YYYY-MM-DD)")))
}

fn after_midnight(s: &str) -> Result<i64> {
    Ok(midnight_ts(parse_date(s, "after")?))
}

/// `before:` is exclusive-next-day: `mtime <` next day 00:00:00 local, so
/// `before:2026-01-15` includes all of Jan 14.
fn before_midnight(s: &str) -> Result<i64> {
    let d = parse_date(s, "before")?;
    let next = d
        .succ_opt()
        .ok_or_else(|| Error::usage(format!("date `{s}` overflows")))?;
    Ok(midnight_ts(next))
}

fn midnight_ts(d: NaiveDate) -> i64 {
    let dt = NaiveDateTime::new(d, NaiveTime::MIN);
    Local
        .from_local_datetime(&dt)
        .earliest()
        .map(|t| t.timestamp())
        .unwrap_or_else(|| dt.and_utc().timestamp())
}

/// Tilde-expand, then canonicalize; a path that does not exist is a CLI error
/// (spec §6.3: missing DIR is exit 2).
fn canonical_dir(p: &str) -> Result<PathBuf> {
    let expanded = expand_tilde(p);
    std::fs::canonicalize(&expanded)
        .map_err(|_| Error::usage(format!("path does not exist: {}", expanded.display())))
}
