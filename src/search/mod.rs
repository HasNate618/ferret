//! Search: query parsing (prefixes, filters, semantic/keyword copies) and
//! ranking (RRF, SQL filters, JSON).

pub mod json;
pub mod parse;
pub mod rrf;
pub mod sql;

pub use json::{render_human, render_json};
pub use parse::{parse_query, FilterFlags, Filters, ParsedQuery};
pub use rrf::{rrf, Arm};

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

use rusqlite::OptionalExtension;

use crate::config::Config;
use crate::embedder::{EmbedInput, Embedder};
use crate::error::{Error, Result};
use crate::store::Store;
use crate::types::{ChunkKind, FileKind, QueryTask, SearchMode};

/// A search request: the parsed query plus the mode/limits the CLI resolved.
pub struct SearchRequest {
    pub parsed: ParsedQuery,
    pub mode: SearchMode,
    pub limit: usize,
    pub min_score: f64,
    pub json: bool,
}

/// Per-arm display scores. `None` means that arm did not list the hit
/// (JSON prints `null`; spec §6.4).
pub struct HitScores {
    pub vec: Option<f64>,      // cosine
    pub fts: Option<f64>,      // raw FTS5 rank
    pub filename: Option<f64>, // raw FTS5 rank
}

pub struct Hit {
    pub path: String,
    pub kind: FileKind,
    pub chunk_kind: ChunkKind,
    pub collection: String,
    pub ext: String,
    pub page: Option<i64>,
    pub ordinal: i64,
    pub mtime: i64,
    pub score: f64, // fused RRF
    pub scores: HitScores,
    pub snippet: String,
}

pub struct SearchOutput {
    pub query: String,
    pub task: QueryTask,
    pub mode: SearchMode,
    pub limit: usize,
    pub min_score: f64,
    pub filters: Filters,
    pub hits: Vec<Hit>,
}

/// Run the three-arm hybrid search (spec §6.1):
/// 1. Empty semantic copy: skip the vec arm; `mode = semantic` errors (exit 2).
/// 2. Vec arm with metadata prefilter; `--under` / `--name` inflate `k` and
///    JOIN `files` (no second KNN loop).
/// 3. Keyword arm: FTS5 MATCH on chunk text, same filters, capped.
/// 4. Filename arm: FTS5 MATCH on filename/path only.
/// 5. RRF (unless single-arm mode), drop `< min_score`, take `limit`.
pub fn run_search(
    store: &Store,
    embedder: &dyn Embedder,
    cfg: &Config,
    req: SearchRequest,
) -> Result<SearchOutput> {
    let parsed = req.parsed;
    let filters = parsed.filters;
    let semantic = parsed.semantic_query.trim().to_string();
    let keyword = parsed.keyword_query.trim().to_string();
    let candidates = cfg.search.candidates.max(1);

    // Which arms run. `mode = semantic` with an empty semantic copy is a
    // usage error; hybrid simply skips the vec arm (browse/filter mode).
    let want_vec = match req.mode {
        SearchMode::Semantic => {
            if semantic.is_empty() {
                return Err(Error::usage(
                    "mode=semantic needs query text (only filters were given)",
                ));
            }
            true
        }
        SearchMode::Hybrid => !semantic.is_empty(),
        SearchMode::Keyword => false,
    };
    let want_fts = req.mode != SearchMode::Semantic && !keyword.is_empty();
    let want_filename = req.mode == SearchMode::Hybrid && !keyword.is_empty();

    // Snippet match terms come from the semantic copy (operators removed).
    // Built before the vec arm moves `semantic` into the embed input.
    let terms: Vec<String> = semantic
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .collect();

    let conn = store.conn();
    let mut entries: Vec<(i64, Arm, u32)> = Vec::new();
    let mut vec_scores: HashMap<i64, f64> = HashMap::new();
    let mut fts_scores: HashMap<i64, f64> = HashMap::new();
    let mut fn_scores: HashMap<i64, f64> = HashMap::new();

    if want_vec {
        // A file-scoped error makes no sense for a query: escalate to fatal.
        let query_vec = embedder
            .embed(EmbedInput::Query {
                text: semantic,
                task: parsed.task,
            })
            .map_err(|e| {
                if e.is_file_error() {
                    Error::fatal(e.to_string())
                } else {
                    e
                }
            })?;
        for (i, row) in sql::vec_arm(conn, &query_vec, &filters, candidates)?
            .into_iter()
            .enumerate()
        {
            vec_scores.insert(row.chunk_id, row.display);
            entries.push((row.chunk_id, Arm::Vec, i as u32 + 1));
        }
    }
    if want_fts {
        for (i, row) in sql::fts_arm(conn, &keyword, &filters, candidates, None)?
            .into_iter()
            .enumerate()
        {
            fts_scores.insert(row.chunk_id, row.display);
            entries.push((row.chunk_id, Arm::Fts, i as u32 + 1));
        }
    }
    if want_filename {
        for (i, row) in sql::fts_arm(conn, &keyword, &filters, candidates, Some("{filename path}"))?
            .into_iter()
            .enumerate()
        {
            fn_scores.insert(row.chunk_id, row.display);
            entries.push((row.chunk_id, Arm::Filename, i as u32 + 1));
        }
    }

    let fused = rrf(
        &entries,
        cfg.search.rrf_k,
        cfg.search.weight_vec,
        cfg.search.weight_fts,
        cfg.search.weight_filename,
    );

    let mut hits = Vec::new();
    for (chunk_id, score) in fused {
        if score < req.min_score {
            continue;
        }
        if hits.len() >= req.limit {
            break;
        }
        hits.push(load_hit(
            conn,
            chunk_id,
            score,
            &vec_scores,
            &fts_scores,
            &fn_scores,
            &terms,
        )?);
    }

    Ok(SearchOutput {
        query: parsed.nl_query,
        task: parsed.task,
        mode: req.mode,
        limit: req.limit,
        min_score: req.min_score,
        filters,
        hits,
    })
}

/// Fetch one hit's row data and build its snippet.
fn load_hit(
    conn: &rusqlite::Connection,
    chunk_id: i64,
    score: f64,
    vec_scores: &HashMap<i64, f64>,
    fts_scores: &HashMap<i64, f64>,
    fn_scores: &HashMap<i64, f64>,
    terms: &[String],
) -> Result<Hit> {
    let row = conn
        .query_row(
            "SELECT f.path, f.kind, c.kind, f.collection, f.ext, f.mtime, c.page, c.ordinal, c.text
             FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.id = ?",
            [chunk_id],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, i64>(5)?,
                    r.get::<_, Option<i64>>(6)?,
                    r.get::<_, i64>(7)?,
                    r.get::<_, Option<String>>(8)?,
                ))
            },
        )
        .optional()?
        .ok_or_else(|| Error::fatal(format!("chunk {chunk_id} vanished mid-search")))?;
    let (path, kind, chunk_kind, collection, ext, mtime, page, ordinal, text) = row;

    let snippet = match text.filter(|t| !t.trim().is_empty()) {
        Some(t) => snippet(&t, terms),
        None => basename(&path),
    };

    Ok(Hit {
        path: path.clone(),
        kind: FileKind::from_str(&kind).map_err(|e| Error::fatal(format!("files.kind: {e}")))?,
        chunk_kind: ChunkKind::from_str(&chunk_kind)
            .map_err(|e| Error::fatal(format!("chunks.kind: {e}")))?,
        collection,
        ext,
        page,
        ordinal,
        mtime,
        score,
        scores: HitScores {
            vec: vec_scores.get(&chunk_id).copied(),
            fts: fts_scores.get(&chunk_id).copied(),
            filename: fn_scores.get(&chunk_id).copied(),
        },
        snippet,
    })
}

/// A basename fallback snippet (spec §6.4: null/empty chunk text).
fn basename(path: &str) -> String {
    PathBuf::from(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// ≤200 characters from `text` around the first query-term match window;
/// with no match, the first 200 characters (spec §6.4).
fn snippet(text: &str, terms: &[String]) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= 200 {
        return text.to_string();
    }
    let lower: Vec<char> = text.to_lowercase().chars().collect();
    let mut best: Option<usize> = None;
    for term in terms {
        let needle: Vec<char> = term.chars().collect();
        if needle.is_empty() {
            continue;
        }
        if let Some(pos) = find_subslice(&lower, &needle) {
            best = Some(best.map_or(pos, |b: usize| b.min(pos)));
        }
    }
    let start = best.map_or(0, |p| p.saturating_sub(60));
    let end = (start + 200).min(chars.len());
    chars[start..end].iter().collect()
}

fn find_subslice(hay: &[char], needle: &[char]) -> Option<usize> {
    if needle.len() > hay.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}
