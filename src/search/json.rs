//! Search output rendering (spec §6.4).
//!
//! JSON is a frozen client contract: one object on stdout, nothing else.
//! Warnings and progress go to stderr. Human mode is not a contract.

use serde_json::{Value, json};

use crate::search::SearchOutput;

/// Render the frozen JSON object. `scores.*` is `null` when that arm did
/// not list the hit; `page` is `null` for non-PDF.
pub fn render_json(out: &SearchOutput) -> String {
    let filters = json!({
        "kind": out.filters.kind.iter().map(|k| k.to_string()).collect::<Vec<_>>(),
        "ext": out.filters.ext,
        "under": out.filters.under.as_ref().map(|p| p.to_string_lossy().into_owned()),
        "collection": out.filters.collection,
        "name": out.filters.name,
        "after": out.filters.after,
        "before": out.filters.before,
    });
    let hits: Vec<Value> = out
        .hits
        .iter()
        .map(|h| {
            json!({
                "path": h.path,
                "kind": h.kind.to_string(),
                "chunk_kind": h.chunk_kind.to_string(),
                "collection": h.collection,
                "ext": h.ext,
                "page": h.page,
                "ordinal": h.ordinal,
                "mtime": h.mtime,
                "score": h.score,
                "scores": {
                    "vec": h.scores.vec,
                    "fts": h.scores.fts,
                    "filename": h.scores.filename,
                },
                "snippet": h.snippet,
                "snippet_spans": h
                    .snippet_spans
                    .iter()
                    .map(|(start, end)| json!({ "start": start, "end": end }))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    json!({
        "query": out.query,
        "task": out.task.to_string(),
        "mode": out.mode.to_string(),
        "limit": out.limit,
        "min_score": out.min_score,
        "filters": filters,
        "hits": hits,
        "warnings": out.warnings,
    })
    .to_string()
}

/// Human mode: one hit per line, `path[:page]` then the snippet. The fused
/// RRF score is never printed as a percent.
pub fn render_human(out: &SearchOutput) -> String {
    out.hits
        .iter()
        .map(|h| {
            let loc = match h.page {
                Some(p) => format!("{}:{p}", h.path),
                None => h.path.clone(),
            };
            let snippet = h.snippet.lines().next().unwrap_or("").trim();
            if snippet.is_empty() {
                loc
            } else {
                format!("{loc}  {snippet}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{Filters, Hit, HitScores, ParsedQuery, QueryTask, SearchMode};
    use crate::types::ChunkKind::PdfPage;
    use crate::types::FileKind::{Pdf, Text};
    use std::path::PathBuf;

    fn output() -> SearchOutput {
        SearchOutput {
            query: "pets".into(),
            task: QueryTask::Search,
            mode: SearchMode::Hybrid,
            limit: 20,
            min_score: 0.0,
            filters: Filters {
                kind: vec![Text],
                ext: vec![],
                under: None,
                collection: None,
                name: None,
                after: None,
                before: None,
            },
            hits: vec![Hit {
                path: "/d/lease.pdf".into(),
                kind: Pdf,
                chunk_kind: PdfPage,
                collection: "d".into(),
                ext: "pdf".into(),
                page: Some(12),
                ordinal: 11,
                mtime: 1717200000,
                score: 0.0412,
                scores: HitScores {
                    vec: Some(0.81),
                    fts: Some(-5.12),
                    filename: None,
                },
                snippet: "...pets are not permitted...".into(),
                snippet_spans: vec![(4, 8)],
            }],
            warnings: vec!["browse_no_vec"],
        }
    }

    #[test]
    fn json_null_scores_and_page() {
        let v: Value = serde_json::from_str(&render_json(&output())).unwrap();
        let h = &v["hits"][0];
        assert_eq!(h["page"], 12);
        assert_eq!(h["chunk_kind"], "pdf_page");
        assert!(h["scores"]["filename"].is_null());
        assert_eq!(h["scores"]["vec"], 0.81);
        assert_eq!(v["warnings"], serde_json::json!(["browse_no_vec"]));
        assert_eq!(v["hits"][0]["snippet_spans"], serde_json::json!([{"start": 4, "end": 8}]));
    }

    #[test]
    fn human_line_has_page_suffix_and_no_percent() {
        let line = render_human(&output());
        assert!(line.starts_with("/d/lease.pdf:12"));
        assert!(!line.contains('%'));
    }

    #[test]
    fn parsed_query_unused_in_render() {
        // render takes SearchOutput only; ParsedQuery stays an input to
        // run_search. This test pins that the types stay separate.
        let _p = ParsedQuery {
            nl_query: "x".into(),
            semantic_query: "x".into(),
            keyword_query: "x".into(),
            task: QueryTask::Search,
            filters: Filters {
                kind: vec![],
                ext: vec![],
                under: Some(PathBuf::from("/x")),
                collection: None,
                name: None,
                after: None,
                before: None,
            },
        };
    }
}
