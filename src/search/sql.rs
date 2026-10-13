//! Per-arm SQL for search (spec §6.1).
//!
//! Filters apply in SQL before ranking, never in Rust after fetch:
//! `kind` / `ext` / `mtime` / `collection` are vec0 metadata (true KNN
//! prefilter); `under` / `name` cannot be evaluated during KNN, so when
//! either is set the vec arm inflates `k`, JOINs `files`, drops mismatches,
//! and keeps at most `candidates` — never a second KNN loop.
//!
//! `ferret_basename(path)` is the scalar function `store::open` registers;
//! `--name` GLOBs the basename and stays case-sensitive.

use rusqlite::types::Value;

use crate::search::Filters;

/// One candidate row from an arm. `display` is the arm's display score
/// (vec0 L2 distance as cosine, or raw FTS5 `rank`); ranks come from the
/// row's position (1-based), never from the score.
pub struct ArmRow {
    pub chunk_id: i64,
    pub display: f64,
    /// Full chunk text with FTS matches wrapped in \x1f…\x1e sentinels
    /// (`highlight()` column 0; `None` for the vec arm, which has no match
    /// info). `offsets()` is unusable on this SQLite (3.53.3 refuses it in
    /// every context), so spans derive from these markers instead.
    pub highlight: Option<String>,
}

/// Escape GLOB metacharacters so a literal string can be embedded in a
/// pattern (`/a[b` must not start a character class).
fn glob_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        match c {
            '*' | '?' | '[' => {
                out.push('[');
                out.push(c);
                out.push(']');
            }
            _ => out.push(c),
        }
    }
    out
}

/// Little-endian f32 blob for a sqlite-vec `MATCH` argument.
fn vec_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Append the vec0-metadata / files-JOIN filter fragments shared by the
/// vec arm. `use_files` selects the `f.` column prefix; without it the
/// columns are the vec0 metadata (`file_kind`, not `kind`).
fn push_metadata(sql: &mut String, params: &mut Vec<Value>, use_files: bool, filters: &Filters) {
    let col = |name: &str| {
        if use_files {
            format!("f.{name}")
        } else if name == "kind" {
            // vec0 metadata column name (files.kind is `kind`).
            "file_kind".to_string()
        } else {
            name.to_string()
        }
    };
    if !filters.kind.is_empty() {
        let ph = vec!["?"; filters.kind.len()].join(", ");
        sql.push_str(&format!(" AND {} IN ({ph})", col("kind")));
        for k in &filters.kind {
            params.push(Value::Text(k.to_string()));
        }
    }
    if !filters.ext.is_empty() {
        let ph = vec!["?"; filters.ext.len()].join(", ");
        sql.push_str(&format!(" AND {} IN ({ph})", col("ext")));
        for e in &filters.ext {
            params.push(Value::Text(e.clone()));
        }
    }
    if let Some(after) = filters.after {
        sql.push_str(&format!(" AND {} >= ?", col("mtime")));
        params.push(Value::Integer(after));
    }
    if let Some(before) = filters.before {
        sql.push_str(&format!(" AND {} < ?", col("mtime")));
        params.push(Value::Integer(before));
    }
    if let Some(collection) = &filters.collection {
        sql.push_str(&format!(" AND {} = ?", col("collection")));
        params.push(Value::Text(collection.clone()));
    }
    if use_files {
        if let Some(under) = &filters.under {
            // Canonical prefix; trailing '/' trimmed so `path = u` and
            // `path GLOB u||'/*'` cover the root itself and its contents.
            // The equality leg binds the raw path; only the GLOB leg is
            // escaped (escaping `=` would break literal `*?[` paths).
            let u = under.to_string_lossy().trim_end_matches('/').to_string();
            sql.push_str(" AND (f.path = ? OR f.path GLOB ? || '/*')");
            params.push(Value::Text(u.clone()));
            params.push(Value::Text(glob_escape(&u)));
        }
        if let Some(name) = &filters.name {
            // Basename only; the user's GLOB is used as given (case-sensitive).
            sql.push_str(" AND ferret_basename(f.path) GLOB ?");
            params.push(Value::Text(name.clone()));
        }
    }
}

/// Vec arm: KNN over `chunks_vec` with metadata prefilter; JOIN `files` and
/// inflate `k` only when `--under` / `--name` is set.
pub fn vec_arm(
    conn: &rusqlite::Connection,
    embedding: &[f32],
    filters: &Filters,
    candidates: usize,
) -> Result<Vec<ArmRow>, rusqlite::Error> {
    let joined = filters.under.is_some() || filters.name.is_some();
    // Inflated KNN width for path-filtered searches, never below the
    // requested pool and capped to bound the join cost.
    let k: i64 = if joined {
        (candidates as i64).max((candidates as i64 * 8).min(400))
    } else {
        candidates as i64
    };

    let mut sql = String::from("SELECT cv.rowid, cv.distance FROM chunks_vec cv");
    let mut params: Vec<Value> = vec![Value::Blob(vec_blob(embedding)), Value::Integer(k)];
    if joined {
        sql.push_str(" JOIN chunks c ON c.id = cv.rowid JOIN files f ON f.id = c.file_id");
    }
    sql.push_str(" WHERE cv.embedding MATCH ? AND cv.k = ?");
    push_metadata(&mut sql, &mut params, joined, filters);
    sql.push_str(" ORDER BY cv.distance");
    if joined {
        sql.push_str(" LIMIT ?");
        params.push(Value::Integer(candidates as i64));
    }

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |r| {
        let d: f64 = r.get(1)?;
        // Vectors are L2-normalized, so cosine = 1 - d²/2, clamped.
        let cosine = (1.0 - (d * d) / 2.0).clamp(0.0, 1.0);
        Ok(ArmRow {
            chunk_id: r.get(0)?,
            display: cosine,
            highlight: None,
        })
    })?;
    rows.collect()
}

/// Keyword arm: FTS5 MATCH. `column_filter` is an FTS5 column selector like
/// `{filename path}` for the filename arm (matches only those columns),
/// or `None` for the plain body match. Candidates are capped at
/// `candidates` after ordering by FTS5 `rank`.
pub fn fts_arm(
    conn: &rusqlite::Connection,
    query: &str,
    filters: &Filters,
    candidates: usize,
    column_filter: Option<&str>,
) -> Result<Vec<ArmRow>, rusqlite::Error> {
    let match_arg = match column_filter {
        // FTS5 column filters require the colon: `{a b}: query`.
        Some(cols) => format!("{cols}: {query}"),
        None => query.to_string(),
    };
    // `highlight()` (unlike `offsets()`) is legal here; markers use ASCII
    // control chars. Column 0 is the chunk text (schema order).
    let mut sql = String::from(
        "SELECT m.rowid, m.rank, m.h FROM \
         (SELECT rowid, rank, highlight(chunks_fts, 0, char(31), char(30)) AS h \
         FROM chunks_fts WHERE chunks_fts MATCH ?) m \
         JOIN chunks c ON c.id = m.rowid JOIN files f ON f.id = c.file_id \
         WHERE 1 = 1",
    );
    let mut params: Vec<Value> = vec![Value::Text(match_arg)];
    push_metadata(&mut sql, &mut params, true, filters);
    sql.push_str(" ORDER BY m.rank LIMIT ?");
    params.push(Value::Integer(candidates as i64));

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |r| {
        Ok(ArmRow {
            chunk_id: r.get(0)?,
            display: r.get(1)?,
            highlight: r.get(2)?,
        })
    })?;
    rows.collect()
}

/// Escape a GLOB pattern's metacharacters in `raw`. Public for tests.
pub fn glob_escape_for_test(raw: &str) -> String {
    glob_escape(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_escape_neutralizes_metacharacters() {
        assert_eq!(glob_escape("/a/b"), "/a/b");
        assert_eq!(glob_escape("/a[b*?"), "/a[[]b[*][?]");
    }

    #[test]
    fn blob_is_little_endian_f32() {
        assert_eq!(vec_blob(&[1.0]), vec![0, 0, 128, 63]);
        assert_eq!(vec_blob(&[]), Vec::<u8>::new());
    }
}
