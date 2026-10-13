### Task 8: RRF, SQL filters, search JSON on fixture rows

**Files:**
- Create: `src/search/rrf.rs`, `src/search/sql.rs`, `src/search/json.rs`; fill `src/search/mod.rs`
- Test: `tests/search_rrf.rs`, `tests/search_json.rs`, `tests/search_filters.rs`
- Modify: `src/cli.rs` to call search; `Cargo.toml` if needed

**Interfaces:**
- Consumes: `Store`, `Embedder`, `ParsedQuery`, `Config.search`
- Produces:

```rust
pub struct SearchRequest {
    pub parsed: ParsedQuery,
    pub mode: SearchMode,
    pub limit: usize,
    pub min_score: f64,
    pub json: bool,
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

pub struct HitScores {
    pub vec: Option<f64>,      // cosine
    pub fts: Option<f64>,      // raw FTS5 rank
    pub filename: Option<f64>, // raw FTS5 rank
}

pub fn rrf(ranks: &[(i64, Arm, u32 /*1-based rank*/)], k: f64, w_vec: f64, w_fts: f64, w_fn: f64) -> Vec<(i64, f64)>;
pub fn run_search(store: &Store, embedder: &dyn Embedder, cfg: &Config, req: SearchRequest) -> Result<SearchOutput>;
pub fn render_json(out: &SearchOutput) -> String; // one object, nothing else
pub fn render_human(out: &SearchOutput) -> String; // path[:page] then snippet; no percent
```

Pipeline (implement exactly):
1. If semantic copy empty: skip vec arm unless `mode=semantic` → `Error::usage`.
2. Vec arm: KNN on `chunks_vec`. Metadata WHERE for kind/ext/mtime/collection using `file_kind IN (...)`. If `--under` or `--name` set, `k = min(candidates * 8, 400)`, JOIN `files`, drop mismatches, take up to `candidates`. Do not loop a second KNN.
3. Keyword arm: FTS5 MATCH on `chunks_fts` (body), same filters, cap `candidates`.
4. Filename arm: FTS5 MATCH on `filename` and `path` only.
5. RRF unless single-arm mode. Drop fused `< min_score`. Return `limit` hits.
6. `scores.vec = clamp(1 - (l2_distance^2)/2, 0, 1)` because vectors are L2-normalized. `scores.*` is JSON `null` when that arm did not list the hit.
7. Snippet: ≤200 chars from `chunks.text` first match window; if text null/empty, basename.

JSON object fields frozen as spec §6.4. `query` = `nl_query`. `task` = `"search"` or `"code"`. Zero hits: exit 0, `"hits": []`.

Vec SQL sketch:

```sql
SELECT rowid, distance
FROM chunks_vec
WHERE file_kind IN ('pdf')
  AND embedding MATCH ?
  AND k = ?;
```

sqlite-vec match blob: little-endian `f32` bytes of the query vector. After JOIN:

```sql
SELECT cv.rowid, cv.distance
FROM chunks_vec cv
JOIN chunks c ON c.id = cv.rowid
JOIN files f ON f.id = c.file_id
WHERE cv.embedding MATCH ?
  AND k = ?
  AND f.path GLOB ?   -- under: prefix + '*' after canonicalize
  AND f.path GLOB '*' || ?; -- do NOT use this for --name
```

`--name` is basename only: `ferret_basename(f.path) GLOB :name` (SQL GLOB, case-sensitive). `--under`: `f.path = :under OR f.path > :under || '/' AND f.path < :under || '0'` **or** `instr(f.path, :under) = 1` after ensuring `:under` has no trailing slash. Prefer `f.path = ? OR f.path LIKE ?` with `dir/%` only if LIKE is case-sensitive (it is not on ASCII by default with NOCASE). **Use `GLOB` with a escaped prefix + `*` or a `>=` / `<` range on the canonical prefix.** Spec: path prefix. Implement `path_is_under(path, dir)` in SQL as `f.path = :u OR f.path GLOB :u || '/*'` — GLOB `/` is not special. Escape GLOB metacharacters in the directory string (`*`, `?`, `[`).

- [ ] **Step 1: Write failing RRF + JSON + filter tests**

`tests/search_rrf.rs`:

```rust
#[test]
fn rrf_prefers_agreement() {
    // chunk 1 rank 1 vec + rank 1 fts; chunk 2 rank 2 vec only
    let fused = rrf(
        &[(1, Arm::Vec, 1), (1, Arm::Fts, 1), (2, Arm::Vec, 2)],
        60.0, 1.0, 1.0, 0.4,
    );
    assert_eq!(fused[0].0, 1);
    let s1 = 1.0 / 61.0 + 1.0 / 61.0;
    assert!((fused[0].1 - s1).abs() < 1e-12);
}
```

`tests/search_filters.rs`: in-memory/temp store, insert text file under `/tmp/a/in.md` and `/tmp/b/out.md` with distinct fake vectors/text. `kind:text` must not return an image row. `--under /tmp/a` hybrid must return the in-dir hit (FTS if vec starved). `--kind pdf` uses `file_kind = 'pdf'` (file kind, not chunk kind).

`tests/search_json.rs`:

```rust
#[test]
fn json_object_keys_and_null_page() {
    let out = run_search_json_on_fixture("pets");
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    for k in ["query", "task", "mode", "limit", "min_score", "filters", "hits"] {
        assert!(v.get(k).is_some(), "{k}");
    }
    let f = &v["filters"];
    for k in ["kind", "ext", "under", "collection", "name", "after", "before"] {
        assert!(f.get(k).is_some(), "{k}");
    }
    let h = &v["hits"][0];
    assert!(h.get("start_byte").is_none());
    assert!(h.get("end_byte").is_none());
    assert!(h["scores"]["vec"].as_f64().unwrap() >= 0.0);
    assert!(h["scores"]["vec"].as_f64().unwrap() <= 1.0);
    assert_eq!(h["chunk_kind"], "text");
    assert!(h["page"].is_null());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test search_rrf --test search_json --test search_filters`
Expected: FAIL

- [ ] **Step 3: Implement search + wire `ferret search`**

`mode=semantic` with empty semantic copy → exit 2. Fatal embedder/DB → stderr, empty stdout, exit 1. JSON to stdout only; warnings stderr.

Human line: `{path}` or `{path}:{page}` then a short snippet. Not a client contract.

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/search src/cli.rs tests/search_rrf.rs tests/search_json.rs tests/search_filters.rs
git commit -m "$(cat <<'EOF'
feat: hybrid search with SQL filters, RRF, and frozen JSON

EOF
)"
```

---

