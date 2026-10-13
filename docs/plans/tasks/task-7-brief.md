### Task 7: Search query parser

**Files:**
- Create: `src/search/mod.rs`, `src/search/parse.rs`
- Modify: `src/lib.rs`
- Test: `tests/query_parse.rs`

**Interfaces:**
- Consumes: `FileKind`, `QueryTask`, `SearchMode`, `Error`
- Produces:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Filters {
    pub kind: Vec<FileKind>,
    pub ext: Vec<String>,        // no dot, lower
    pub under: Option<PathBuf>,  // canonical; missing DIR => Error::usage (exit 2)
    pub collection: Option<String>,
    pub name: Option<String>,    // glob, as given
    pub after: Option<i64>,      // unix seconds local 00:00:00
    pub before: Option<i64>,     // next day 00:00:00 local
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedQuery {
    pub nl_query: String,         // remainder after prefix strip (JSON `query`)
    pub semantic_query: String,   // unquote phrases, drop standalone OR/AND/NOT
    pub keyword_query: String,    // remainder including quotes and operators
    pub task: QueryTask,
    pub filters: Filters,
}

pub fn parse_query(raw: &str, flag_overrides: &FilterFlags) -> Result<ParsedQuery>;
```

`FilterFlags` holds the clap-parsed flags. Flags win on conflict with prefixes. Repeatable flags union. `code:` at start of remainder or `--code` sets `task = Code`.

**Prefix tokenizer:** token at start or after whitespace, `name:` then value. `kind:pdf` and `kind: pdf` are the same. Values until next whitespace. Quoted values allowed (`path:"/home/nate/My Documents"`). Known prefixes: `kind`/`type`, `ext`, `path`/`under`, `collection`, `name`, `after`, `before`, `code`. Unknown prefixes stay in `nl_query`.

**Hyphens:** `+` / `-` are FTS operators only at the start of the remainder or after whitespace. `left-wing` and `COVID-19` are **not** subtraction. Do not rewrite them when building `keyword_query`.

**Dates:** naive local calendar `YYYY-MM-DD`. Invalid → `Error::usage`. `after` = that date 00:00:00 local; `before` = next day 00:00:00 local. Use `chrono::Local`.

`--under` / `path:`: tilde-expand, then `std::fs::canonicalize`. If DIR does not exist, `Error::usage`.

- [ ] **Step 1: Write failing tests**

```rust
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
    let flags = FilterFlags { kind: vec![FileKind::Text], ..Default::default() };
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
    let err = parse_query("after:2026-13-01 x", &FilterFlags::default()).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}
```

For `leading_minus`, semantic copy removes standalone FTS operators `OR`, `AND`, `NOT`, and `+`/`-` tokens (the operator plus its immediately-following token? Spec: "removes standalone FTS operators `OR`, `AND`, `NOT`" and `+`/`-` are operators at start/after whitespace. Semantic copy unquotes phrases (inner text kept) and removes those operators. For `-wing birds`, semantic is `birds` (operator token `-wing` removed as a `-` operator with value). Keep this explicit in `parse.rs` comments.

Quoted phrases: `"pets in the lease"` → semantic `pets in the lease`, keyword keeps quotes.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test query_parse`
Expected: FAIL

- [ ] **Step 3: Implement parser** (no regex-replace of interior hyphens)

- [ ] **Step 4: Run tests**

Run: `cargo test --test query_parse`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/search tests/query_parse.rs
git commit -m "$(cat <<'EOF'
feat: parse search prefixes without treating hyphens as FTS minus

EOF
)"
```

---

