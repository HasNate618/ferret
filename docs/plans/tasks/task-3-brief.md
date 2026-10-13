### Task 3: CLI stubs — clap, init, config show/path, exit codes

**Files:**
- Create: `src/cli.rs`
- Modify: `src/main.rs`, `Cargo.toml` (clap with `derive`)
- Test: `tests/cli_init.rs`

**Interfaces:**
- Consumes: `config::{load, compiled_defaults, to_display_toml, LoadOpts}`, `paths::ferret_dirs`, `Error`
- Produces: `cli::run() -> Result<()>` (caller in `main` prints `Display` on stderr and `process::exit(err.exit_code())`)
- Produces clap `Cli`: global `--config FILE`, `--index PATH`; subcommands `init { roots: Vec<PathBuf> /* --root, Append */, force: bool }`, `index { paths: Vec<PathBuf>, json: bool, strict: bool }`, `search { query: Option<String>, json: bool, kind: Vec<String>, ext: Vec<String>, under: Option<PathBuf>, collection: Option<String>, name: Option<String>, after: Option<String>, before: Option<String>, min_score: Option<f64>, limit: Option<usize>, mode: Option<SearchMode>, code: bool }`, `rm { paths: Vec<PathBuf> }`, `gc`, `status { json: bool }`, `config { ConfigCmd::Show | Path }`
- Search flags: `--kind`, `--ext`, `--under`, `--collection`, `--name`, `--after`, `--before`, `--min-score`, `--limit`, `--mode`, `--code` (bool). Repeatable `--kind` / `--ext` union values (clap `ArgAction::Append`, then split on comma and merge).
- `ferret search` with neither QUERY nor any filter flag → `Error::usage`. If QUERY omitted and stdin is not a TTY, read one line; if stdin is a TTY, do not prompt.

`main.rs`:

```rust
fn main() {
    if let Err(err) = ferret::cli::run() {
        eprintln!("{err}");
        std::process::exit(err.exit_code());
    }
}
```

- [ ] **Step 1: Write failing CLI tests**

Use `assert_cmd` + `predicates` (add as dev-deps) **or** call `cli::run_with_args(args: &[&str], stdin, is_tty) -> Result<(i32_would, stdout, stderr)>` without spawning. Prefer an internal `run_with` so tests do not need a built binary for every case:

```rust
pub struct RunOutput { pub stdout: String, pub stderr: String }

pub fn try_run(args: &[&str]) -> ferret::error::Result<RunOutput>;
pub fn try_run_with(args: &[&str], stdin: &str, is_tty: bool) -> ferret::error::Result<RunOutput>;

#[test]
fn search_with_no_query_or_filters_is_usage() {
    let err = ferret::cli::try_run(&["ferret", "search"]).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn init_refuses_existing_without_force() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("XDG_CONFIG_HOME", tmp.path());
    ferret::cli::try_run(&["ferret", "init"]).unwrap();
    let err = ferret::cli::try_run(&["ferret", "init"]).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn config_path_prints_defaults_layer() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("XDG_CONFIG_HOME", tmp.path());
    let out = ferret::cli::try_run(&["ferret", "config", "path"]).unwrap();
    assert!(out.stdout.contains("defaults"));
}

#[test]
fn init_writes_roots_and_not_api_key() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("XDG_CONFIG_HOME", tmp.path());
    ferret::cli::try_run(&["ferret", "init", "--root", "/tmp/docs"]).unwrap();
    let body = std::fs::read_to_string(tmp.path().join("ferret/config.toml")).unwrap();
    assert!(body.contains("roots"));
    assert!(!body.contains("api_key"));
    assert!(!body.contains("multimodal"));
}
```

Do not call `set_var` from parallel tests without isolating: add `serial_test` **or** pass dirs into `try_run` via `FERRET_TEST_DIRS` later. Prefer `try_run` constructing `FerretDirs` from `XDG_*` already set in the test process, and mark these tests `#[serial]` (`serial_test = "3"` dev-dep).

`init` behavior: create `{config_dir}` (0700), write short TOML with `roots` from `--root` (repeatable), refuse if file exists unless `--force`. Do **not** write `api_key` or `multimodal`. Do not hit the network. If compiled/default backend is local, no upload warning.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test cli_init`
Expected: FAIL

- [ ] **Step 3: Implement clap + init + config show/path**

Other subcommands may return `Error::fatal("not implemented")` **only if** tests do not call them yet. Prefer empty stubs that return usage-ok no-ops for `gc`/`rm` so Task 13 can fill them. `index`/`search`/`status` wait for later tasks.

`config show`: merged TOML, comments stripped, `api_key` redacted.

`config path`: loaded file paths one per line, then `defaults` / `env` / `flags` if those layers contributed.

`--help` for search must include: `--under` / `--name` can starve the semantic arm on a large index; hybrid still has FTS. `--name` is case-sensitive GLOB.

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/cli.rs src/main.rs tests/cli_init.rs
git commit -m "$(cat <<'EOF'
feat: add ferret CLI scaffolding, init, and config show

EOF
)"
```

---

