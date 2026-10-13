# Task 3 Report: CLI stubs — clap, init, config show/path, exit codes

Status: **DONE**
Commit: `4cef1f1` (on top of Task 2's `b81c731`)

## What was implemented

### `src/cli.rs` (new)

- `RunOutput { stdout, stderr }` and `try_run` / `try_run_with` — signatures transcribed
  verbatim from the brief. `try_run(args)` delegates with empty non-TTY stdin. Both run
  the CLI in-process into buffers (no binary spawn). `RunOutput` derives `Debug` (needed
  by `unwrap_err()`; not spelled out in the brief).
- `run()` — real argv, real stdin (one line when not a TTY), real stdout/stderr.
- clap `Cli` exactly as the brief lists it: global `--config FILE`, `--index PATH`;
  subcommands `init { roots (--root, Append), force }`, `index { paths, json, strict }`,
  `search { query, json, kind, ext, under, collection, name, after, before, min_score,
  limit, mode: Option<SearchMode>, code }`, `rm { paths }`, `gc`, `status { json }`,
  `config { Show | Path }`. All fields exist even where behavior lands later.
- `--help` / `--version` render to stdout and exit 0 (via `Cli::try_parse_from`, no
  process exit — works under `try_run`). Any other clap error (unknown flag, missing
  subcommand) → `Error::usage` (exit 2). `search --help` includes the required notes:
  `--under` / `--name` can starve the semantic arm on a large index (hybrid still has
  FTS), and `--name` is a case-sensitive GLOB.
- `search` stub: when QUERY is omitted and stdin is not a TTY, reads one line (never
  prompts when it is a TTY); neither QUERY nor any filter flag (`--kind/--ext/--under/
  --collection/--name/--after/--before/--min-score/--limit/--mode/--code`) →
  `Error::usage` (exit 2). Valid invocations return `Error::fatal("not implemented")`
  (search behavior is Task 16). `index` / `status` also return
  `Error::fatal("not implemented")`. `rm` / `gc` are usage-ok no-op stubs (Task 13
  fills them), per the brief's preference.
- `init`: creates `{config_dir}` (mode 0700 on unix), refuses an existing
  `{config_dir}/config.toml` with `Error::usage` (exit 2) unless `--force`, and writes a
  short TOML — `[index]\nroots = [...]` shorthand from `--root` (repeatable, empty array
  when none). Never writes `api_key` or `multimodal`, never touches the network.
  Duplicate collection names (last component of the canonical root) → `Error::fatal`
  (exit 1), per plan line "collision at start of index/init → exit 1".
- `config show`: `config::load` with the CLI flags → `to_display_toml` (api_key
  redacted) on stdout. `--config` pointing at a missing file → exit 1 (verified).
- `config path`: file layers' paths one per line (in load order), then `defaults` /
  `env` / `flags` lines for non-file layers that contributed (design §7.1 wording:
  "loaded file paths one per line, then defaults / env / flags"). Uses a local
  `load_with_layers` helper wiring `config::load_from` the same way `config::load`
  does (load does not return layers).

### `src/main.rs`, `src/lib.rs`

- `main` exactly as the brief: print `Display` on stderr, `process::exit(err.exit_code())`.
- `pub mod cli;` added.

### `Cargo.toml`

- `clap = { version = "4.6", features = ["derive"] }` (plan specifies clap 4.6; resolved
  4.6.7) — no other dependency versions touched.
- Dev-dep `serial_test = "3"` for `#[serial]` on tests that call `std::env::set_var`.
- `Cargo.lock` updated accordingly and committed (tracked file; required for a
  reproducible build of the new deps).

## Exit-code matrix (verified by tests + release-binary smoke test)

| Invocation | Exit |
| --- | --- |
| `init`, `init --root DIR...`, `init --force` | 0 |
| `init` onto existing file (no `--force`) | 2 |
| `init` with colliding collection names | 1 |
| `config show` / `config path` | 0 |
| `--config` missing file | 1 |
| `search` with no QUERY and no filter flag (empty non-TTY stdin) | 2 |
| unknown flag (`--bogus`) | 2 |
| `gc` / `rm PATH` (stubs) | 0 |
| `index` / `status` (stubs) | 1 ("not implemented") |
| `--help` / `--version` | 0 (rendered on stdout) |

## Files changed

- Created: `src/cli.rs`, `tests/cli_init.rs`
- Modified: `src/main.rs`, `src/lib.rs`, `Cargo.toml`, `Cargo.lock`
- Committed as `4cef1f1`: `feat: add ferret CLI scaffolding, init, and config show`.

## Test results

Command: `cargo test` (full suite)

```
test result: ok. 2 passed; 0 failed; 0 ignored   (lib unit tests)
test result: ok. 12 passed; 0 failed; 0 ignored  (tests/cli_init.rs)
test result: ok. 16 passed; 0 failed; 0 ignored  (tests/config_merge.rs)
test result: ok. 3 passed; 0 failed; 0 ignored   (tests/paths.rs)
```

`cargo test --test cli_init` tail:

```
test config_path_lists_user_file_then_layers ... ok
test config_show_redacts_api_key ... ok
test init_creates_config_dir_private_and_loadable ... ok
test search_with_no_query_or_filters_is_usage ... ok
test unknown_flag_is_usage ... ok
...
test result: ok. 12 passed; 0 failed; 0 measured; 0 filtered out; finished in 0.01s
```

`tests/cli_init.rs` — the brief's four tests verbatim (only mechanical change:
`unsafe { std::env::set_var(..) }` because `set_var` is unsafe in edition 2024) plus
eight additional tests: help/version rendering (incl. `search --help` starve/GLOB
notes), unknown flag → 2, missing `--config` file → 1, `config show` api_key
redaction, `config path` user-file + `defaults` ordering, init 0700 dir +
round-trip load + `--force` overwrite + never api_key/multimodal, colliding
collections → 1, `rm`/`gc` stubs exit 0. All env-mutating tests are `#[serial]`.
Also smoke-tested the release binary end-to-end (`init`, `config path/show`,
`search` usage, `gc`, `--bogus`, `--version`, exit codes, 0700 dir, written TOML).

`cargo clippy --all-targets`: clean. `cargo fmt --check`: clean for all Task 3 files.

## Self-review findings and fixes

1. First build: `clap::Error::render()` returns `StyledStr` in clap 4.6 (String in ≤4.5)
   — used `e.to_string()` for the message instead.
2. `Error::fatal` takes `impl Into<String>`, so `map_err(Error::fatal)` on an
   `io::Error` doesn't compile — spelled out `map_err(|e| Error::fatal(e.to_string()))`.
3. `config path` initially printed layers in merge order (defaults first); the design
   spec (§7.1) says file paths first, then `defaults`/`env`/`flags` — fixed to two passes.
4. A stray no-op loop and an unused `Path` import were left in `cli.rs` by an early
   draft; removed.
5. Running repo-wide `cargo fmt` reformatted Task 2's `src/config.rs`, `src/types.rs`,
   `tests/config_merge.rs` (formatting-only). Reverted those to HEAD so this commit
   stays scoped to Task 3; see residual risks.
6. `enum Command` trips clippy `large_enum_variant` (flat `Search` variant); added
   `#[allow(clippy::large_enum_variant)]` rather than boxing — the field list is fixed
   by the brief.

## Residual risks / notes

- Task 2's committed files are not rustfmt-clean: `cargo fmt` would produce the
  formatting-only diffs I reverted. Recommend a small follow-up `style: cargo fmt`
  commit so future tasks' fmt runs don't dirty unrelated files.
- `ferret` with no subcommand → clap "missing subcommand" → exit 2 (usage), which
  matches the global exit-code table; bare `--help`/`--version` exit 0 as required.
- The brief's TDD Step 2 (run tests, watch them fail) was folded into the
  implementation pass — the first `cargo test` compile failed before the module
  existed, then tests failed until `cli.rs` was complete; final state is green.
- `rm`/`gc` silently succeed as no-ops until Task 13; `index`/`search`/`status`
  return exit 1 "not implemented" until Tasks 11/16/13. All clap fields for those
  commands already exist.
- `/etc/ferret/config.toml` is hardcoded in `cli.rs`'s `config path` printer, matching
  the identical hardcode in `config.rs` (consistent behavior on non-unix: never reads).
