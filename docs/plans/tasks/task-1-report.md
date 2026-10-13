# Task 1 Report: Crate scaffold, errors, path policy

**Status:** DONE (with two environment repairs, both operator-approved — see Concerns)

**Commits:**
- `bb47c7f` — feat: scaffold ferret crate with XDG-aware path policy (8 files, 295 insertions)

## What was implemented

Exactly per `task-1-brief.md`, transcribed verbatim:

- `Cargo.toml` — crate `ferret` 0.1.0, edition 2024, rust-version 1.85, deps `thiserror = "2"`, `dirs = "6"` (no extract/embed crates, as required). `dirs` resolved to 6.0.0; `thiserror` to 2.0.21.
- `.gitignore` — `/target`, `/result`, `.direnv`.
- `src/error.rs` — `Error::{Usage, Fatal}`, `usage`/`fatal` constructors, `exit_code()` (2 for Usage, 1 otherwise), `Display`, `std::error::Error`, `From<std::io::Error>`, `pub type Result<T>`.
- `src/paths.rs` — `FerretDirs {config_dir, data_dir, cache_dir}`, `pub const INDEX_FILENAME: &str = "index.sqlite"`, `ferret_dirs(env: &dyn Fn(&str) -> Option<String>)` where any set/non-empty `XDG_*` var wins on every OS; otherwise platform defaults via `dirs`, with macOS config parent `~/Library/Preferences` (not Application Support).
- `src/lib.rs` — `pub mod error; pub mod paths;`
- `src/main.rs` — stub `main` printing `ferret: not implemented`, exit 1.
- `tests/paths.rs` — the three brief tests, verbatim.

TDD sequence followed: test written first, `cargo test --test paths` failed (`error: could not find Cargo.toml`), then implementation, then pass.

## Test results

Command: `cargo test --test paths` (also ran full `cargo test`)

```
running 3 tests
test xdg_config_and_cache_win ... ok
test xdg_data_home_wins_on_any_os ... ok
test missing_xdg_uses_platform_defaults ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Full `cargo test` (lib, bin, integration, doc-tests): all pass, 0 failures. `cargo build`: no warnings.

## Self-review findings and fixes

1. **Broken rust toolchain (environment, not task code).** Both installed rustup toolchains' nix-patched `gcc-ld` wrapper scripts exec'd a garbage-collected nix store path (`319n2f2...-rustup-1.29.0/nix-support/ld-wrapper.sh` — missing), so every cargo build failed at link. Workarounds via `RUSTFLAGS` (`-C link-self-contained=-linker`, `-C linker=...unwrapped/ld.lld -C linker-flavor=ld.lld`) failed. **Fix (operator-approved):** sed'd the dead hash → live rustup store path `0l25mxqz...` in all 8 wrappers (`~/.rustup/toolchains/{stable,1.96.1}/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld/{ld64.lld,ld.lld,lld-link,wasm-ld}`). Reversible; no source changes needed.
2. **Repo was mid-interactive-rebase.** master had an in-progress rebase (`edit 61a3642`, no commands remaining), contradicting the given context ("clean, one commit"). **Fix (operator-approved):** `git rebase --continue` (nothing staged, so it completed the rebase without changes; master = `61a3642` on new root). Untracked `docs/plans/` files (controller's) left untouched.
3. Brief step 5's `git add` list omitted `Cargo.lock`; supervisor instructed including it — included (pinned via caret-compatible resolution).
4. Verified file contents against the brief line-by-line after writing; no drift.

## Concerns

- The rustup toolchain repair was an out-of-repo environment change. If the nix rustup package is ever rebuilt/reinstalled, the wrappers will be regenerated correctly; if the live `0l25mxqz...` store path is ever GC'd, the same breakage returns. A `nixos-rebuild`/profile update of rustup would be the durable fix (out of scope).
- `dirs = "6"` per the brief (as instructed, not changed). If a later task needs `dirs` 7 features, the bump is that task's decision.
