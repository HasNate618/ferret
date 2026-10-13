# Task 2 Report: Config types, defaults, layering, validation

Status: **DONE**
Commit: `b81c731` (on top of Task 1's `bb47c7f`)

## What was implemented

### `src/types.rs` (new)

All enums live here with `Display`/`FromStr`/serde all using the spec strings:

- `FileKind { Text => "text", Image => "image", Pdf => "pdf" }`
- `ChunkKind { Text => "text", Image => "image", PdfPage => "pdf_page" }`
- `QueryTask { Search => "search", Code => "code" }`
- `Backend { LlamaServer => "llama_server", LitertLm => "litert_lm", Sglang => "sglang", Openai => "openai" }`
- `PrefixProfile { EmbeddingGemma2 => "embeddinggemma2", LitertLm => "litert_lm" }`
- `SearchMode { Hybrid => "hybrid", Semantic => "semantic", Keyword => "keyword" }`
- `TextChunkMode { Chars => "chars", Sentences => "sentences" }`
- `HeicMode { Skip => "skip", Magick => "magick" }`

A `spec_enum!` macro generates the enum + `Display` + `FromStr` (error lists the valid
values) + serde `rename` per variant. Two unit tests cover display strings and
round-trips (including `FileKind::from_str("audio")` and `Backend::from_str("ollama")`
errors).

### `src/config.rs` (new)

- All public structs exactly as the brief specifies (`Config`, `IndexConfig`, `RootEntry`,
  `EmbedderConfig`, `SearchConfig`, `ChunkConfig`/`Text`/`Pdf`/`Image`, `ExtractConfig`,
  `FilesConfig`, `LoadOpts`, `LayerSource`).
- `load(opts)` — real env (`ferret_dirs(&std::env::var)`) + real cwd + real filesystem reader.
- `load_from(&LoadOpts, dirs: &FerretDirs, env: &dyn Fn(&str) -> Option<String>, cwd: &Path, read: &dyn Fn(&Path) -> Result<String>) -> Result<(Config, Vec<LayerSource>)>` — signature transcribed verbatim (5 params).
- `compiled_defaults(dirs)` — spec §7.2 values verbatim, including `index.path = {data_dir}/index.sqlite`, default `exclude_globs`, and the full §4.1 `extension_kind` map (`.heic`/`.heif` intentionally not listed; they are walk-skipped unless `files.heic = "magick"`).
- `to_display_toml(cfg)` — merged TOML, `api_key` → `"***"` when non-empty, `multimodal` omitted when `None`.

Layer order (last-wins deep merge into a `toml::Table` before typed deserialization):
compiled defaults → `/etc/ferret/config.toml` (skippable) → `{config_dir}/config.toml` or
`--config FILE` (the flag **replaces** that layer; a read error is fatal) →
`./.ferret.toml` (skippable) → `FERRET__SECTION__KEY` env (nested sections via `__`) →
`--index` into `index.path`. `LayerSource` records only layers that contributed.

Env handling: a registry (`ENV_PATHS`) of every scalar dotted path is probed as
`FERRET__<PATH_WITH_/_→__>`; values are coerced to the field's type (`u32`/`u64`/`usize`
as TOML `i64`, `f64`, `bool`, strings/enums; enum validity is enforced at deserialize).
`FERRET_EMBEDDER` is NOT probed (special, Task 6).

Roots: both TOML shapes via an untagged `RootSpec` (`Vec<String>` shorthand /
`Vec<RawRoot{path, collection}>` explicit). Mixing both keys in one file matches neither
variant → serde error → `Error::fatal` (exit 1). Paths are tilde-expanded
(`dirs::home_dir()`), canonicalized if they exist; shorthand collection = last component
of the resulting path.

`prefix_profile` default is applied after the merged backend is known: `litert_lm` when
`backend = "litert_lm"`, else `embeddinggemma2` (covers both the file and env override
cases; verified by tests).

`files.max_size`: untagged `SizeSpec` (`Bytes(u64)` integer or `Human(String)` with
`KiB`/`MiB`/`GiB`, 1024-based, exact match — `"40MB"` is fatal). Stored as `u64` bytes.

`files.extension_kind`: user entries are merged **into** the compiled default map (the
merged map keeps every compiled default, per the §13 fingerprint requirement).

### Validation (all `Error::fatal`, exit 1)

- `chunk.text.overlap >= chunk.text.max_chars`
- `embedder.dimensions` not in `{768, 512, 256, 128}`
- two roots with the same collection name
- both root shapes in one file (untagged deserialize failure)
- invalid `chunk.text.mode` / `embedder.backend` / other enum strings (deserialize failure)
- unix: a loaded TOML with non-empty `embedder.api_key` whose file mode is more permissive
  than 0600 (`#[cfg(unix)]`, `PermissionsExt`, checks each file layer individually)

### `tests/common.rs` (new)

`test_dirs(tmp)` exactly as the brief specifies (verbatim except one line: the brief's
`use std::path::PathBuf;` import is unused at this point and triggered a dead-import
warning because cargo compiles `tests/common.rs` as its own test target; it was removed
and Task 11 can re-add it when `test_env()` lands).

### `tests/config_merge.rs` (new)

The brief's 6 tests verbatim, plus 10 self-review tests (same style, HashMap-backed
`read`) covering: `[[index.roots]]` table form + layer-source list; mixed root shapes
fatal; duplicate collections fatal; dimensions outside the set fatal; cwd file layered
over user file; `--index` flag override + `Flags` layer; `max_size` bytes/`KiB`/`MiB`/`GiB`
parse + `"40MB"` fatal; env type coercion + prefix profile derived from a merged env
backend; invalid env values fatal; display TOML redaction (empty key shown as `""`,
`multimodal` omitted).

### `src/lib.rs`, `Cargo.toml`

`pub mod config; pub mod types;`. Dependencies added exactly as pinned by the brief:
`serde = { version = "1", features = ["derive"] }`, `toml = "0.8"`, `serde_json = "1"`;
dev-dep `tempfile = "3"`. `Cargo.lock` updated and committed alongside (it is tracked
from Task 1).

## Test results

Command: `cargo test --test config_merge --test paths`

```
running 16 tests
test index_flag_overrides_and_adds_flags_layer ... ok
test litert_backend_defaults_prefix_profile ... ok
test overlap_ge_max_chars_is_fatal ... ok
test api_key_file_mode_0644_is_fatal ... ok
test roots_table_form_and_layer_sources ... ok
test max_size_parses_bytes_and_kib_mib_gib ... ok

test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 3 tests
test xdg_config_and_cache_win ... ok
test xdg_data_home_wins_on_any_os ... ok
test missing_xdg_uses_platform_defaults ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Full `cargo test` (all targets): 21 passed, 0 failed, zero warnings.
`cargo clippy --all-targets`: clean.

## Self-review findings and fixes

1. `toml::Value` has no `From<u32/u64/usize>` — numeric env values and `to_display_toml`
   integers go through `i64` (target field types are still enforced when the merged table
   deserializes into `RawConfig`; verified by `invalid_env_values_are_fatal`).
2. Initial run failed: raw section structs required every section key present
   (`missing field index` when a file only had `[embedder]`). Fixed by `#[serde(default)]`
   on `RawConfig` and each nested `Raw*` struct.
3. The brief's `tests/common.rs` had an unused `PathBuf` import (warning because cargo
   auto-discovers it as its own test target); removed (see above).
4. A `for (a, b)` pattern typo in one test was caught at compile and fixed.
5. Verified during self-review (10 throwaway scratch tests, then folded the valuable
   ones into `tests/config_merge.rs` and deleted the scratch file): table-form roots,
   mixed-shape fatal, duplicate collections, bad dimensions, cwd layering, `--index`
   flag, size parsing, env coercion, prefix-profile-from-env-backend, display redaction.

## Residual risks / notes

- `LayerSource::Etc` corresponds to the hardcoded `/etc/ferret/config.toml` path; on
  non-unix it simply never reads. Skippable layers treat any `read` error as absent.
- The api_key permission check applies to file layers only (TOML files), not env values,
  per the brief.
- `max_size` display in `to_display_toml` is integer bytes (valid TOML, round-trips);
  the spec's illustrative `"40MiB"` string form is accepted on input.
