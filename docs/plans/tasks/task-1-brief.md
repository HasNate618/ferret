### Task 1: Crate scaffold, errors, path policy

**Files:**
- Create: `Cargo.toml`, `.gitignore`, `src/main.rs`, `src/lib.rs`, `src/error.rs`, `src/paths.rs`, `tests/paths.rs`
- Create: `docs/superpowers/plans/2026-10-12-ferret-v1.md` already exists (this file)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ferret::error::{Error, Result}` with `Error::usage(String)`, `Error::fatal(String)`, `fn exit_code(&self) -> i32` (`2` for usage, `1` otherwise)
  - `ferret::paths::{FerretDirs, ferret_dirs}` — `ferret_dirs(env: &dyn Fn(&str) -> Option<String>) -> FerretDirs` where `FerretDirs { config_dir, data_dir, cache_dir: PathBuf }`
  - `pub const INDEX_FILENAME: &str = "index.sqlite";`

- [ ] **Step 1: Write the failing path test**

Create `tests/paths.rs`:

```rust
use ferret::paths::ferret_dirs;
use std::path::PathBuf;

fn env_map<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |k| {
        pairs
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    }
}

#[test]
fn xdg_data_home_wins_on_any_os() {
    let dirs = ferret_dirs(&env_map(&[("XDG_DATA_HOME", "/tmp/xdg-data")]));
    assert_eq!(dirs.data_dir, PathBuf::from("/tmp/xdg-data/ferret"));
}

#[test]
fn xdg_config_and_cache_win() {
    let dirs = ferret_dirs(&env_map(&[
        ("XDG_CONFIG_HOME", "/tmp/xdg-cfg"),
        ("XDG_CACHE_HOME", "/tmp/xdg-cache"),
    ]));
    assert_eq!(dirs.config_dir, PathBuf::from("/tmp/xdg-cfg/ferret"));
    assert_eq!(dirs.cache_dir, PathBuf::from("/tmp/xdg-cache/ferret"));
}

#[test]
fn missing_xdg_uses_platform_defaults() {
    let dirs = ferret_dirs(&|_| None);
    let cfg = dirs.config_dir.to_string_lossy();
    let data = dirs.data_dir.to_string_lossy();
    let cache = dirs.cache_dir.to_string_lossy();
    assert!(cfg.ends_with("ferret"), "{cfg}");
    assert!(data.ends_with("ferret"), "{data}");
    assert!(cache.ends_with("ferret"), "{cache}");
    #[cfg(target_os = "macos")]
    {
        assert!(cfg.contains("Library/Preferences"), "{cfg}");
        assert!(data.contains("Library/Application Support"), "{data}");
        assert!(cache.contains("Library/Caches"), "{cache}");
    }
    #[cfg(target_os = "linux")]
    {
        assert!(cfg.contains(".config/ferret"), "{cfg}");
        assert!(data.contains(".local/share/ferret"), "{data}");
        assert!(cache.contains(".cache/ferret"), "{cache}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nate/Projects/ferret && cargo test --test paths`
Expected: FAIL (no `Cargo.toml` / unresolved crate)

- [ ] **Step 3: Write crate + implementation**

`.gitignore`:

```
/target
/result
.direnv
```

`Cargo.toml` (pin caret ranges; do not add extract/embed crates yet):

```toml
[package]
name = "ferret"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
description = "Local-first semantic multimodal file search"
license = "MIT"
default-run = "ferret"

[dependencies]
thiserror = "2"
dirs = "6"

[dev-dependencies]
```

If `dirs` 7 is what cargo resolves, that is fine. Use `dirs = "6"` or `"7"` — both expose `config_dir`/`data_dir`/`cache_dir`. **Do not** use `dirs::config_dir()` as the macOS config path; spec wants Preferences.

`src/error.rs`:

```rust
use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Usage(String),
    Fatal(String),
}

impl Error {
    pub fn usage(msg: impl Into<String>) -> Self {
        Self::Usage(msg.into())
    }
    pub fn fatal(msg: impl Into<String>) -> Self {
        Self::Fatal(msg.into())
    }
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            Self::Fatal(_) => 1,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(s) | Self::Fatal(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Fatal(e.to_string())
    }
}
```

`src/paths.rs`:

```rust
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FerretDirs {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

pub const INDEX_FILENAME: &str = "index.sqlite";

pub fn ferret_dirs(env: &dyn Fn(&str) -> Option<String>) -> FerretDirs {
    let config_dir = xdg_or(env, "XDG_CONFIG_HOME", platform_config_parent()).join("ferret");
    let data_dir = xdg_or(env, "XDG_DATA_HOME", platform_data_parent()).join("ferret");
    let cache_dir = xdg_or(env, "XDG_CACHE_HOME", platform_cache_parent()).join("ferret");
    FerretDirs {
        config_dir,
        data_dir,
        cache_dir,
    }
}

fn xdg_or(env: &dyn Fn(&str) -> Option<String>, key: &str, fallback: PathBuf) -> PathBuf {
    match env(key) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => fallback,
    }
}

fn platform_config_parent() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        dirs::home_dir()
            .expect("home directory")
            .join("Library/Preferences")
    }
    #[cfg(not(target_os = "macos"))]
    {
        dirs::config_dir().expect("config directory")
    }
}

fn platform_data_parent() -> PathBuf {
    dirs::data_dir().expect("data directory")
}

fn platform_cache_parent() -> PathBuf {
    dirs::cache_dir().expect("cache directory")
}
```

`src/lib.rs`:

```rust
pub mod error;
pub mod paths;
```

`src/main.rs`:

```rust
fn main() {
    eprintln!("ferret: not implemented");
    std::process::exit(1);
}
```

Then `git init` in `/home/nate/Projects/ferret` if `.git` is absent.

- [ ] **Step 4: Run tests**

Run: `cargo test --test paths`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml .gitignore src tests/paths.rs
git commit -m "$(cat <<'EOF'
feat: scaffold ferret crate with XDG-aware path policy

EOF
)"
```

---

