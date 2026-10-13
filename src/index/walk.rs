//! Corpus walk (spec §4.5).
//!
//! Enumerates each configured root (plus extra paths) and classifies every
//! path as a `File` event or a walk-skip. Order after extension typing:
//! skip-rule list (reserved/hidden components, sensitive basenames,
//! oversize, user exclude globs) → include_globs → gitignore.
//!
//! Gitignore is implemented with `ignore::gitignore` driven by this
//! walker (not `WalkBuilder`'s own filtering) so ignored paths surface as
//! `Skip { reason: Gitignore }` events instead of vanishing, and it only
//! applies inside a git repo — a `.git` at or above the file, at or below
//! the configured root. Directories are never filtered by include_globs,
//! only files are.

use std::path::{Path, PathBuf};

use globset::{GlobSet, GlobSetBuilder};
use ignore::gitignore::{Gitignore, GitignoreBuilder};

use crate::config::{FilesConfig, RootEntry};
use crate::error::{Error, Result};
use crate::types::FileKind;

pub use super::skip::SkipReason;

/// Reserved path components never walked, even under a hidden root.
const SKIP_COMPONENTS: &[&str] = &[
    ".git",
    "node_modules",
    ".cache",
    "target",
    "dist",
    "__pycache__",
    ".venv",
    "venv",
];

/// Sensitive basename patterns: key material and credential files.
const SENSITIVE_GLOBS: &[&str] = &[
    "*.pem",
    "*.key",
    "*.p8",
    "*.p12",
    "*.pfx",
    ".netrc",
    "*.netrc",
    ".env",
    ".env.*",
    "credentials.json",
    "*.sqlite-journal",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
];

/// One path seen during the walk: either an indexable file or a skip.
#[derive(Debug, Clone, PartialEq)]
pub enum WalkEvent {
    File {
        path: PathBuf,
        collection: String,
        kind: FileKind,
        ext: String,
    },
    Skip {
        path: PathBuf,
        reason: SkipReason,
    },
}

/// Lowercase extension without the dot, or `""` when none.
fn ext_of(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

/// File kind for a lowercase extension (no dot), or `None` when the
/// extension is unsupported in this index.
pub fn kind_for_ext(ext: &str, map: &std::collections::BTreeMap<String, FileKind>) -> Option<FileKind> {
    map.get(ext).copied()
}

/// Walk every configured root plus the extra paths, returning events in a
/// deterministic (name-sorted) order.
pub fn walk(
    roots: &[RootEntry],
    extra: &[PathBuf],
    files_cfg: &FilesConfig,
    follow_symlinks: bool,
) -> Result<Vec<WalkEvent>> {
    let exclude = build_globset(&files_cfg.exclude_globs)?;
    let include = if files_cfg.include_globs.is_empty() {
        None
    } else {
        Some(build_globset(&files_cfg.include_globs)?)
    };
    let sensitive = build_globset(SENSITIVE_GLOBS)?;

    let mut w = Walker {
        files_cfg,
        exclude,
        include,
        sensitive,
        follow_symlinks,
        events: Vec::new(),
    };

    for root in roots {
        w.walk_dir(&root.path, &root.path, &root.collection, Vec::new(), false)?;
    }
    for path in extra {
        // An extra file/dir under a configured root inherits that root's
        // collection; otherwise it is its own collection (last component).
        let collection = roots
            .iter()
            .find(|r| path.starts_with(&r.path))
            .map(|r| r.collection.clone())
            .unwrap_or_else(|| {
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
        if path.is_dir() {
            w.walk_dir(path, path, &collection, Vec::new(), false)?;
        } else {
            w.emit_file(path, path, &collection)?;
        }
    }
    Ok(w.events)
}

fn build_globset<I, S>(patterns: I) -> Result<GlobSet>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        let p = p.as_ref();
        let glob = globset::Glob::new(p).map_err(|e| Error::usage(format!("bad glob `{p}`: {e}")))?;
        builder.add(glob);
    }
    builder
        .build()
        .map_err(|e| Error::usage(format!("glob set: {e}")))
}

struct Walker<'a> {
    files_cfg: &'a FilesConfig,
    exclude: GlobSet,
    include: Option<GlobSet>,
    sensitive: GlobSet,
    follow_symlinks: bool,
    events: Vec<WalkEvent>,
}

impl Walker<'_> {
    /// Walk one directory. `ignores` is the stack of gitignore matchers
    /// from ancestors (innermost last); `in_repo` says a `.git` exists at
    /// or above this directory.
    fn walk_dir(
        &mut self,
        dir: &Path,
        root: &Path,
        collection: &str,
        parent_ignores: Vec<Gitignore>,
        parent_in_repo: bool,
    ) -> Result<()> {
        let mut ignores = parent_ignores;
        let in_repo = parent_in_repo || dir.join(".git").exists();
        if self.files_cfg.respect_gitignore && in_repo {
            if let Some(gi) = load_gitignore(dir) {
                ignores.push(gi);
            }
        }

        let mut entries: Vec<_> = std::fs::read_dir(dir)?
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            let file_type = entry.file_type()?;
            let is_symlink = file_type.is_symlink();
            if is_symlink && !self.follow_symlinks {
                self.skip(&path, SkipReason::SkipRule);
                continue;
            }
            let is_dir = if self.follow_symlinks {
                entry.metadata().map(|m| m.is_dir()).unwrap_or(false)
            } else {
                file_type.is_dir()
            };
            let rel = path.strip_prefix(root).unwrap_or(&path);

            if self.skip_components(rel) {
                self.skip(&path, SkipReason::SkipRule);
                continue;
            }
            if !is_dir
                && self.is_sensitive(path.file_name().and_then(|n| n.to_str()).unwrap_or(""))
            {
                self.skip(&path, SkipReason::SkipRule);
                continue;
            }
            if self.exclude.is_match(rel) {
                self.skip(&path, SkipReason::SkipRule);
                continue;
            }
            if self.ignored(&ignores, &path, is_dir) {
                self.skip(&path, SkipReason::Gitignore);
                continue;
            }

            if is_dir {
                self.walk_dir(&path, root, collection, ignores.clone(), in_repo)?;
                continue;
            }
            self.emit_file(&path, root, collection)?;
        }
        Ok(())
    }

    /// Extension typing, heic gate, then the file skip-rule list,
    /// include_globs, and gitignore, in the spec's order.
    fn emit_file(&mut self, path: &Path, root: &Path, collection: &str) -> Result<()> {
        // 0. HEIC/HEIF gate first (spec §4.3): `skip` (default) is a
        // HeicSkip. `magick` types unmapped files as images below even
        // though the default extension map omits heic/heif — otherwise
        // the kind lookup would report unsupported_ext and the magick
        // path would be unreachable. An explicit user mapping still wins.
        if matches!(ext_of(path).as_str(), "heic" | "heif")
            && matches!(self.files_cfg.heic, crate::types::HeicMode::Skip)
        {
            self.skip(path, SkipReason::HeicSkip);
            return Ok(());
        }
        // 1. Extension typing (unknown extensions are walk-skips).
        let ext = ext_of(path);
        let kind = match kind_for_ext(&ext, &self.files_cfg.extension_kind) {
            Some(k) => k,
            None if matches!(ext.as_str(), "heic" | "heif") => FileKind::Image,
            None => {
                self.skip(path, SkipReason::UnsupportedExt);
                return Ok(());
            }
        };
        // NoMagick gate (spec §4.3): heic/heif that would need conversion
        // with no converter on PATH. An explicit non-image mapping
        // bypasses it — the file is read as that kind, never converted.
        if matches!(ext.as_str(), "heic" | "heif")
            && matches!(self.files_cfg.heic, crate::types::HeicMode::Magick)
            && kind == FileKind::Image
            && crate::extract::image::find_magick().is_none()
        {
            self.skip(path, SkipReason::NoMagick);
            return Ok(());
        }

        let rel = path.strip_prefix(root).unwrap_or(path);

        // 2. Skip-rule list.
        if self.skip_components(rel) {
            self.skip(path, SkipReason::SkipRule);
            return Ok(());
        }
        if self.is_sensitive(path.file_name().and_then(|n| n.to_str()).unwrap_or("")) {
            self.skip(path, SkipReason::SkipRule);
            return Ok(());
        }
        if self.oversize(path) {
            self.skip(path, SkipReason::SkipRule);
            return Ok(());
        }
        if self.exclude.is_match(rel) {
            self.skip(path, SkipReason::SkipRule);
            return Ok(());
        }

        // 3. include_globs (relative path from the collection root).
        if let Some(include) = &self.include {
            if !include.is_match(rel) {
                self.skip(path, SkipReason::IncludeGlobs);
                return Ok(());
            }
        }

        // 4. gitignore is checked in walk_dir for directories; for files it
        // was already checked before the ext-typing short-circuit above, so
        // nothing remains.

        self.events.push(WalkEvent::File {
            path: path.to_path_buf(),
            collection: collection.to_string(),
            kind,
            ext,
        });
        Ok(())
    }

    /// Reserved components, and hidden components unless `include_hidden`.
    /// The root itself never appears in `rel`, so a hidden configured root
    /// (e.g. `~/.ssh`) is exempt.
    fn skip_components(&self, rel: &Path) -> bool {
        for comp in rel.components() {
            let name = comp.as_os_str().to_string_lossy();
            if SKIP_COMPONENTS.iter().any(|c| *c == name) {
                return true;
            }
            if name.starts_with('.') && !self.files_cfg.include_hidden {
                return true;
            }
        }
        false
    }

    fn is_sensitive(&self, basename: &str) -> bool {
        // `id_*` key files, except the public halves (`id_*.pub`).
        (basename.starts_with("id_") && !basename.ends_with(".pub")) || self.sensitive.is_match(basename)
    }

    fn oversize(&self, path: &Path) -> bool {
        match std::fs::metadata(path) {
            Ok(md) => md.len() > self.files_cfg.max_size,
            Err(_) => false,
        }
    }

    fn ignored(&self, ignores: &[Gitignore], path: &Path, is_dir: bool) -> bool {
        ignores
            .iter()
            .any(|g| g.matched(path, is_dir).is_ignore())
    }

    fn skip(&mut self, path: &Path, reason: SkipReason) {
        self.events.push(WalkEvent::Skip {
            path: path.to_path_buf(),
            reason,
        });
    }
}

/// Load `dir/.gitignore` as a matcher rooted at `dir` (patterns are
/// relative to the containing directory, per git semantics).
fn load_gitignore(dir: &Path) -> Option<Gitignore> {
    let file = dir.join(".gitignore");
    if !file.is_file() {
        return None;
    }
    let mut builder = GitignoreBuilder::new(dir);
    builder.add(file);
    builder.build().ok()
}
