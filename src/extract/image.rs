//! Image extraction (spec §4.3): first frame of animations, long-edge
//! downscale to `max_edge_px`, PNG-encoded into the chunk. HEIC/HEIF with
//! `heic = "magick"` converts via `magick`/`convert` into `{cache_dir}`;
//! a missing binary is a walk-skip (`no_magick`, decided in `index`), and a
//! failed conversion is an extract error.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use crate::config::ImageChunkConfig;
use crate::error::{Error, Result};
use crate::types::{ChunkKind, HeicMode};

use super::{Chunk, EmbedDecision, ExtractOutcome};

/// Preferred converter first: `magick`, then legacy `convert`.
const CONVERTERS: [&str; 2] = ["magick", "convert"];

/// First executable on PATH among [`CONVERTERS`], if any. Empty PATH
/// entries are skipped (they would resolve relative to the cwd) and
/// non-executable files do not count.
pub(crate) fn find_magick() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .filter(|dir| !dir.as_os_str().is_empty())
            .find_map(|dir| {
                CONVERTERS.iter().find_map(|bin| {
                    let candidate = dir.join(bin);
                    if is_executable(&candidate) {
                        Some(candidate)
                    } else {
                        None
                    }
                })
            })
    })
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.is_file()
        && path
            .metadata()
            .is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

pub fn extract_image(
    path: &Path,
    image_cfg: &ImageChunkConfig,
    cache_dir: &Path,
    heic: HeicMode,
) -> Result<ExtractOutcome> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    // HEIC/HEIF pixels come from a magick conversion (the `image` crate
    // cannot decode them). `skip` mode never reaches here via the walk
    // (HeicSkip first); a direct call with `skip` is a contract error.
    if matches!(ext.as_str(), "heic" | "heif") && matches!(heic, HeicMode::Skip) {
        return Err(Error::fatal(format!(
            "heic file {} reached image extract with files.heic=skip (walk must skip it first)",
            path.display()
        )));
    }
    // The missing-binary case never reaches here either: the indexer
    // skips it as `no_magick` first.
    let source: PathBuf = if matches!(ext.as_str(), "heic" | "heif") {
        convert_heic(path, cache_dir)?
    } else {
        path.to_path_buf()
    };

    let img = image::open(&source).map_err(|e| {
        Error::file(format!("cannot decode image {}: {e}", path.display()))
    })?;
    // `image::open` yields the first frame of GIF/WebP animations.
    let (w, h) = (img.width(), img.height());
    let edge = image_cfg.max_edge_px.max(1);
    let img = if w.max(h) > edge {
        img.thumbnail(edge, edge)
    } else {
        img
    };
    let mut png = Vec::new();
    img.write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| Error::fatal(format!("cannot PNG-encode {}: {e}", path.display())))?;

    Ok(vec![Chunk {
        kind: ChunkKind::Image,
        text: None,
        start_byte: None,
        end_byte: None,
        page: None,
        embed: EmbedDecision::Vec,
        image_bytes: Some(png),
    }])
}

/// `magick <heic> png:<out>` (falling back to `convert`), writing beside
/// nothing in the corpus: converted pixels live under
/// `{cache_dir}/heic/{stem}-{pathhash}.png`, so same-named files in
/// different directories never collide. Converted pixels are stable across
/// runs; stale entries (corpus file deleted) linger until the operator
/// clears the cache dir — only `{cache}/pages/` (PDF rasters) is deleted
/// after use by contract. A failed conversion is an extract error. The source path is canonicalized so a corpus file named like a
/// flag (`-define`, `--version`) cannot inject converter options.
fn convert_heic(path: &Path, cache_dir: &Path) -> Result<PathBuf> {
    let converter =
        find_magick().ok_or_else(|| Error::fatal("heic conversion needs magick or convert on PATH"))?;
    let out_dir = cache_dir.join("heic");
    std::fs::create_dir_all(&out_dir)?;
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_string());
    let path_hash = blake3::hash(path.to_string_lossy().as_bytes()).to_hex();
    let out = out_dir.join(format!("{stem}-{:.16}.png", path_hash));
    let source = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let status = std::process::Command::new(&converter)
        .arg(source)
        .arg(format!("png:{}", out.display()))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| Error::fatal(format!("could not run {}: {e}", converter.display())))?;
    if !status.success() {
        return Err(Error::fatal(format!(
            "heic conversion failed for {} ({} exited {status})",
            path.display(),
            converter.display(),
        )));
    }
    if !out.is_file() {
        return Err(Error::fatal(format!(
            "heic conversion reported success but wrote nothing for {}",
            path.display()
        )));
    }
    Ok(out)
}
