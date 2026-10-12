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

use super::{Chunk, EmbedDecision};

/// Preferred converter first: `magick`, then legacy `convert`.
const CONVERTERS: [&str; 2] = ["magick", "convert"];

/// First executable on PATH among [`CONVERTERS`], if any.
pub(crate) fn find_magick() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            CONVERTERS.iter().find_map(|bin| {
                let candidate = dir.join(bin);
                if candidate.is_file() {
                    Some(candidate)
                } else {
                    None
                }
            })
        })
    })
}

pub fn extract_image(
    path: &Path,
    image_cfg: &ImageChunkConfig,
    cache_dir: &Path,
    heic: HeicMode,
) -> Result<Vec<Chunk>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    // HEIC/HEIF pixels come from a magick conversion (the `image` crate
    // cannot decode them). The missing-binary case never reaches here:
    // the indexer skips it as `no_magick` first.
    let source: PathBuf = if matches!(ext.as_str(), "heic" | "heif") && matches!(heic, HeicMode::Magick) {
        convert_heic(path, cache_dir)?
    } else {
        path.to_path_buf()
    };

    let img = image::open(&source).map_err(|e| {
        Error::fatal(format!("cannot decode image {}: {e}", path.display()))
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
/// `{cache_dir}/heic/{stem}.png`. A failed conversion is an extract error.
fn convert_heic(path: &Path, cache_dir: &Path) -> Result<PathBuf> {
    let converter =
        find_magick().ok_or_else(|| Error::fatal("heic conversion needs magick or convert on PATH"))?;
    let out_dir = cache_dir.join("heic");
    std::fs::create_dir_all(&out_dir)?;
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_string());
    let out = out_dir.join(format!("{stem}.png"));
    let status = std::process::Command::new(&converter)
        .arg(path)
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
    Ok(out)
}
