//! Image extraction (spec §4.3) — Task 14. Stub: every image is a per-file
//! extract error until then, so `index` records `files.error` and the walk
//! continues (a walk continues on extract errors; only embedder/DB/schema
//! failures abort).

use std::path::Path;

use crate::config::ImageChunkConfig;
use crate::error::{Error, Result};
use crate::types::HeicMode;

use super::Chunk;

pub fn extract_image(
    _path: &Path,
    _image_cfg: &ImageChunkConfig,
    _cache_dir: &Path,
    _heic: HeicMode,
) -> Result<Vec<Chunk>> {
    Err(Error::fatal("image extract not implemented"))
}
