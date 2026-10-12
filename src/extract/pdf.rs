//! PDF page extraction (spec §4.4) — Task 15. Stub: every PDF is a per-file
//! extract error until then; Task 15 replaces the body with the per-page
//! transaction loop and the `next_page` resume cursor.

use std::path::Path;

use crate::config::{ImageChunkConfig, PdfChunkConfig, TextChunkConfig};
use crate::error::{Error, Result};

use super::Chunk;

pub fn extract_pdf(
    _path: &Path,
    _text_cfg: &TextChunkConfig,
    _pdf_cfg: &PdfChunkConfig,
    _image_cfg: &ImageChunkConfig,
    _cache_dir: &Path,
) -> Result<Vec<Chunk>> {
    Err(Error::fatal("pdf extract not implemented"))
}
