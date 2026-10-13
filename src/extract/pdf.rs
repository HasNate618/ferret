//! PDF page primitives (spec §4.4): poppler binary resolution, page
//! count, page text, page rendering. The per-page transaction loop and the
//! `next_page` resume cursor live in `crate::index`, which drives these
//! one page at a time so an abort never loses committed pages.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::config::ExtractConfig;
use crate::error::{Error, Result};

/// Resolve one poppler binary: the explicit `extract.<name>` setting wins,
/// else `{poppler_dir}/<name>` when `poppler_dir` is set, else a PATH
/// lookup. Anything missing is `<name> not found` as `Error::fatal` — the
/// indexer reclassifies it into the per-file extract-error path (a missing
/// binary must not abort the walk).
pub fn resolve_tool(cfg: &ExtractConfig, name: &str) -> Result<PathBuf> {
    let explicit = match name {
        "pdfinfo" => &cfg.pdfinfo,
        "pdftotext" => &cfg.pdftotext,
        "pdftoppm" => &cfg.pdftoppm,
        _ => return Err(Error::fatal(format!("unknown poppler tool `{name}`"))),
    };
    if !explicit.is_empty() {
        let p = PathBuf::from(explicit);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(Error::fatal(format!(
                "{name} not found (extract.{name} = `{explicit}`)"
            )))
        };
    }
    if !cfg.poppler_dir.is_empty() {
        let p = PathBuf::from(&cfg.poppler_dir).join(name);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(Error::fatal(format!(
                "{name} not found (extract.poppler_dir = `{}`)",
                cfg.poppler_dir
            )))
        };
    }
    std::env::var_os("PATH")
        .and_then(|paths| {
            std::env::split_paths(&paths)
                .filter(|dir| !dir.as_os_str().is_empty())
                .map(|dir| dir.join(name))
                .find(|p| p.is_file())
        })
        .ok_or_else(|| Error::fatal(format!("{name} not found (no poppler on PATH)")))
}

/// `pdfinfo` page count (`Pages:` line). Poppler warnings on stderr are
/// ignored; a nonzero exit or unparsable count is an extract error.
pub fn page_count(path: &Path, pdfinfo: &Path) -> Result<u32> {
    let out = Command::new(pdfinfo)
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| Error::fatal(format!("could not run {}: {e}", pdfinfo.display())))?;
    if !out.status.success() {
        return Err(Error::fatal(format!(
            "pdfinfo failed for {} (exited {})",
            path.display(),
            out.status
        )));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            if key.trim() == "Pages" {
                value.trim().parse::<u32>().ok()
            } else {
                None
            }
        })
        .ok_or_else(|| {
            Error::fatal(format!("pdfinfo gave no page count for {}", path.display()))
        })
}

/// `pdftotext -f N -l N -layout <path> -`: page text (possibly empty for
/// scanned pages — emptiness is data, not an error).
pub fn page_text(path: &Path, page: u32, pdftotext: &Path) -> Result<String> {
    let range = page.to_string();
    let out = Command::new(pdftotext)
        .args(["-f", &range, "-l", &range, "-layout"])
        .arg(path)
        .arg("-")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| Error::fatal(format!("could not run {}: {e}", pdftotext.display())))?;
    if !out.status.success() {
        return Err(Error::fatal(format!(
            "pdftotext failed for {} page {page} (exited {})",
            path.display(),
            out.status
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `pdftoppm -f N -l N -png -r <dpi> -singlefile <path> <cache>/pages/<id>-<page>`:
/// renders exactly one PNG and returns its path. The caller deletes the
/// raster after the page transaction commits.
pub fn render_page(
    path: &Path,
    page: u32,
    file_id: i64,
    render_dpi: u32,
    pdftoppm: &Path,
    cache_dir: &Path,
) -> Result<PathBuf> {
    let pages_dir = cache_dir.join("pages");
    std::fs::create_dir_all(&pages_dir)?;
    let stem = pages_dir.join(format!("{file_id}-{page}"));
    let range = page.to_string();
    let out = Command::new(pdftoppm)
        .args([
            "-f",
            &range,
            "-l",
            &range,
            "-png",
            "-r",
            &render_dpi.to_string(),
            "-singlefile",
        ])
        .arg(path)
        .arg(&stem)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| Error::fatal(format!("could not run {}: {e}", pdftoppm.display())))?;
    let png = stem.with_extension("png");
    if !out.status.success() || !png.is_file() {
        return Err(Error::fatal(format!(
            "pdftoppm failed for {} page {page} (exited {})",
            path.display(),
            out.status
        )));
    }
    Ok(png)
}
