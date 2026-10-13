//! Audio primitives (spec §4.4): ffmpeg/ffprobe resolution and duration
//! probe. Segmentation (`extract_audio`) lands in Task 3; the index drives
//! that one file at a time like the PDF two-pass scheduler.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::error::{Error, Result};

/// Resolve `ffmpeg`: `{ffmpeg_dir}/ffmpeg` when a dir is given, else a PATH
/// lookup. Anything missing is `ffmpeg not found` as `Error::file` (a
/// missing binary must not abort the walk). Candidates must be executable
/// files (mirrors `resolve_tool` / `find_magick`), so a non-executable
/// decoy on PATH is skipped rather than failing at spawn.
pub fn find_ffmpeg(ffmpeg_dir: Option<&Path>) -> Result<PathBuf> {
    resolve("ffmpeg", ffmpeg_dir)
}

/// Resolve `ffprobe` with the same dir-then-PATH priority as [`find_ffmpeg`].
pub fn find_ffprobe(ffmpeg_dir: Option<&Path>) -> Result<PathBuf> {
    resolve("ffprobe", ffmpeg_dir)
}

fn resolve(name: &str, ffmpeg_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = ffmpeg_dir {
        let p = dir.join(name);
        return if is_executable_file(&p) {
            Ok(p)
        } else {
            Err(Error::file(format!(
                "{name} not found (extract.ffmpeg_dir = `{}`)",
                dir.display()
            )))
        };
    }
    std::env::var_os("PATH")
        .and_then(|paths| {
            std::env::split_paths(&paths)
                .filter(|dir| !dir.as_os_str().is_empty())
                .map(|dir| dir.join(name))
                .find(|p| is_executable_file(p))
        })
        .ok_or_else(|| Error::file(format!("{name} not found (no ffmpeg on PATH)")))
}

/// `ffprobe` duration in seconds (`format=duration`, bare number).
/// A nonzero exit, missing output, or unparseable number is an extract
/// error (`Error::file`): the walk continues past undecodable files.
/// The corpus path is canonicalized so a file named like a flag cannot
/// inject ffprobe options (parity with the poppler path).
pub fn audio_duration(ffmpeg_dir: Option<&Path>, path: &Path) -> Result<f64> {
    let ffprobe = find_ffprobe(ffmpeg_dir)?;
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let out = Command::new(&ffprobe)
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("csv=p=0")
        .arg("--")
        .arg(&canonical)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| Error::file(format!("ffprobe failed on {}: {e}", path.display())))?;
    if !out.status.success() {
        return Err(Error::file(format!(
            "ffprobe failed on {} (exit {})",
            path.display(),
            out.status.code().unwrap_or(-1)
        )));
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    match text.parse::<f64>() {
        Ok(d) if d.is_finite() && d > 0.0 => Ok(d),
        _ => Err(Error::file(format!(
            "ffprobe reported no duration for {}",
            path.display()
        ))),
    }
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.is_file()
        && path
            .metadata()
            .is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}
