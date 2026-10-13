//! Audio primitives (spec §4.4): ffmpeg/ffprobe resolution and duration
//! probe. Segmentation (`extract_audio`) lands in Task 3; the index drives
//! that one file at a time like the PDF two-pass scheduler.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::error::{Error, Result};
use crate::extract::{Chunk, EmbedDecision, ExtractOutcome};
use crate::types::ChunkKind;

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

/// Segment `path` into ~`segment_secs` clips of 16kHz mono WAV, one
/// [`Chunk`] per segment (`ceil(duration / segment_secs)` rows, tails under
/// 0.5s dropped). Clip bytes are loaded into `image_bytes` — the exact
/// embedder payload, mirroring image chunks. Segment WAVs live under
/// `cache_dir/audio/{hash}-{idx}.wav` (path-hash scoped, like the PDF raster
/// dir); each clip is removed right after it is read into memory — the
/// bytes are the payload, the files are scratch, so success leaves
/// nothing behind with no post-commit sweep.
/// Undecodable input is `Error::file` (ffprobe/ffmpeg nonzero exits).
pub fn extract_audio(
    path: &Path,
    segment_secs: u64,
    cache_dir: &Path,
    ffmpeg_dir: Option<&Path>,
) -> Result<ExtractOutcome> {
    if segment_secs == 0 {
        return Err(Error::fatal(
            "extract_audio: segment_secs must be at least 1",
        ));
    }
    let duration = audio_duration(ffmpeg_dir, path)?;
    let ffmpeg = find_ffmpeg(ffmpeg_dir)?;
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let dir = cache_dir.join("audio");
    std::fs::create_dir_all(&dir)
        .map_err(|e| Error::file(format!("cannot create audio cache {}: {e}", dir.display())))?;
    // Scope clip names to the file so concurrent files never collide.
    let hash = blake3::hash(canonical.to_string_lossy().as_bytes()).to_hex();
    let tag = &hash[..16.min(hash.len())];

    let mut chunks = Vec::new();
    // Clips cut before an error: removed on failure so a failed file
    // leaves no partials (success-path retention + post-commit deletion
    // is the caller's job).
    let mut written: Vec<PathBuf> = Vec::new();
    let mut start = 0.0;
    let mut idx = 0u32;
    while start < duration {
        let len = (duration - start).min(segment_secs as f64);
        if len < 0.5 {
            break;
        }
        let stem = dir.join(format!("{tag}-{idx}"));
        // Refuse to cut through a symlinked output path (cache plant).
        if stem.with_extension("wav").is_symlink() {
            return Err(Error::file(format!(
                "refusing to cut {}: output path is a symlink",
                path.display()
            )));
        }
        let out = Command::new(&ffmpeg)
            .arg("-v")
            .arg("error")
            .arg("-y")
            .arg("-ss")
            .arg(start.to_string())
            .arg("-t")
            .arg(len.to_string())
            .arg("-i")
            .arg(&canonical)
            .arg("-ar")
            .arg("16000")
            .arg("-ac")
            .arg("1")
            .arg("-c:a")
            .arg("pcm_s16le")
            .arg(stem.with_extension("wav"))
            .stdin(Stdio::null())
            .output()
            .map_err(|e| Error::file(format!("could not run {}: {e}", ffmpeg.display())))?;
        let clip = stem.with_extension("wav");
        if !out.status.success() || !clip.is_file() {
            clean_partials(&written);
            return Err(Error::file(format!(
                "ffmpeg failed for {} offset {start}s (exited {})",
                path.display(),
                out.status.code().unwrap_or(-1)
            )));
        }
        let bytes = std::fs::read(&clip).map_err(|e| {
            clean_partials(&written);
            Error::file(format!("cannot read segment {}: {e}", clip.display()))
        })?;
        written.push(clip.clone());
        // Clips are scratch: bytes are already in memory, so remove each
        // segment right after reading it. Success leaves nothing behind
        // (no post-commit sweep needed, unlike PDF rasters which the
        // embed path reads from disk); `clean_partials` stays as the
        // error-path net for races.
        let _ = std::fs::remove_file(&clip);
        chunks.push(Chunk {
            kind: ChunkKind::AudioSegment,
            text: None,
            start_byte: None,
            end_byte: None,
            page: None,
            embed: EmbedDecision::Vec,
            t_start_ms: Some((start * 1000.0).round() as i64),
            t_end_ms: Some(((start + len) * 1000.0).round() as i64),
            image_bytes: Some(bytes),
        });
        start += segment_secs as f64;
        idx += 1;
    }
    Ok(chunks)
}

/// Best-effort removal of clips cut before a failure; errors ignored
/// (the cache is scratch space, and a failed unlink must not mask the
/// real error).
fn clean_partials(written: &[PathBuf]) {
    for p in written {
        let _ = std::fs::remove_file(p);
    }
}
