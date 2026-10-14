//! Video primitives (spec §4.4): cut ~`segment_secs` H.264 clips for the
//! native video encoder. Resolution (`find_ffmpeg`) and duration
//! (`audio_duration`) are reused from [`super::audio`] — ffprobe
//! `format=duration` is container-agnostic, so no video-specific resolve
//! or duration probe exists.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::error::{Error, Result};
use crate::extract::audio::{audio_duration, find_ffmpeg};
use crate::extract::{Chunk, EmbedDecision, ExtractOutcome};
use crate::types::ChunkKind;

/// Segment `path` into ~`segment_secs` mp4 clips, one [`Chunk`] per segment
/// (`ceil(duration / segment_secs)` rows, tails under 0.5s dropped). Clip
/// bytes are loaded into `image_bytes` — the exact embedder payload,
/// mirroring audio segments (the field name is historical). Segment clips
/// live under `cache_dir/video/{hash}-{idx}.mp4` (path-hash scoped, like
/// audio); each clip is removed right after it is read into memory — the
/// bytes are the payload, the files are scratch, so success leaves
/// nothing behind with no post-commit sweep.
/// Clips carry the muxed audio stream (the server's video encoder sees
/// both); ferret creates no separate audio chunks for video files.
/// Undecodable input is `Error::file` (ffprobe/ffmpeg nonzero exits).
pub fn extract_video(
    path: &Path,
    segment_secs: u64,
    cache_dir: &Path,
    ffmpeg_dir: Option<&Path>,
) -> Result<ExtractOutcome> {
    if segment_secs == 0 {
        return Err(Error::fatal(
            "extract_video: segment_secs must be at least 1",
        ));
    }
    let duration = audio_duration(ffmpeg_dir, path)?;
    let ffmpeg = find_ffmpeg(ffmpeg_dir)?;
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let dir = cache_dir.join("video");
    std::fs::create_dir_all(&dir)
        .map_err(|e| Error::file(format!("cannot create video cache {}: {e}", dir.display())))?;
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
        if stem.with_extension("mp4").is_symlink() {
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
            .arg("-map")
            .arg("0:v:0")
            .arg("-map")
            .arg("0:a?")
            .arg("-c:v")
            .arg("libx264")
            .arg("-preset")
            .arg("veryfast")
            .arg("-crf")
            .arg("28")
            .arg("-vf")
            .arg("scale=640:-2")
            .arg("-c:a")
            .arg("aac")
            .arg("-b:a")
            .arg("64k")
            .arg(stem.with_extension("mp4"))
            .stdin(Stdio::null())
            .output()
            .map_err(|e| Error::file(format!("could not run {}: {e}", ffmpeg.display())))?;
        let clip = stem.with_extension("mp4");
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
        // segment right after reading it. Success leaves nothing behind;
        // `clean_partials` stays as the error-path net for races.
        let _ = std::fs::remove_file(&clip);
        chunks.push(Chunk {
            kind: ChunkKind::VideoSegment,
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
