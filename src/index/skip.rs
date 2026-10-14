//! Walk-skip taxonomy (spec §4.5 / §10 NDJSON reasons).

/// Why a path was skipped during the walk. The strings match the spec's
/// NDJSON `reason` values exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// No entry in `files.extension_kind` for the extension (spec §4.5:
    /// unknown extensions, docx/odt/pptx, audio, video, archives).
    UnsupportedExt,
    /// A skip rule matched: reserved component, hidden component, sensitive
    /// basename glob, oversize file, or user `exclude_globs`.
    SkipRule,
    /// The embedder cannot embed images (Task 11 capability check).
    NoImageBackend,
    /// `.heic` / `.heif` with `files.heic = "skip"` (the v1 default).
    HeicSkip,
    /// `files.heic = "magick"` but no `magick` / `convert` on PATH (Task 14).
    NoMagick,
    /// Audio file but no `ffmpeg` / `ffprobe` resolvable (audio milestone).
    NoFfmpeg,
    /// A gitignore rule matched inside a git repo.
    Gitignore,
    /// Audio file no backend can embed (capability probe said no audio).
    NoAudioBackend,
    /// Video file no backend can embed (capability probe said no video).
    NoVideoBackend,
    /// A non-empty `include_globs` and the relative path matched none.
    IncludeGlobs,
}

impl SkipReason {
    /// The spec NDJSON reason string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UnsupportedExt => "unsupported_ext",
            Self::SkipRule => "skip_rule",
            Self::NoImageBackend => "no_image_backend",
            Self::HeicSkip => "heic_skip",
            Self::NoMagick => "no_magick",
            Self::NoFfmpeg => "no_ffmpeg",
            Self::Gitignore => "gitignore",
            Self::NoAudioBackend => "no_audio_backend",
            Self::NoVideoBackend => "no_video_backend",
            Self::IncludeGlobs => "include_globs",
        }
    }
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
