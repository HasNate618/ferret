//! Task 14: image extraction — one PNG chunk per file, first frame of
//! animations, long-edge downscale, HEIC via magick, corrupt → error.

use std::path::Path;

use ferret::config::ImageChunkConfig;
use ferret::extract::image::extract_image;
use ferret::extract::EmbedDecision;
use ferret::types::{ChunkKind, HeicMode};
use image::GenericImageView;

fn img_cfg(max_edge_px: u32) -> ImageChunkConfig {
    ImageChunkConfig { max_edge_px }
}

fn tmpdir() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

/// Decode the extracted PNG bytes back to pixels (the `image` crate is a
/// main dependency, so tests may use it).
fn decode_png(bytes: &[u8]) -> image::DynamicImage {
    image::load_from_memory_with_format(bytes, image::ImageFormat::Png).unwrap()
}

#[test]
fn tiny_png_extracts_one_image_chunk() {
    let tmp = tmpdir();
    let chunks = extract_image(
        Path::new("testdata/one.png"),
        &img_cfg(1024),
        tmp.path(),
        HeicMode::Skip,
    )
    .unwrap();
    assert_eq!(chunks.len(), 1);
    let c = &chunks[0];
    assert_eq!(c.kind, ChunkKind::Image);
    assert_eq!(c.text, None);
    assert_eq!(c.start_byte, None);
    assert_eq!(c.end_byte, None);
    assert_eq!(c.page, None);
    assert_eq!(c.embed, EmbedDecision::Vec);
    let img = decode_png(c.image_bytes.as_ref().unwrap());
    assert_eq!(img.dimensions(), (1, 1));
}

#[test]
fn animated_gif_keeps_first_frame_only() {
    // Frame 0 is 2x1 red, frame 1 is 1x2 blue: the chunk must be 2x1.
    let tmp = tmpdir();
    let chunks = extract_image(
        Path::new("testdata/two.gif"),
        &img_cfg(1024),
        tmp.path(),
        HeicMode::Skip,
    )
    .unwrap();
    assert_eq!(chunks.len(), 1);
    let img = decode_png(chunks[0].image_bytes.as_ref().unwrap());
    assert_eq!(img.dimensions(), (2, 1));
}

#[test]
fn wide_image_downscales_long_edge() {
    // 2000x10 with max_edge 8 → 8x1 (aspect preserved, never upscaled).
    let tmp = tmpdir();
    let chunks = extract_image(
        Path::new("testdata/wide.png"),
        &img_cfg(8),
        tmp.path(),
        HeicMode::Skip,
    )
    .unwrap();
    let img = decode_png(chunks[0].image_bytes.as_ref().unwrap());
    assert_eq!(img.dimensions(), (8, 1));
}

#[test]
fn small_image_is_never_upscaled() {
    let tmp = tmpdir();
    let chunks = extract_image(
        Path::new("testdata/one.png"),
        &img_cfg(1024),
        tmp.path(),
        HeicMode::Skip,
    )
    .unwrap();
    let img = decode_png(chunks[0].image_bytes.as_ref().unwrap());
    assert_eq!(img.dimensions(), (1, 1));
}

#[test]
fn corrupt_image_is_extract_error() {
    let tmp = tmpdir();
    let bad = tmp.path().join("bad.png");
    std::fs::write(&bad, b"this is not an image").unwrap();
    let err = extract_image(&bad, &img_cfg(1024), tmp.path(), HeicMode::Skip).unwrap_err();
    assert!(err.is_file_error(), "corrupt image is a per-file failure: {err}");
}

#[test]
fn heic_magick_mode_converts_with_binary_present() {
    // Real 1x1 HEIC fixture; skip (do not fail) on hosts without magick.
    if !host_has_magick() {
        eprintln!("skip: no magick or convert on PATH");
        return;
    }
    let tmp = tmpdir();
    let cache = tmp.path().join("cache");
    let chunks = extract_image(
        Path::new("testdata/sample.heic"),
        &img_cfg(1024),
        &cache,
        HeicMode::Magick,
    )
    .unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].kind, ChunkKind::Image);
    let img = decode_png(chunks[0].image_bytes.as_ref().unwrap());
    assert_eq!(img.dimensions(), (1, 1));
}

/// Host capability gate for the conversion test (`find_magick` is
/// `pub(crate)`; mirrors its executable-bit predicate).
fn host_has_magick() -> bool {
    #[cfg(unix)]
    fn is_exec(p: &std::path::Path) -> bool {
        use std::os::unix::fs::PermissionsExt;
        p.is_file()
            && p.metadata()
                .is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    fn is_exec(p: &std::path::Path) -> bool {
        p.is_file()
    }
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths)
            .filter(|dir| !dir.as_os_str().is_empty())
            .any(|dir| {
                ["magick", "convert"].iter().any(|bin| is_exec(&dir.join(bin)))
            })
    })
}
