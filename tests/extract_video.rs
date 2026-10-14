//! Video milestone Task 2: clip extraction into timestamped mp4 segments.

use std::path::Path;

use ferret::extract::audio::find_ffmpeg;
use ferret::extract::video::extract_video;
use ferret::types::ChunkKind;

fn tmpdir() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

#[test]
fn clip_mp4_yields_three_timestamped_chunks() {
    if find_ffmpeg(None).is_err() {
        eprintln!("skip: no ffmpeg");
        return;
    }
    let tmp = tmpdir();
    let chunks = extract_video(
        Path::new("testdata/clip.mp4"),
        10,
        &tmp.path().join("cache"),
        None,
    )
    .unwrap();
    assert_eq!(chunks.len(), 3);
    assert!(
        chunks
            .iter()
            .all(|c| c.kind == ChunkKind::VideoSegment && c.text.is_none())
    );
    assert_eq!(
        chunks
            .iter()
            .map(|c| (c.t_start_ms, c.t_end_ms))
            .collect::<Vec<_>>(),
        vec![
            (Some(0), Some(10000)),
            (Some(10000), Some(20000)),
            (Some(20000), Some(25000))
        ]
    );
    assert!(
        chunks
            .iter()
            .all(|c| c.image_bytes.as_ref().is_some_and(|b| b.len() > 1000))
    );
}

#[test]
fn corrupt_mp4_is_file_error() {
    let tmp = tmpdir();
    let bad = tmp.path().join("bad.mp4");
    std::fs::write(&bad, b"\x00\x00\x00ftyp-not-really-mp4").unwrap();
    let err = extract_video(&bad, 10, &tmp.path().join("cache"), None).unwrap_err();
    assert!(err.is_file_error());
}
