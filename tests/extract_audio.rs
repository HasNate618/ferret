use common::EnvGuard;
use ferret::extract::audio::{audio_duration, find_ffmpeg};
use serial_test::serial;
use std::path::Path;

mod common;

fn write_stub(dir: &Path, name: &str, body: &str) {
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
    }
    // ETXTBSY guard: fsync file + dir before any exec.
    std::fs::File::open(&path).unwrap().sync_all().unwrap();
    std::fs::File::open(dir).unwrap().sync_all().unwrap();
}

fn tmpdir() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

#[test]
#[serial]
fn ffmpeg_dir_is_preferred_over_path() {
    let tmp = tmpdir();
    let tools = tmp.path().join("tools");
    let decoy = tmp.path().join("decoy");
    std::fs::create_dir_all(&tools).unwrap();
    std::fs::create_dir_all(&decoy).unwrap();
    write_stub(&tools, "ffmpeg", "exit 0");
    write_stub(&decoy, "ffmpeg", "exit 0");
    let _path = EnvGuard::set("PATH", decoy.to_str().unwrap());
    let found = find_ffmpeg(Some(tools.as_path())).unwrap();
    assert_eq!(found, tools.join("ffmpeg"));
}

#[test]
#[serial]
fn path_lookup_finds_ffmpeg() {
    let tmp = tmpdir();
    let tools = tmp.path().join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    write_stub(&tools, "ffmpeg", "exit 0");
    let _path = EnvGuard::set("PATH", tools.to_str().unwrap());
    assert_eq!(find_ffmpeg(None).unwrap(), tools.join("ffmpeg"));
}

#[test]
#[serial]
fn non_executable_decoy_on_path_is_skipped() {
    let tmp = tmpdir();
    let first = tmp.path().join("first");
    let second = tmp.path().join("second");
    std::fs::create_dir_all(&first).unwrap();
    std::fs::create_dir_all(&second).unwrap();
    std::fs::write(first.join("ffmpeg"), "not executable").unwrap();
    write_stub(&second, "ffmpeg", "exit 0");
    let path = format!(
        "{}:{}",
        first.to_str().unwrap(),
        second.to_str().unwrap()
    );
    let _guard = EnvGuard::set("PATH", &path);
    assert_eq!(find_ffmpeg(None).unwrap(), second.join("ffmpeg"));
}

#[test]
#[serial]
fn missing_ffmpeg_is_file_error() {
    let tmp = tmpdir();
    let empty = tmp.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let _path = EnvGuard::set("PATH", empty.to_str().unwrap());
    let err = find_ffmpeg(None).unwrap_err();
    assert!(err.is_file_error());
    assert!(err.to_string().contains("not found"), "unexpected: {err}");
}

#[test]
#[serial]
fn audio_duration_reads_ffprobe_output() {
    let tmp = tmpdir();
    let tools = tmp.path().join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    write_stub(&tools, "ffprobe", "printf '93.5'");
    let clip = tmp.path().join("clip.wav");
    std::fs::write(&clip, b"RIFF").unwrap();
    assert_eq!(audio_duration(Some(tools.as_path()), &clip).unwrap(), 93.5);
}

#[test]
#[serial]
fn audio_duration_failure_is_file_error() {
    let tmp = tmpdir();
    let tools = tmp.path().join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    write_stub(&tools, "ffprobe", "exit 1");
    let clip = tmp.path().join("clip.wav");
    std::fs::write(&clip, b"RIFF").unwrap();
    let err = audio_duration(Some(tools.as_path()), &clip).unwrap_err();
    assert!(err.is_file_error());
}

#[test]
#[serial]
fn audio_duration_garbage_is_file_error() {
    let tmp = tmpdir();
    let tools = tmp.path().join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    write_stub(&tools, "ffprobe", "printf 'N/A'");
    let clip = tmp.path().join("clip.wav");
    std::fs::write(&clip, b"RIFF").unwrap();
    assert!(
        audio_duration(Some(tools.as_path()), &clip)
            .unwrap_err()
            .is_file_error()
    );
}

#[test]
#[serial]
fn explicit_dir_without_ffmpeg_is_file_error() {
    let tmp = tmpdir();
    let tools = tmp.path().join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    let err = find_ffmpeg(Some(tools.as_path())).unwrap_err();
    assert!(err.is_file_error());
    assert!(err.to_string().contains("not found"), "unexpected: {err}");
}

#[test]
#[serial]
fn audio_duration_zero_and_nan_are_file_errors() {
    for output in ["printf '0'", "printf 'nan'", "printf 'inf'"] {
        let tmp = tmpdir();
        let tools = tmp.path().join("tools");
        std::fs::create_dir_all(&tools).unwrap();
        write_stub(&tools, "ffprobe", output);
        let clip = tmp.path().join("clip.wav");
        std::fs::write(&clip, b"RIFF").unwrap();
        assert!(
            audio_duration(Some(tools.as_path()), &clip).unwrap_err().is_file_error(),
            "output {output:?} should be a file error"
        );
    }
}

#[test]
fn sweep_wav_yields_three_timestamped_chunks() {
    use ferret::extract::audio::{extract_audio, find_ffmpeg};
    use ferret::types::ChunkKind;
    if find_ffmpeg(None).is_err() {
        eprintln!("skip: no ffmpeg");
        return;
    }
    let tmp = tmpdir();
    let chunks = extract_audio(
        Path::new("testdata/sweep.wav"),
        30,
        &tmp.path().join("cache"),
        None,
    )
    .unwrap();
    assert_eq!(chunks.len(), 3);
    assert!(
        chunks
            .iter()
            .all(|c| c.kind == ChunkKind::AudioSegment && c.text.is_none())
    );
    assert_eq!(
        chunks
            .iter()
            .map(|c| (c.t_start_ms, c.t_end_ms))
            .collect::<Vec<_>>(),
        vec![
            (Some(0), Some(30000)),
            (Some(30000), Some(60000)),
            (Some(60000), Some(65000))
        ]
    );
    assert!(
        chunks
            .iter()
            .all(|c| c.image_bytes.as_ref().is_some_and(|b| b.len() > 44))
    );
}

#[test]
fn corrupt_wav_is_file_error() {
    use ferret::extract::audio::extract_audio;
    let tmp = tmpdir();
    let bad = tmp.path().join("bad.wav");
    std::fs::write(&bad, b"RIFF-not-really-a-wav").unwrap();
    let err = extract_audio(&bad, 30, &tmp.path().join("cache"), None).unwrap_err();
    assert!(err.is_file_error());
}

#[test]
#[serial]
fn failed_segment_removes_earlier_clips() {
    use ferret::extract::audio::extract_audio;
    let tmp = tmpdir();
    let tools = tmp.path().join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    // 65s duration so three segments are attempted.
    write_stub(&tools, "ffprobe", "printf '65'");
    // First ffmpeg call writes a fake clip and succeeds; later calls fail.
    // The output path is the last argv; the call count rides an env var.
    write_stub(
        &tools,
        "ffmpeg",
        "n=$(cat \"$AUDIO_STUB_COUNT\" 2>/dev/null || echo 0); echo $((n + 1)) > \"$AUDIO_STUB_COUNT\"; for last; do :; done; if [ \"$n\" -ge 1 ]; then exit 1; fi; printf 'WAVDATA' > \"$last\"",
    );
    let count = tmp.path().join("count");
    let _env = EnvGuard::set("AUDIO_STUB_COUNT", count.to_str().unwrap());
    let clip = tmp.path().join("clip.wav");
    std::fs::write(&clip, b"RIFF").unwrap();
    let cache = tmp.path().join("cache");
    let err = extract_audio(&clip, 30, &cache, Some(tools.as_path())).unwrap_err();
    assert!(err.is_file_error());
    let audio_dir = cache.join("audio");
    assert_eq!(std::fs::read_dir(&audio_dir).unwrap().count(), 0);
}
