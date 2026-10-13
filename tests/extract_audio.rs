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
