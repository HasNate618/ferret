//! Task 15: PDF primitives — binary resolution, page count, page text,
//! page rendering. Positive tests skip when poppler is absent; resolution
//! failures are tested without poppler.

use std::path::{Path, PathBuf};

use ferret::config::{ExtractConfig, compiled_defaults};
use ferret::extract::pdf::{page_count, page_text, render_page, resolve_tool};

fn test_dirs() -> ferret::paths::FerretDirs {
    ferret::paths::FerretDirs {
        config_dir: PathBuf::from("/ferret-pdf-cfg"),
        data_dir: PathBuf::from("/ferret-pdf-data"),
        cache_dir: PathBuf::from("/ferret-pdf-cache"),
    }
}

fn extract_cfg() -> ExtractConfig {
    compiled_defaults(&test_dirs()).extract
}

/// Positive tests need real poppler (mirrors the Task 14 magick gate).
fn host_has_poppler() -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| {
            ["pdfinfo", "pdftotext", "pdftoppm"]
                .iter()
                .all(|bin| dir.join(bin).is_file())
        })
    })
}

#[test]
fn page_count_of_fixture_is_two() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let cfg = extract_cfg();
    let pdfinfo = resolve_tool(&cfg, "pdfinfo").unwrap();
    assert_eq!(
        page_count(Path::new("testdata/two_pages.pdf"), &pdfinfo).unwrap(),
        2
    );
}

#[test]
fn page_text_matches_fixture() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let cfg = extract_cfg();
    let pdftotext = resolve_tool(&cfg, "pdftotext").unwrap();
    let t1 = page_text(Path::new("testdata/two_pages.pdf"), 1, &pdftotext).unwrap();
    assert!(
        t1.contains("page one pets"),
        "unexpected page 1 text: {t1:?}"
    );
    let t2 = page_text(Path::new("testdata/two_pages.pdf"), 2, &pdftotext).unwrap();
    assert!(
        t2.contains("page two ferrets"),
        "unexpected page 2 text: {t2:?}"
    );
}

#[test]
fn render_page_writes_png_under_cache() {
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let cfg = extract_cfg();
    let pdftoppm = resolve_tool(&cfg, "pdftoppm").unwrap();
    let png = render_page(
        Path::new("testdata/two_pages.pdf"),
        1,
        41,
        120,
        &pdftoppm,
        tmp.path(),
    )
    .unwrap();
    assert_eq!(png, tmp.path().join("pages/41-1.png"));
    let bytes = std::fs::read(&png).unwrap();
    let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).unwrap();
    use image::GenericImageView;
    assert!(img.width() > 0 && img.height() > 0);
}

#[test]
fn explicit_missing_binary_is_not_found_error() {
    // No poppler needed: an explicit bogus path must fail resolution.
    let mut cfg = extract_cfg();
    cfg.pdftotext = "/no/such/pdftotext".to_string();
    let err = resolve_tool(&cfg, "pdftotext").unwrap_err();
    assert!(err.to_string().contains("pdftotext"), "unexpected: {err}");
    assert!(err.to_string().contains("not found"), "unexpected: {err}");
}

#[test]
fn poppler_dir_is_preferred_over_path() {
    // poppler_dir set: resolution uses it even when PATH also has tools.
    if !host_has_poppler() {
        eprintln!("skip: no poppler on PATH");
        return;
    }
    let mut cfg = extract_cfg();
    let dir = Path::new("/etc/profiles/per-user/nate/bin");
    if !dir.join("pdfinfo").is_file() {
        eprintln!("skip: fixture poppler dir layout differs");
        return;
    }
    cfg.poppler_dir = dir.to_str().unwrap().to_string();
    assert_eq!(resolve_tool(&cfg, "pdfinfo").unwrap(), dir.join("pdfinfo"));
}

/// Write an executable shell stub: `name` echoes `stdout` and exits 0.
/// fsyncs file and directory: executing a just-written script can hit
/// ETXTBSY ("Text file busy") on some filesystems otherwise.
fn write_stub(dir: &Path, name: &str, stdout: &str) {
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\nprintf '%s' \"{stdout}\"\n")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
    }
    let file = std::fs::File::open(&path).unwrap();
    file.sync_all().unwrap();
    drop(file);
    let dir_fd = std::fs::File::open(dir).unwrap();
    dir_fd.sync_all().unwrap();
}

#[test]
fn resolve_priority_is_explicit_then_poppler_dir_then_path() {
    // Fully hermetic: stub binaries in tempdirs, no host poppler needed.
    let tmp = tempfile::tempdir().unwrap();
    let tools = tmp.path().join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    write_stub(&tools, "pdfinfo", "Pages: 7\n");
    write_stub(&tools, "pdftotext", "stub text");
    write_stub(&tools, "pdftoppm", "");

    // poppler_dir wins over PATH (whatever PATH holds).
    let mut cfg = extract_cfg();
    cfg.poppler_dir = tools.to_str().unwrap().to_string();
    assert_eq!(
        resolve_tool(&cfg, "pdfinfo").unwrap(),
        tools.join("pdfinfo")
    );
    // ...and the stub actually answers through the real call path.
    assert_eq!(
        page_count(Path::new("whatever.pdf"), &tools.join("pdfinfo")).unwrap(),
        7
    );

    // Explicit setting beats poppler_dir — including with a bogus value.
    cfg.pdftotext = "/no/such/pdftotext".to_string();
    let err = resolve_tool(&cfg, "pdftotext").unwrap_err();
    assert!(err.to_string().contains("pdftotext"), "unexpected: {err}");
    assert!(
        err.to_string().contains("/no/such/pdftotext"),
        "unexpected: {err}"
    );
}
