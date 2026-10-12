//! Task 10: corpus walk, skip rules, gitignore, include globs.

use ferret::config::{compiled_defaults, FilesConfig};
use ferret::index::walk::{
    kind_for_ext, walk, SkipReason, WalkEvent,
};
use ferret::types::FileKind;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn files_cfg() -> FilesConfig {
    compiled_defaults(&ferret_test_dirs()).files
}

fn ferret_test_dirs() -> ferret::paths::FerretDirs {
    ferret::paths::FerretDirs {
        config_dir: PathBuf::from("/ferret-walk-cfg"),
        data_dir: PathBuf::from("/ferret-walk-data"),
        cache_dir: PathBuf::from("/ferret-walk-cache"),
    }
}

/// A root entry with the default collection (last path component).
fn root_entry(root: &Path) -> ferret::config::RootEntry {
    ferret::config::RootEntry {
        collection: root
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        path: root.to_path_buf(),
    }
}

fn walk_one(root: &Path) -> Vec<WalkEvent> {
    walk(&[root_entry(root)], &[], &files_cfg(), false).unwrap()
}

fn fixture_tree() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("corpus");
    std::fs::create_dir_all(root.join(".hidden")).unwrap();
    std::fs::write(root.join("ok.md"), "hi").unwrap();
    std::fs::write(root.join("secret.pem"), "k").unwrap();
    std::fs::write(root.join(".hidden/a.txt"), "x").unwrap();
    std::fs::write(root.join("skipme.rs"), "fn x(){}").unwrap();
    (tmp, root)
}

#[test]
fn pem_is_skip_rule() {
    let (_tmp, root) = fixture_tree();
    let ev = walk_one(&root);
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::Skip { reason: SkipReason::SkipRule, .. })));
}

#[test]
fn hidden_skipped_unless_include_hidden() {
    let (_tmp, root) = fixture_tree();
    let ev = walk_one(&root);
    // The `.hidden` directory is pruned (its contents are never visited),
    // which is the SkipRule event for this fixture.
    assert!(ev.iter().any(|e| match e {
        WalkEvent::Skip { path, reason: SkipReason::SkipRule } => path.ends_with(".hidden"),
        _ => false,
    }));
    // With include_hidden the hidden file is walked instead.
    let mut cfg = files_cfg();
    cfg.include_hidden = true;
    let ev = walk(&[root_entry(&root)], &[], &cfg, false).unwrap();
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::File { path, .. } if path.ends_with("a.txt"))));
}

#[test]
fn include_globs_skips_non_matching() {
    let (_tmp, root) = fixture_tree();
    let mut cfg = files_cfg();
    cfg.include_globs = vec!["*.md".into()];
    let ev = walk(&[root_entry(&root)], &[], &cfg, false).unwrap();
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::Skip { reason: SkipReason::IncludeGlobs, .. })));
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::File { kind: FileKind::Text, ext, .. } if ext == "md")));
}

#[test]
fn gitignore_skip() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("repo");
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::write(root.join(".gitignore"), "skipme.txt\n").unwrap();
    std::fs::write(root.join("skipme.txt"), "x").unwrap();
    std::fs::write(root.join("keep.md"), "y").unwrap();
    let ev = walk(&[root_entry(&root)], &[], &files_cfg(), false).unwrap();
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::Skip { reason: SkipReason::Gitignore, .. })));
    assert!(ev
        .iter()
        .any(|e| matches!(e, WalkEvent::File { path, .. } if path.ends_with("keep.md"))));
}

#[test]
fn extra_path_under_root_inherits_collection() {
    let (_tmp, root) = fixture_tree();
    let ev = walk(
        &[root_entry(&root)],
        &[root.join("ok.md")],
        &files_cfg(),
        false,
    )
    .unwrap();
    let file = ev
        .iter()
        .find_map(|e| match e {
            WalkEvent::File { path, collection, .. } if path.ends_with("ok.md") => {
                Some(collection.clone())
            }
            _ => None,
        })
        .expect("ok.md walked");
    assert_eq!(
        file,
        root.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap()
    );
}

// --- controller-added coverage -------------------------------------------

#[test]
fn skip_reason_strings_match_spec() {
    assert_eq!(SkipReason::UnsupportedExt.as_str(), "unsupported_ext");
    assert_eq!(SkipReason::SkipRule.as_str(), "skip_rule");
    assert_eq!(SkipReason::NoImageBackend.as_str(), "no_image_backend");
    assert_eq!(SkipReason::HeicSkip.as_str(), "heic_skip");
    assert_eq!(SkipReason::NoMagick.as_str(), "no_magick");
    assert_eq!(SkipReason::Gitignore.as_str(), "gitignore");
    assert_eq!(SkipReason::IncludeGlobs.as_str(), "include_globs");
}

#[test]
fn unsupported_ext_is_walk_skip() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("c");
    std::fs::create_dir_all(&root).unwrap();
    // `.rs` is a text kind in the default map (not a skip); `.mp4` is not
    // in the map at all.
    std::fs::write(root.join("code.rs"), "fn x() {}").unwrap();
    std::fs::write(root.join("clip.mp4"), "x").unwrap();
    let ev = walk_one(&root);
    assert!(ev.iter().any(|e| matches!(
        e,
        WalkEvent::Skip { path, reason: SkipReason::UnsupportedExt } if path.ends_with("clip.mp4")
    )));
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::File { ext, .. } if ext == "rs")));
}

#[test]
fn kind_for_ext_lookup() {
    let map: BTreeMap<String, FileKind> = compiled_defaults(&ferret_test_dirs())
        .files
        .extension_kind
        .clone();
    assert_eq!(kind_for_ext("md", &map), Some(FileKind::Text));
    assert_eq!(kind_for_ext("png", &map), Some(FileKind::Image));
    assert_eq!(kind_for_ext("pdf", &map), Some(FileKind::Pdf));
    assert_eq!(kind_for_ext("zzz", &map), None);
    // Extension match is case-insensitive on lookup? Caller lowercases;
    // kind_for_ext itself is exact on the map keys.
    assert_eq!(kind_for_ext("svg", &map), Some(FileKind::Text));
}

#[test]
fn gitignored_directory_pruned() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("repo");
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::create_dir_all(root.join("build")).unwrap();
    std::fs::write(root.join(".gitignore"), "build/\n").unwrap();
    std::fs::write(root.join("build/out.bin"), "x").unwrap();
    std::fs::write(root.join("keep.md"), "y").unwrap();
    let ev = walk(&[root_entry(&root)], &[], &files_cfg(), false).unwrap();
    // The directory is reported skipped and its contents are not walked.
    assert!(ev.iter().any(|e| matches!(
        e,
        WalkEvent::Skip { path, reason: SkipReason::Gitignore } if path.ends_with("build")
    )));
    assert!(!ev.iter().any(|e| matches!(e, WalkEvent::File { path, .. } if path.ends_with("out.bin"))));
}

#[test]
fn exclude_globs_and_max_size_are_skip_rules() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("c");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("big.md"), "abcdefghij").unwrap();
    std::fs::write(root.join("x.log"), "small").unwrap();
    let mut cfg = files_cfg();
    cfg.exclude_globs = vec!["*.log".into()];
    cfg.max_size = 5;
    let ev = walk(&[root_entry(&root)], &[], &cfg, false).unwrap();
    assert!(ev.iter().any(|e| matches!(
        e,
        WalkEvent::Skip { path, reason: SkipReason::SkipRule } if path.ends_with("x.log")
    )));
    assert!(ev.iter().any(|e| matches!(
        e,
        WalkEvent::Skip { path, reason: SkipReason::SkipRule } if path.ends_with("big.md")
    )));
}

#[test]
fn no_git_dir_means_no_gitignore_skips() {
    // respect_gitignore only applies inside a git repo (a .git at or above).
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("plain");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join(".gitignore"), "keep.md\n").unwrap();
    std::fs::write(root.join("keep.md"), "y").unwrap();
    let ev = walk(&[root_entry(&root)], &[], &files_cfg(), false).unwrap();
    assert!(ev.iter().any(|e| matches!(e, WalkEvent::File { path, .. } if path.ends_with("keep.md"))));
    assert!(!ev.iter().any(|e| matches!(e, WalkEvent::Skip { reason: SkipReason::Gitignore, .. })));
}

#[test]
fn explicit_heic_mapping_wins_over_magick_default() {
    // The default map omits heic/heif (magick mode types them as images),
    // but an explicit user mapping still wins.
    use ferret::types::HeicMode;
    let (_tmp, root) = fixture_tree();
    std::fs::write(root.join("pic.heic"), "x").unwrap();
    let mut cfg = files_cfg();
    cfg.heic = HeicMode::Magick;
    cfg.extension_kind.insert("heic".to_string(), FileKind::Text);
    let ev = walk(&[root_entry(&root)], &[], &cfg, false).unwrap();
    assert!(ev.iter().any(|e| matches!(
        e,
        WalkEvent::File { kind: FileKind::Text, ext, .. } if ext == "heic"
    )));
}

#[test]
fn heic_magick_mode_types_unmapped_heic_as_image() {
    // No user mapping + magick mode: the heic gate must not report
    // unsupported_ext (the magick path would otherwise be unreachable).
    use ferret::types::HeicMode;
    let (_tmp, root) = fixture_tree();
    std::fs::write(root.join("pic.heic"), "x").unwrap();
    let mut cfg = files_cfg();
    cfg.heic = HeicMode::Magick;
    let ev = walk(&[root_entry(&root)], &[], &cfg, false).unwrap();
    assert!(ev.iter().any(|e| matches!(
        e,
        WalkEvent::File { kind: FileKind::Image, ext, .. } if ext == "heic"
    )));
    assert!(!ev.iter().any(|e| matches!(
        e,
        WalkEvent::Skip { reason: SkipReason::UnsupportedExt, path } if path.ends_with("pic.heic")
    )));
}
