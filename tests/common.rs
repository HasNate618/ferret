use ferret::paths::FerretDirs;
use std::path::PathBuf;

pub fn test_dirs(tmp: &std::path::Path) -> FerretDirs {
    FerretDirs {
        config_dir: tmp.join("cfg"),
        data_dir: tmp.join("data"),
        cache_dir: tmp.join("cache"),
    }
}

/// A whole test environment: XDG dirs, `FERRET_EMBEDDER=fake`, a corpus
/// root with one markdown file, and an `init`-ed config. Mutates the
/// process environment — callers must be `#[serial]`.
pub struct TestEnv {
    pub tmp: tempfile::TempDir,
    pub root: PathBuf,
    pub cfg: PathBuf,
    pub index: PathBuf,
}

pub fn test_env() -> TestEnv {
    let tmp = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("XDG_CONFIG_HOME", tmp.path().join("xdg-cfg"));
        std::env::set_var("XDG_DATA_HOME", tmp.path().join("xdg-data"));
        std::env::set_var("XDG_CACHE_HOME", tmp.path().join("xdg-cache"));
        std::env::set_var("FERRET_EMBEDDER", "fake");
    }
    let root = tmp.path().join("corpus");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("pets.md"), "pets in the lease").unwrap();
    ferret::cli::try_run(&["ferret", "init", "--root", root.to_str().unwrap()]).unwrap();
    TestEnv {
        cfg: tmp.path().join("xdg-cfg/ferret/config.toml"),
        index: tmp.path().join("xdg-data/ferret/index.sqlite"),
        root,
        tmp,
    }
}
