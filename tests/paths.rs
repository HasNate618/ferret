use ferret::paths::ferret_dirs;
use std::path::PathBuf;

fn env_map<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |k| {
        pairs
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    }
}

#[test]
fn xdg_data_home_wins_on_any_os() {
    let dirs = ferret_dirs(&env_map(&[("XDG_DATA_HOME", "/tmp/xdg-data")]));
    assert_eq!(dirs.data_dir, PathBuf::from("/tmp/xdg-data/ferret"));
}

#[test]
fn xdg_config_and_cache_win() {
    let dirs = ferret_dirs(&env_map(&[
        ("XDG_CONFIG_HOME", "/tmp/xdg-cfg"),
        ("XDG_CACHE_HOME", "/tmp/xdg-cache"),
    ]));
    assert_eq!(dirs.config_dir, PathBuf::from("/tmp/xdg-cfg/ferret"));
    assert_eq!(dirs.cache_dir, PathBuf::from("/tmp/xdg-cache/ferret"));
}

#[test]
fn missing_xdg_uses_platform_defaults() {
    let dirs = ferret_dirs(&|_| None);
    let cfg = dirs.config_dir.to_string_lossy();
    let data = dirs.data_dir.to_string_lossy();
    let cache = dirs.cache_dir.to_string_lossy();
    assert!(cfg.ends_with("ferret"), "{cfg}");
    assert!(data.ends_with("ferret"), "{data}");
    assert!(cache.ends_with("ferret"), "{cache}");
    #[cfg(target_os = "macos")]
    {
        assert!(cfg.contains("Library/Preferences"), "{cfg}");
        assert!(data.contains("Library/Application Support"), "{data}");
        assert!(cache.contains("Library/Caches"), "{cache}");
    }
    #[cfg(target_os = "linux")]
    {
        assert!(cfg.contains(".config/ferret"), "{cfg}");
        assert!(data.contains(".local/share/ferret"), "{data}");
        assert!(cache.contains(".cache/ferret"), "{cache}");
    }
}
