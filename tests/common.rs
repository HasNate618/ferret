use ferret::paths::FerretDirs;

pub fn test_dirs(tmp: &std::path::Path) -> FerretDirs {
    FerretDirs {
        config_dir: tmp.join("cfg"),
        data_dir: tmp.join("data"),
        cache_dir: tmp.join("cache"),
    }
}
