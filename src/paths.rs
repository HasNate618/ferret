use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FerretDirs {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

pub const INDEX_FILENAME: &str = "index.sqlite";

pub fn ferret_dirs(env: &dyn Fn(&str) -> Option<String>) -> FerretDirs {
    let config_dir = xdg_or(env, "XDG_CONFIG_HOME", platform_config_parent()).join("ferret");
    let data_dir = xdg_or(env, "XDG_DATA_HOME", platform_data_parent()).join("ferret");
    let cache_dir = xdg_or(env, "XDG_CACHE_HOME", platform_cache_parent()).join("ferret");
    FerretDirs {
        config_dir,
        data_dir,
        cache_dir,
    }
}

fn xdg_or(env: &dyn Fn(&str) -> Option<String>, key: &str, fallback: PathBuf) -> PathBuf {
    match env(key) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => fallback,
    }
}

fn platform_config_parent() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        dirs::home_dir()
            .expect("home directory")
            .join("Library/Preferences")
    }
    #[cfg(not(target_os = "macos"))]
    {
        dirs::config_dir().expect("config directory")
    }
}

fn platform_data_parent() -> PathBuf {
    dirs::data_dir().expect("data directory")
}

fn platform_cache_parent() -> PathBuf {
    dirs::cache_dir().expect("cache directory")
}
