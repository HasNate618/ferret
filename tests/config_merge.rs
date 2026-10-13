use ferret::config::{LoadOpts, compiled_defaults, load_from};
use ferret::paths::FerretDirs;
use ferret::types::PrefixProfile;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn dirs() -> FerretDirs {
    FerretDirs {
        config_dir: PathBuf::from("/tmp/ferret-cfg"),
        data_dir: PathBuf::from("/tmp/ferret-data"),
        cache_dir: PathBuf::from("/tmp/ferret-cache"),
    }
}

#[test]
fn compiled_index_path_uses_data_dir() {
    let c = compiled_defaults(&dirs());
    assert_eq!(c.index.path, PathBuf::from("/tmp/ferret-data/index.sqlite"));
    assert!(c.index.roots.is_empty());
    assert!(matches!(
        c.embedder.prefix_profile,
        PrefixProfile::EmbeddingGemma2
    ));
}

#[test]
fn litert_backend_defaults_prefix_profile() {
    let files: HashMap<PathBuf, String> = HashMap::from([(
        PathBuf::from("/tmp/ferret-cfg/config.toml"),
        "[embedder]\nbackend = \"litert_lm\"\n".into(),
    )]);
    let (c, _) = load_from(
        &LoadOpts {
            config_flag: None,
            index_flag: None,
        },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|p| {
            files
                .get(p)
                .cloned()
                .ok_or_else(|| ferret::error::Error::fatal("missing"))
        },
    )
    .unwrap();
    assert!(matches!(c.embedder.prefix_profile, PrefixProfile::LitertLm));
}

#[test]
fn env_overrides_endpoint() {
    let (c, _) = load_from(
        &LoadOpts {
            config_flag: None,
            index_flag: None,
        },
        &dirs(),
        &|k| (k == "FERRET__EMBEDDER__ENDPOINT").then(|| "http://127.0.0.1:9999".into()),
        Path::new("/tmp/cwd"),
        &|_| Err(ferret::error::Error::fatal("no files")),
    )
    .unwrap();
    assert_eq!(c.embedder.endpoint, "http://127.0.0.1:9999");
}

#[test]
fn overlap_ge_max_chars_is_fatal() {
    let files: HashMap<PathBuf, String> = HashMap::from([(
        PathBuf::from("/tmp/ferret-cfg/config.toml"),
        "[chunk.text]\nmax_chars = 100\noverlap = 100\n".into(),
    )]);
    let err = load_from(
        &LoadOpts {
            config_flag: None,
            index_flag: None,
        },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|p| {
            files
                .get(p)
                .cloned()
                .ok_or_else(|| ferret::error::Error::fatal("missing"))
        },
    )
    .unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
fn missing_config_flag_is_fatal() {
    let err = load_from(
        &LoadOpts {
            config_flag: Some(PathBuf::from("/no/such/ferret.toml")),
            index_flag: None,
        },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|_| Err(ferret::error::Error::fatal("not found")),
    )
    .unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

fn user_files(map: &[(&str, &str)]) -> HashMap<PathBuf, String> {
    map.iter()
        .map(|(k, v)| (PathBuf::from(k), v.to_string()))
        .collect()
}

fn load_with(
    opts: &LoadOpts,
    files: &HashMap<PathBuf, String>,
) -> Result<(ferret::config::Config, Vec<ferret::config::LayerSource>), ferret::error::Error> {
    load_from(opts, &dirs(), &|_| None, Path::new("/tmp/cwd"), &|p| {
        files
            .get(p)
            .cloned()
            .ok_or_else(|| ferret::error::Error::fatal("missing"))
    })
}

#[test]
fn roots_table_form_and_layer_sources() {
    let files = user_files(&[(
        "/tmp/ferret-cfg/config.toml",
        "[[index.roots]]\npath = \"/tmp/nowhere\"\ncollection = \"docs\"\n\n[[index.roots]]\npath = \"/tmp/other\"\ncollection = \"pics\"\n",
    )]);
    let (c, layers) = load_with(&LoadOpts::default(), &files).unwrap();
    assert_eq!(c.index.roots.len(), 2);
    assert_eq!(c.index.roots[0].collection, "docs");
    assert_eq!(c.index.roots[0].path, PathBuf::from("/tmp/nowhere"));
    assert_eq!(layers.len(), 2, "{layers:?}");
    assert!(matches!(
        layers[1],
        ferret::config::LayerSource::UserFile(_)
    ));
}

#[test]
fn mixed_root_shapes_in_one_file_is_fatal() {
    let files = user_files(&[(
        "/tmp/ferret-cfg/config.toml",
        "[index]\nroots = [\"/tmp/nowhere\"]\n\n[[index.roots]]\npath = \"/tmp/other\"\ncollection = \"pics\"\n",
    )]);
    let err = load_with(&LoadOpts::default(), &files).unwrap_err();
    assert_eq!(err.exit_code(), 1, "{err}");
}

#[test]
fn duplicate_collections_are_fatal() {
    let files = user_files(&[(
        "/tmp/ferret-cfg/config.toml",
        "[index]\nroots = [\"/tmp/nowhere/docs\", \"/tmp/elsewhere/docs\"]\n",
    )]);
    let err = load_with(&LoadOpts::default(), &files).unwrap_err();
    assert_eq!(err.exit_code(), 1, "{err}");
}

#[test]
fn dimensions_outside_supported_set_is_fatal() {
    let files = user_files(&[(
        "/tmp/ferret-cfg/config.toml",
        "[embedder]\ndimensions = 64\n",
    )]);
    let err = load_with(&LoadOpts::default(), &files).unwrap_err();
    assert_eq!(err.exit_code(), 1, "{err}");
}

#[test]
fn cwd_file_layers_over_user_file() {
    let files = user_files(&[
        (
            "/tmp/ferret-cfg/config.toml",
            "[search]\nmode = \"keyword\"\n",
        ),
        ("/tmp/cwd/.ferret.toml", "[search]\nmode = \"semantic\"\n"),
    ]);
    let (c, layers) = load_with(&LoadOpts::default(), &files).unwrap();
    assert_eq!(c.search.mode, ferret::types::SearchMode::Semantic);
    assert!(
        layers.contains(&ferret::config::LayerSource::CwdFile(PathBuf::from(
            "/tmp/cwd/.ferret.toml"
        ))),
        "{layers:?}"
    );
}

#[test]
fn index_flag_overrides_and_adds_flags_layer() {
    let files = user_files(&[(
        "/tmp/ferret-cfg/config.toml",
        "[index]\npath = \"/tmp/other.sqlite\"\n",
    )]);
    let (c, layers) = load_with(
        &LoadOpts {
            config_flag: None,
            index_flag: Some(PathBuf::from("/tmp/flag.sqlite")),
        },
        &files,
    )
    .unwrap();
    assert_eq!(c.index.path, PathBuf::from("/tmp/flag.sqlite"));
    assert!(
        layers.contains(&ferret::config::LayerSource::Flags),
        "{layers:?}"
    );
    let (c, _) = load_with(&LoadOpts::default(), &files).unwrap();
    assert_eq!(c.index.path, PathBuf::from("/tmp/other.sqlite"));
}

#[test]
fn max_size_parses_bytes_and_kib_mib_gib() {
    for (text, bytes) in [
        ("4096", 4096u64),
        ("\"2KiB\"", 2048),
        ("\"40MiB\"", 40 * 1024 * 1024),
        ("\"1GiB\"", 1024 * 1024 * 1024),
    ] {
        let files = user_files(&[(
            "/tmp/ferret-cfg/config.toml",
            format!("[files]\nmax_size = {text}\n").as_str(),
        )]);
        let (c, _) = load_with(&LoadOpts::default(), &files).unwrap();
        assert_eq!(c.files.max_size, bytes, "{text}");
    }
    let files = user_files(&[(
        "/tmp/ferret-cfg/config.toml",
        "[files]\nmax_size = \"40MB\"\n",
    )]);
    let err = load_with(&LoadOpts::default(), &files).unwrap_err();
    assert_eq!(err.exit_code(), 1, "{err}");
}

#[test]
fn env_derives_prefix_profile_from_merged_backend() {
    let files = user_files(&[]);
    let (c, layers) = load_from(
        &LoadOpts::default(),
        &dirs(),
        &|k| match k {
            "FERRET__EMBEDDER__BACKEND" => Some("litert_lm".into()),
            "FERRET__EMBEDDER__MANAGED" => Some("true".into()),
            "FERRET__CHUNK__TEXT__MAX_CHARS" => Some("500".into()),
            _ => None,
        },
        Path::new("/tmp/cwd"),
        &|p| {
            files
                .get(p)
                .cloned()
                .ok_or_else(|| ferret::error::Error::fatal("missing"))
        },
    )
    .unwrap();
    assert!(matches!(c.embedder.prefix_profile, PrefixProfile::LitertLm));
    assert!(c.embedder.managed);
    assert_eq!(c.chunk.text.max_chars, 500);
    assert!(
        layers.contains(&ferret::config::LayerSource::Env),
        "{layers:?}"
    );
}

#[test]
fn invalid_env_values_are_fatal() {
    for (key, value) in [
        ("FERRET__EMBEDDER__DIMENSIONS", "not-a-number"),
        ("FERRET__EMBEDDER__BACKEND", "ollama"),
    ] {
        let err = load_from(
            &LoadOpts::default(),
            &dirs(),
            &|k| (k == key).then(|| value.to_string()),
            Path::new("/tmp/cwd"),
            &|_| Err(ferret::error::Error::fatal("no files")),
        )
        .unwrap_err();
        assert_eq!(err.exit_code(), 1, "{key} = {value}: {err}");
    }
}

#[test]
fn display_toml_redacts_empty_and_nonempty_api_key() {
    let files = user_files(&[(
        "/tmp/ferret-cfg/config.toml",
        "[embedder]\napi_key = \"\"\n",
    )]);
    let (c, _) = load_with(&LoadOpts::default(), &files).unwrap();
    let out = ferret::config::to_display_toml(&c);
    assert!(out.contains("api_key = \"\""), "{out}");
    assert!(!out.contains("multimodal"), "{out}");
    assert!(out.contains("[chunk.pdf]"), "{out}");
}

#[cfg(unix)]
#[test]
fn api_key_file_mode_0644_is_fatal() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.toml");
    std::fs::write(&p, "[embedder]\napi_key = \"secret\"\n").unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
    let err = load_from(
        &LoadOpts {
            config_flag: Some(p.clone()),
            index_flag: None,
        },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|path| {
            std::fs::read_to_string(path).map_err(|e| ferret::error::Error::fatal(e.to_string()))
        },
    )
    .unwrap_err();
    assert_eq!(err.exit_code(), 1);

    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).unwrap();
    let (c, _) = load_from(
        &LoadOpts {
            config_flag: Some(p),
            index_flag: None,
        },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|path| {
            std::fs::read_to_string(path).map_err(|e| ferret::error::Error::fatal(e.to_string()))
        },
    )
    .unwrap();
    assert!(ferret::config::to_display_toml(&c).contains("api_key = \"***\""));
}

#[test]
fn env_max_size_accepts_bare_digits_and_suffixes() {
    for (raw, want) in [
        ("52428800", 52428800u64),
        ("40MiB", 40 * 1024 * 1024),
        ("2KiB", 2048),
    ] {
        let (c, _) = load_from(
            &LoadOpts {
                config_flag: None,
                index_flag: None,
            },
            &dirs(),
            &|k| (k == "FERRET__FILES__MAX_SIZE").then(|| raw.into()),
            Path::new("/tmp/cwd"),
            &|_| Err(ferret::error::Error::fatal("no files")),
        )
        .unwrap();
        assert_eq!(c.files.max_size, want, "for FERRET__FILES__MAX_SIZE={raw}");
    }
}

#[test]
fn root_slash_shorthand_gets_nonempty_collection() {
    let files: HashMap<PathBuf, String> = HashMap::from([(
        PathBuf::from("/tmp/ferret-cfg/config.toml"),
        "[index]\nroots = [\"/\"]\n".into(),
    )]);
    let (c, _) = load_from(
        &LoadOpts {
            config_flag: None,
            index_flag: None,
        },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|p| {
            files
                .get(p)
                .cloned()
                .ok_or_else(|| ferret::error::Error::fatal("missing"))
        },
    )
    .unwrap();
    assert_eq!(c.index.roots.len(), 1);
    assert!(!c.index.roots[0].collection.is_empty());
}

#[test]
fn audio_kinds_parse_and_display() {
    use ferret::types::{ChunkKind, FileKind};
    assert_eq!("audio".parse::<FileKind>().unwrap(), FileKind::Audio);
    assert_eq!(ChunkKind::AudioSegment.to_string(), "audio_segment");
}

#[test]
fn compiled_defaults_include_audio() {
    let c = compiled_defaults(&dirs());
    for ext in ["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac"] {
        assert_eq!(c.files.extension_kind.get(ext), Some(&ferret::types::FileKind::Audio));
    }
    assert_eq!(c.chunk.audio.segment_secs, 30);
    assert!(c.extract.ffmpeg_dir.is_empty());
}

#[test]
fn zero_segment_secs_is_fatal() {
    use std::collections::HashMap;
    let files: HashMap<PathBuf, String> = HashMap::from([(
        PathBuf::from("/tmp/ferret-cfg/config.toml"),
        "[chunk.audio]\nsegment_secs = 0\n".into(),
    )]);
    let err = load_from(
        &LoadOpts { config_flag: None, index_flag: None },
        &dirs(),
        &|_| None,
        Path::new("/tmp/cwd"),
        &|p| files.get(p).cloned().ok_or_else(|| ferret::error::Error::fatal("missing")),
    )
    .unwrap_err();
    assert!(err.to_string().contains("segment_secs"));
}

#[test]
fn env_overrides_audio_and_ffmpeg_dir() {
    let (c, _) = load_from(
        &LoadOpts { config_flag: None, index_flag: None },
        &dirs(),
        &|k| match k {
            "FERRET__CHUNK__AUDIO__SEGMENT_SECS" => Some("45".into()),
            "FERRET__EXTRACT__FFMPEG_DIR" => Some("/opt/ffmpeg".into()),
            _ => None,
        },
        Path::new("/tmp/cwd"),
        &|_| Err(ferret::error::Error::fatal("no files")),
    )
    .unwrap();
    assert_eq!(c.chunk.audio.segment_secs, 45);
    assert_eq!(c.extract.ffmpeg_dir, "/opt/ffmpeg");
}
