//! CLI: `init`, `config show` / `config path`, exit codes, `--help`.

use ferret::cli::try_run;
use serial_test::serial;

#[test]
#[serial]
fn search_with_no_query_or_filters_is_usage() {
    let err = try_run(&["ferret", "search"]).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
#[serial]
fn init_refuses_existing_without_force() {
    let tmp = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("XDG_CONFIG_HOME", tmp.path()) };
    try_run(&["ferret", "init"]).unwrap();
    let err = try_run(&["ferret", "init"]).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
#[serial]
fn config_path_prints_defaults_layer() {
    let tmp = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("XDG_CONFIG_HOME", tmp.path()) };
    let out = try_run(&["ferret", "config", "path"]).unwrap();
    assert!(out.stdout.contains("defaults"));
}

#[test]
#[serial]
fn init_writes_roots_and_not_api_key() {
    let tmp = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("XDG_CONFIG_HOME", tmp.path()) };
    try_run(&["ferret", "init", "--root", "/tmp/docs"]).unwrap();
    let body = std::fs::read_to_string(tmp.path().join("ferret/config.toml")).unwrap();
    assert!(body.contains("roots"));
    assert!(!body.contains("api_key"));
    assert!(!body.contains("multimodal"));
}

#[test]
#[serial]
fn init_creates_config_dir_private_and_loadable() {
    let tmp = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("XDG_CONFIG_HOME", tmp.path()) };
    try_run(&[
        "ferret",
        "init",
        "--root",
        "/tmp/docs",
        "--root",
        "/tmp/pics",
    ])
    .unwrap();
    let cfg_dir = tmp.path().join("ferret");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&cfg_dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);
    }
    // The written file must load cleanly back through the config system.
    let out = try_run(&["ferret", "config", "show"]).unwrap();
    assert!(out.stdout.contains("/tmp/docs"));
    assert!(out.stdout.contains("/tmp/pics"));
    // --force overwrites; without it the existing file is refused (exit 2).
    try_run(&["ferret", "init", "--force", "--root", "/tmp/only"]).unwrap();
    let body = std::fs::read_to_string(cfg_dir.join("config.toml")).unwrap();
    assert!(body.contains("/tmp/only"));
    assert!(!body.contains("/tmp/docs"));
}

#[test]
#[serial]
fn init_rejects_colliding_collections() {
    let tmp = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("XDG_CONFIG_HOME", tmp.path()) };
    let a = tmp.path().join("docs");
    let b = tmp.path().join("elsewhere").join("docs");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    let err = try_run(&[
        "ferret",
        "init",
        "--root",
        a.to_str().unwrap(),
        "--root",
        b.to_str().unwrap(),
    ])
    .unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
#[serial]
fn unknown_flag_is_usage() {
    let err = try_run(&["ferret", "--bogus"]).unwrap_err();
    assert_eq!(err.exit_code(), 2);
    let err = try_run(&["ferret", "search", "--bogus"]).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn help_and_version_render_to_stdout() {
    let out = try_run(&["ferret", "--help"]).unwrap();
    assert!(out.stdout.contains("search"));
    assert!(out.stdout.contains("init"));
    let out = try_run(&["ferret", "search", "--help"]).unwrap();
    assert!(out.stdout.contains("starve the semantic arm"));
    assert!(out.stdout.contains("case-sensitive"));
    let out = try_run(&["ferret", "--version"]).unwrap();
    assert!(out.stdout.contains("ferret"));
}

#[test]
#[serial]
fn missing_config_flag_file_is_fatal() {
    let tmp = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("XDG_CONFIG_HOME", tmp.path()) };
    let missing = tmp.path().join("missing.toml");
    let err = try_run(&[
        "ferret",
        "--config",
        missing.to_str().unwrap(),
        "config",
        "show",
    ])
    .unwrap_err();
    assert_eq!(err.exit_code(), 1);
}

#[test]
#[serial]
#[cfg(unix)]
fn config_show_redacts_api_key() {
    let tmp = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("XDG_CONFIG_HOME", tmp.path()) };
    let file = tmp.path().join("ferret/config.toml");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(
        &file,
        "[embedder]\napi_key = \"sk-secret\"\nbackend = \"openai\"\n",
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(&file).unwrap().permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(&file, perms).unwrap();
    let out = try_run(&["ferret", "config", "show"]).unwrap();
    assert!(out.stdout.contains("***"));
    assert!(!out.stdout.contains("sk-secret"));
}

#[test]
#[serial]
fn config_path_lists_user_file_then_layers() {
    let tmp = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("XDG_CONFIG_HOME", tmp.path()) };
    try_run(&["ferret", "init"]).unwrap();
    let out = try_run(&["ferret", "config", "path"]).unwrap();
    let user_file = tmp.path().join("ferret/config.toml");
    assert!(out.stdout.contains(user_file.to_str().unwrap()));
    let defaults_line = out
        .stdout
        .lines()
        .last()
        .unwrap_or_default()
        .trim()
        .to_string();
    assert_eq!(defaults_line, "defaults");
}

#[test]
#[serial]
fn rm_and_gc_stubs_exit_zero() {
    let tmp = tempfile::tempdir().unwrap();
    // Isolate every XDG dir: under sandboxes (e.g. `nix build`) HOME is
    // not writable, so the default data dir must not be resolved there.
    unsafe {
        std::env::set_var("XDG_CONFIG_HOME", tmp.path().join("cfg"));
        std::env::set_var("XDG_DATA_HOME", tmp.path().join("data"));
        std::env::set_var("XDG_CACHE_HOME", tmp.path().join("cache"));
    }
    try_run(&["ferret", "rm", "/nonexistent/path"]).unwrap();
    try_run(&["ferret", "gc"]).unwrap();
}
