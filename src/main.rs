fn main() {
    if let Err(err) = ferret::cli::run() {
        eprintln!("{err}");
        std::process::exit(err.exit_code());
    }
}
