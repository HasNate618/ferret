use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Usage(String),
    Fatal(String),
    /// One file's embedding failed (e.g. the server 4xx'd its text).
    /// The indexer records this as a per-file error row and continues;
    /// anything reaching the top level exits 1 like `Fatal`.
    File(String),
}

impl Error {
    pub fn usage(msg: impl Into<String>) -> Self {
        Self::Usage(msg.into())
    }
    pub fn fatal(msg: impl Into<String>) -> Self {
        Self::Fatal(msg.into())
    }
    pub fn file(msg: impl Into<String>) -> Self {
        Self::File(msg.into())
    }
    /// True for per-file errors: the indexer records a row and continues
    /// instead of aborting the walk.
    pub fn is_file_error(&self) -> bool {
        matches!(self, Self::File(_))
    }
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            Self::Fatal(_) | Self::File(_) => 1,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(s) | Self::Fatal(s) | Self::File(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Fatal(e.to_string())
    }
}
