use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Usage(String),
    Fatal(String),
}

impl Error {
    pub fn usage(msg: impl Into<String>) -> Self {
        Self::Usage(msg.into())
    }
    pub fn fatal(msg: impl Into<String>) -> Self {
        Self::Fatal(msg.into())
    }
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            Self::Fatal(_) => 1,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(s) | Self::Fatal(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Fatal(e.to_string())
    }
}
