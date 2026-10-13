//! Shared value types: file/chunk kinds, backends, search modes, and their
//! spec strings (`Display` / `FromStr` and serde serialization all use the
//! same spelling, e.g. `pdf_page`, `litert_lm`, `embeddinggemma2`).

use serde::{Deserialize, Serialize};

macro_rules! spec_enum {
    (
        $(#[$doc:meta])*
        $name:ident {
            $($variant:ident => $spec:expr),+ $(,)?
        }
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum $name {
            $(
                #[serde(rename = $spec)]
                $variant,
            )+
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let s = match self {
                    $(Self::$variant => $spec,)+
                };
                f.write_str(s)
            }
        }

        impl std::str::FromStr for $name {
            type Err = String;

            fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
                match s {
                    $($spec => Ok(Self::$variant),)+
                    other => Err(format!(
                        "unknown {} `{other}` (expected one of {})",
                        stringify!($name),
                        [$($spec),+].join(", ")
                    )),
                }
            }
        }
    };
}

spec_enum! {
    /// v1 file kinds. Only `text`, `image`, and `pdf` exist in v1.
    FileKind {
        Text => "text",
        Image => "image",
        Pdf => "pdf",
    }
}

spec_enum! {
    /// v1 chunk kinds. Only `text`, `image`, and `pdf_page` exist in v1.
    ChunkKind {
        Text => "text",
        Image => "image",
        PdfPage => "pdf_page",
    }
}

spec_enum! {
    /// Query task kinds parsed out of the query string.
    QueryTask {
        Search => "search",
        Code => "code",
    }
}

spec_enum! {
    /// Embedding backends. No Ollama in v1.
    Backend {
        LlamaServer => "llama_server",
        LitertLm => "litert_lm",
        Sglang => "sglang",
        Openai => "openai",
    }
}

spec_enum! {
    /// Embedding prompt-prefix profile (spec §3.2).
    PrefixProfile {
        EmbeddingGemma2 => "embeddinggemma2",
        LitertLm => "litert_lm",
    }
}

spec_enum! {
    /// Search fusion modes.
    SearchMode {
        Hybrid => "hybrid",
        Semantic => "semantic",
        Keyword => "keyword",
    }
}

spec_enum! {
    /// Text chunking modes.
    TextChunkMode {
        Chars => "chars",
        Sentences => "sentences",
    }
}

spec_enum! {
    /// `.heic` / `.heif` handling. `skip` is the v1 default (no ImageMagick).
    HeicMode {
        Skip => "skip",
        Magick => "magick",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn display_uses_spec_strings() {
        assert_eq!(FileKind::Pdf.to_string(), "pdf");
        assert_eq!(ChunkKind::PdfPage.to_string(), "pdf_page");
        assert_eq!(Backend::LlamaServer.to_string(), "llama_server");
        assert_eq!(Backend::LitertLm.to_string(), "litert_lm");
        assert_eq!(
            PrefixProfile::EmbeddingGemma2.to_string(),
            "embeddinggemma2"
        );
        assert_eq!(SearchMode::Hybrid.to_string(), "hybrid");
        assert_eq!(TextChunkMode::Sentences.to_string(), "sentences");
        assert_eq!(HeicMode::Skip.to_string(), "skip");
        assert_eq!(QueryTask::Code.to_string(), "code");
    }

    #[test]
    fn from_str_round_trips() {
        for s in ["text", "image", "pdf"] {
            assert_eq!(FileKind::from_str(s).unwrap().to_string(), s);
        }
        for s in ["llama_server", "litert_lm", "sglang", "openai"] {
            assert_eq!(Backend::from_str(s).unwrap().to_string(), s);
        }
        assert!(FileKind::from_str("audio").is_err());
        assert!(Backend::from_str("ollama").is_err());
    }
}
