//! Search: query parsing (prefixes, filters, semantic/keyword copies) and
//! ranking (RRF, SQL filters, JSON — later tasks).

pub mod parse;

pub use parse::{parse_query, FilterFlags, Filters, ParsedQuery};
