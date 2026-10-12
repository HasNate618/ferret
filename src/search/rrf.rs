//! Reciprocal Rank Fusion (spec §6.2).
//!
//! `score(id) = Σ_arm weight_arm / (rrf_k + rank_arm)` with 1-based ranks.
//! An arm that did not list an id contributes nothing.

/// Which retrieval arm listed a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    Vec,
    Fts,
    Filename,
}

/// Fuse per-arm ranks into RRF scores, best first. Ties break on ascending
/// id so the output order is deterministic.
pub fn rrf(
    entries: &[(i64, Arm, u32)],
    k: f64,
    w_vec: f64,
    w_fts: f64,
    w_fn: f64,
) -> Vec<(i64, f64)> {
    let weight = |arm: Arm| match arm {
        Arm::Vec => w_vec,
        Arm::Fts => w_fts,
        Arm::Filename => w_fn,
    };

    // First-seen order preserved; sums accumulate per id.
    let mut fused: Vec<(i64, f64)> = Vec::new();
    for &(id, arm, rank) in entries {
        let term = weight(arm) / (k + f64::from(rank));
        if let Some(slot) = fused.iter_mut().find(|(i, _)| *i == id) {
            slot.1 += term;
        } else {
            fused.push((id, term));
        }
    }
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_ranked_entry_is_not_listed() {
        // Ranks are 1-based; an id only appears via a real entry.
        let fused = rrf(&[], 60.0, 1.0, 1.0, 0.4);
        assert!(fused.is_empty());
    }

    #[test]
    fn determinism_by_id_on_ties() {
        let fused = rrf(&[(9, Arm::Vec, 1), (4, Arm::Vec, 1)], 60.0, 1.0, 1.0, 0.4);
        assert_eq!(fused[0].0, 4);
        assert_eq!(fused[1].0, 9);
    }
}
