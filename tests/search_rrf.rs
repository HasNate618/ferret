//! Task 8: RRF fusion.

use ferret::search::rrf::{Arm, rrf};

#[test]
fn rrf_prefers_agreement() {
    // chunk 1 rank 1 vec + rank 1 fts; chunk 2 rank 2 vec only
    let fused = rrf(
        &[(1, Arm::Vec, 1), (1, Arm::Fts, 1), (2, Arm::Vec, 2)],
        60.0,
        1.0,
        1.0,
        0.4,
    );
    assert_eq!(fused[0].0, 1);
    let s1 = 1.0 / 61.0 + 1.0 / 61.0;
    assert!((fused[0].1 - s1).abs() < 1e-12);
}

#[test]
fn rrf_orders_by_fused_score_desc() {
    let fused = rrf(
        &[(7, Arm::Vec, 1), (9, Arm::Fts, 1), (7, Arm::Fts, 5)],
        60.0,
        1.0,
        1.0,
        0.4,
    );
    // 7: 1/61 + 1/65; 9: 1/61 -> 7 wins on the extra arm
    assert_eq!(fused[0].0, 7);
    assert_eq!(fused[1].0, 9);
    assert!(fused[0].1 > fused[1].1);
}

#[test]
fn rrf_filename_weight_applies() {
    let fused = rrf(&[(1, Arm::Filename, 1)], 60.0, 1.0, 1.0, 0.4);
    assert!((fused[0].1 - 0.4 / 61.0).abs() < 1e-12);
}

#[test]
fn rrf_same_id_multiple_entries_sum() {
    let fused = rrf(&[(3, Arm::Vec, 2), (3, Arm::Fts, 3)], 60.0, 1.0, 1.0, 0.4);
    let expect = 1.0 / 62.0 + 1.0 / 63.0;
    assert_eq!(fused.len(), 1);
    assert!((fused[0].1 - expect).abs() < 1e-12);
}
