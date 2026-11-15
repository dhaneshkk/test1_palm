//! Integer mirrors of the definitions proven in `lean/PalmID.lean` (Lean 4.31, kernel-only).
//! Each function below is written to be read side-by-side with its Lean definition.

/// Lean: `absDiff a b := if a ≤ b then b - a else a - b`
#[inline]
pub fn abs_diff(a: u8, b: u8) -> u8 {
    if a <= b { b - a } else { a - b }
}

/// Lean: `angDist a b := if absDiff a b ≤ 3 then absDiff a b else 6 - absDiff a b`
/// (orientation codes on the 6-cycle; theorems `angDist_symm`, `angDist_self`, `angDist_le_three`)
#[inline]
pub fn ang_dist(a: u8, b: u8) -> u8 {
    let d = abs_diff(a, b);
    if d <= 3 { d } else { 6 - d }
}

/// Lean: `accept dist tau := dist ≤ tau` (theorems `accept_monotone`, `min_accept`)
#[inline]
pub fn accept(dist: f64, tau: f64) -> bool {
    dist <= tau
}

/// Lean: `closerRel d t d' t' := d * t' < d' * t` — division-free ratio ranking (`closerRel_irrefl`)
#[inline]
pub fn closer(d: f64, t: f64, d2: f64, t2: f64) -> bool {
    d * t2 < d2 * t
}

/// Lean: `confRank n := ((n + 1) * 29 + 29) / 30` = ⌈(n+1)(1−α)⌉ at α = 1/30
/// (`conformal_feasible`: n ≥ 29 → confRank n ≤ n; `conformal_tight`: confRank 28 > 28)
#[inline]
pub fn conf_rank(n: usize) -> usize {
    ((n + 1) * 29 + 29) / 30
}

/// Minimum calibration set size for which the conformal rank is feasible (Lean §3).
pub const N_CAL_FORMAL_BOUND: usize = 29;

/// Lean: `step N cnt hit := if hit then min (cnt + 1) N else cnt - 1` (Nat subtraction saturates at 0)
/// (`step_bounded`, `step_lipschitz`, `run_le`, `announce_needs_K_frames`)
#[inline]
pub fn step(n: u32, cnt: u32, hit: bool) -> u32 {
    if hit { (cnt + 1).min(n) } else { cnt.saturating_sub(1) }
}

/// Lean §5 `gate_cells`: the resolution gate (side ≥ min_side) guarantees ≥ min_side/stride code cells.
#[inline]
pub fn resolution_gate(side: f32, min_side: f32) -> bool {
    side >= min_side
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ang_dist_theorems_exhaustive() {
        for a in 0u8..6 {
            assert_eq!(ang_dist(a, a), 0); // angDist_self
            for b in 0u8..6 {
                assert_eq!(ang_dist(a, b), ang_dist(b, a)); // angDist_symm
                assert!(ang_dist(a, b) <= 3); // angDist_le_three
            }
        }
    }

    #[test]
    fn conformal_theorems() {
        for n in 29..200_000 {
            assert!(conf_rank(n) <= n, "conformal_feasible fails at {n}");
        }
        assert!(conf_rank(28) > 28); // conformal_tight
    }

    #[test]
    fn debounce_theorems() {
        // step_bounded and run_le over all frame sequences of length ≤ 12
        for n in 1u32..8 {
            for len in 0..=12u32 {
                for mask in 0u32..(1 << len) {
                    let mut c = 0u32;
                    for i in 0..len {
                        let prev = c;
                        c = step(n, c, mask >> i & 1 == 1);
                        assert!(c <= n); // step_bounded
                        assert!(c <= prev + 1); // step_lipschitz
                    }
                    assert!(c <= len); // run_le from 0 ⇒ announce_needs_K_frames
                }
            }
        }
    }

    #[test]
    fn accept_and_rank() {
        assert!(accept(0.3, 0.3) && accept(0.3, 0.4)); // accept_monotone instance
        assert!(!closer(0.3, 0.4, 0.3, 0.4)); // closerRel_irrefl instance
    }
}
