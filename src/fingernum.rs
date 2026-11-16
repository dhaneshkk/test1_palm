//! Chinese one-hand finger numbers (1–10) from the 21 MediaPipe hand landmarks: geometric rules,
//! no training.
//!
//! Each finger's state comes from 3-D joint angles in a hand-centred frame: origin at the wrist, unit
//! length = wrist → middle-finger knuckle. That makes the rules independent of position, scale and
//! in-plane rotation, and of which hand it is.
//!
//! | n  | gesture                                   | rule                                                 |
//! |----|-------------------------------------------|------------------------------------------------------|
//! | 1  | index up                                  | extended = index                                     |
//! | 2  | index + middle                            | extended = index, middle (not crossed)               |
//! | 3  | index + middle + ring, or middle+ring+pinky | either pattern                                      |
//! | 4  | four fingers, thumb folded                | extended = index … pinky                             |
//! | 5  | open hand                                 | all five extended                                    |
//! | 6  | thumb + little finger                     | extended = thumb, pinky                              |
//! | 7  | thumb, index and middle tips pinched      | tips within 0.30 units, held out; ring, pinky folded |
//! | 8  | thumb + index ("L")                       | extended = thumb, index                              |
//! | 9  | index hooked                              | index curled ≥ 100° but tip beyond its knuckle; rest tucked |
//! | 10 | fist, or index and middle crossed         | all curled with tips tucked; or tips crossed         |
//!
//! Regional variants differ (10 is a fist in some places, crossed fingers in others; 3 has two
//! forms): both forms are accepted. Anything else is `None`. The landmark model assumes an upright
//! hand, so this inherits that limit even though the rules themselves are rotation-invariant.

use crate::handpose::HandPose;
use serde::Serialize;
use std::collections::VecDeque;

type V = [f32; 3];

fn sub(a: V, b: V) -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn dot(a: V, b: V) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn norm(a: V) -> f32 { dot(a, a).sqrt() }
fn dist(a: V, b: V) -> f32 { norm(sub(a, b)) }

/// Bend at `b` in degrees: 0 when a → b → c is straight.
fn bend(a: V, b: V, c: V) -> f32 {
    let (u, v) = (sub(b, a), sub(c, b));
    let d = norm(u) * norm(v);
    if d < 1e-6 { return 0.0; }
    (dot(u, v) / d).clamp(-1.0, 1.0).acos().to_degrees()
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct FingerState {
    /// bend at the knuckle (wrist → MCP → PIP), degrees
    pub mcp: f32,
    /// thumb only: tip position across the palm (hand units, > 0 = outside on the thumb side)
    pub reach: f32,
    /// summed bend at the two middle joints, degrees
    pub curl: f32,
    pub extended: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct FingerNumber {
    pub value: u8,
    /// which rule fired, e.g. "pattern", "pinch", "hook", "fist", "crossed"
    pub rule: &'static str,
    /// thumb, index, middle, ring, pinky
    pub extended: [bool; 5],
}

/// Thresholds (degrees / hand units). Defaults chosen on the synthetic poses in tests/fingernum.rs
/// and checked on real open hands and fists (opencv_zoo demo clip).
#[derive(Clone, Copy, Debug)]
pub struct Rules {
    pub max_curl: f32,
    pub max_mcp: f32,
    pub thumb_max_curl: f32,
    pub thumb_min_reach: f32,
    pub pinch: f32,
    pub hook_mcp: f32,
    pub hook_curl: f32,
    /// a fingertip is "tucked" when its distance from the wrist is ≤ tuck × its knuckle's
    pub tuck: f32,
}

impl Default for Rules {
    fn default() -> Self {
        Self { max_curl: 80.0, max_mcp: 50.0, thumb_max_curl: 40.0, thumb_min_reach: 0.25, pinch: 0.30, hook_mcp: 40.0, hook_curl: 100.0, tuck: 1.15 }
    }
}

const FINGERS: [[usize; 4]; 4] = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];

/// Per-finger states (index, middle, ring, pinky) and the thumb, in hand units.
pub fn finger_states(p: &[V], r: &Rules) -> ([FingerState; 4], FingerState, f32) {
    let l = dist(p[0], p[9]).max(1e-6);
    let f = FINGERS.map(|[m, q, d, t]| {
        let mcp = bend(p[0], p[m], p[q]);
        let curl = bend(p[m], p[q], p[d]) + bend(p[q], p[d], p[t]);
        let extended = curl < r.max_curl && mcp < r.max_mcp && dist(p[t], p[0]) > dist(p[q], p[0]);
        FingerState { mcp, reach: 0.0, curl, extended }
    });
    let curl = bend(p[1], p[2], p[3]) + bend(p[2], p[3], p[4]);
    // Extended: tip out on the thumb side and straight, or far out (a bent but clearly abducted thumb).
    // reach: where the thumb tip sits along the across-palm axis (index knuckle → little-finger knuckle),
    // measured from the index knuckle; < 0 is outside the palm on the thumb side. A thumb folded
    // across the palm lands on the little-finger side (> 0).
    let side = sub(p[17], p[5]);
    let reach = -dot(sub(p[4], p[5]), side) / norm(side).max(1e-6) / l;
    let thumb = FingerState { mcp: bend(p[0], p[1], p[2]), reach, curl, extended: reach > r.thumb_min_reach && (curl < r.thumb_max_curl || reach > 1.8 * r.thumb_min_reach) };
    (f, thumb, l)
}

/// Classify one hand pose. `None` when no rule matches.
pub fn classify(pose: &HandPose, r: &Rules) -> Option<FingerNumber> {
    let p = &pose.points;
    if p.len() != 21 { return None; }
    let (f, thumb, l) = finger_states(p, r);
    let e = [thumb.extended, f[0].extended, f[1].extended, f[2].extended, f[3].extended];
    let out = |value, rule| Some(FingerNumber { value, rule, extended: e });

    // fingertip beyond its knuckle (seen from the wrist) vs tucked into the palm
    let tucked = |k: usize| dist(p[FINGERS[k][3]], p[0]) <= r.tuck * dist(p[FINGERS[k][0]], p[0]);
    let curled = |k: usize| f[k].curl >= r.hook_curl;
    // 7: thumb, index, middle tips together and held out in front of the knuckles; ring, pinky folded
    let c = [0, 1, 2].map(|k| (p[4][k] + p[8][k] + p[12][k]) / 3.0);
    if [4, 8, 12].iter().all(|&i| dist(p[i], c) < r.pinch * l) && !tucked(0) && !tucked(1)
        && !f[2].extended && !f[3].extended
    {
        return out(7, "pinch");
    }
    // 10 (fist): every finger curled with its tip tucked, thumb not out. Real fists show little
    // knuckle bend in the landmarks, so this looks at curl and tip position only.
    if !thumb.extended && (0..4).all(|k| curled(k) && tucked(k)) {
        return out(10, "fist");
    }
    // 9: index hooked — curled, but its tip still stands out beyond the knuckle; the rest tucked
    if !thumb.extended && curled(0) && !tucked(0) && f[0].mcp < r.hook_mcp && (1..4).all(|k| curled(k) && tucked(k)) {
        return out(9, "hook");
    }
    // 10 (crossed): index and middle up with their tips on the opposite sides of their knuckles
    let side = sub(p[17], p[5]); // across the palm, index → pinky
    let n = norm(side).max(1e-6);
    let (a, b) = (dot(sub(p[8], p[12]), side) / n, dot(sub(p[5], p[9]), side) / n);
    let crossed = a * b < 0.0 && a.abs() > 0.05 * l;
    let up = |s: &FingerState| s.curl < r.max_curl + 20.0 && s.mcp < r.max_mcp + 15.0;
    if crossed && up(&f[0]) && up(&f[1]) && !f[2].extended && !f[3].extended {
        return out(10, "crossed");
    }
    match e {
        [false, true, false, false, false] => out(1, "pattern"),
        [false, true, true, false, false] => out(2, "pattern"),
        [false, true, true, true, false] | [false, false, true, true, true] => out(3, "pattern"),
        [false, true, true, true, true] => out(4, "pattern"),
        [true, true, true, true, true] => out(5, "pattern"),
        [true, false, false, false, true] => out(6, "pattern"),
        [true, true, false, false, false] => out(8, "pattern"),
        _ => None, // includes half-closed hands (folded but neither a fist nor a hook): no number
    }
}

/// Report a number only once it holds: at least `need` of the last `window` frames agree, with
/// `2·need > window` so two different numbers can never both qualify (lean/PalmID.lean §8).
#[derive(Clone, Debug)]
pub struct Stabilizer {
    window: usize,
    need: usize,
    recent: VecDeque<Option<u8>>,
}

impl Default for Stabilizer {
    fn default() -> Self { Self::new(8, 6) }
}

impl Stabilizer {
    pub fn new(window: usize, need: usize) -> Self {
        assert!(2 * need > window && need <= window, "need a strict majority");
        Self { window, need, recent: VecDeque::with_capacity(window) }
    }

    pub fn push(&mut self, v: Option<u8>) -> Option<u8> {
        if self.recent.len() == self.window { self.recent.pop_front(); }
        self.recent.push_back(v);
        (1..=10).find(|&n| self.recent.iter().filter(|&&x| x == Some(n)).count() >= self.need)
    }
}
