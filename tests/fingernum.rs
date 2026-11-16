//! Chinese finger numbers on synthetic 3-D hand poses (forward kinematics), under in-plane rotation,
//! scale, translation, mirroring and landmark noise. These check the rules are self-consistent;
//! accuracy on real hands is measured with `palmid fingers-eval` on recorded, labelled clips.

use palmid::fingernum::{classify, Rules, Stabilizer};
use palmid::handpose::HandPose;

type V = [f32; 3];

/// One finger: knuckle position (hand units, x across, y up the hand), base direction angle from
/// "up" (degrees, + toward the thumb side), segment lengths, joint flexions (MCP, PIP, DIP).
fn chain(mcp: V, dir_deg: f32, len: [f32; 3], flex: [f32; 3]) -> [V; 4] {
    let a = dir_deg.to_radians();
    let d = [-a.sin(), a.cos(), 0.0]; // in-plane direction
    let n = [0.0, 0.0, -1.0];        // toward the camera (palm faces the camera)
    let mut pts = [mcp; 4];
    let mut th = 0f32;
    for k in 0..3 {
        th += flex[k].to_radians();
        let dir = [d[0] * th.cos() + n[0] * th.sin(), d[1] * th.cos() + n[1] * th.sin(), d[2] * th.cos() + n[2] * th.sin()];
        let p = pts[k];
        pts[k + 1] = [p[0] + dir[0] * len[k], p[1] + dir[1] * len[k], p[2] + dir[2] * len[k]];
    }
    pts
}

const UP: [f32; 3] = [0.0, 0.0, 0.0];
const FOLD: [f32; 3] = [85.0, 100.0, 60.0];
const HOOK: [f32; 3] = [5.0, 90.0, 45.0];

#[derive(Clone, Copy)]
struct Pose { f: [[f32; 3]; 4], dirs: [f32; 4], thumb_out: bool, pinch: bool }

fn pose(f: [[f32; 3]; 4], thumb_out: bool) -> Pose {
    Pose { f, dirs: [8.0, 0.0, -6.0, -14.0], thumb_out, pinch: false }
}

fn build(p: Pose) -> Vec<V> {
    let mcps: [V; 4] = [[-0.26, 0.95, 0.0], [0.0, 1.0, 0.0], [0.23, 0.93, 0.0], [0.43, 0.82, 0.0]];
    let lens = [[0.45, 0.27, 0.22], [0.50, 0.30, 0.24], [0.46, 0.28, 0.22], [0.36, 0.22, 0.20]];
    let mut pts = vec![[0.0, 0.0, 0.0]; 21];
    let mut f = [[[0f32; 3]; 4]; 4];
    for k in 0..4 {
        f[k] = chain(mcps[k], p.dirs[k], lens[k], p.f[k]);
        for j in 0..4 { pts[5 + 4 * k + j] = f[k][j]; }
    }
    let cmc = [-0.30, 0.22, 0.0];
    let thumb = if p.pinch {
        // thumb tip meets the index and middle tips
        let t = [0, 1, 2].map(|i| (f[0][3][i] + f[1][3][i]) / 2.0);
        let m = [(cmc[0] + t[0]) / 2.0 - 0.12, (cmc[1] + t[1]) / 2.0, (cmc[2] + t[2]) / 2.0 - 0.1];
        [cmc, [(cmc[0] + m[0]) / 2.0, (cmc[1] + m[1]) / 2.0, (cmc[2] + m[2]) / 2.0], m, t]
    } else if p.thumb_out {
        chain(cmc, 55.0, [0.35, 0.30, 0.25], [0.0, 5.0, 5.0])
    } else {
        chain(cmc, -40.0, [0.30, 0.26, 0.22], [35.0, 45.0, 30.0]) // across the palm
    };
    for j in 0..4 { pts[1 + j] = thumb[j]; }
    pts
}

/// Image coordinates: y down; rotate, scale, translate, optional mirror, deterministic noise.
fn to_image(pts: &[V], deg: f32, scale: f32, mirror: bool, noise: f32, seed: u64) -> HandPose {
    let (s, c) = deg.to_radians().sin_cos();
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((x >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0 };
    let points = pts.iter().map(|p| {
        let px = if mirror { -p[0] } else { p[0] };
        let (u, v) = (px * c - p[1] * s, px * s + p[1] * c);
        [320.0 + scale * (u + noise * rnd()), 400.0 - scale * (v + noise * rnd()), scale * (p[2] + 2.0 * noise * rnd())]
    }).collect();
    HandPose { points, confidence: 0.95, handedness: 0.5 }
}

fn cases() -> Vec<(u8, &'static str, Pose)> {
    let mut seven = pose([[50.0, 35.0, 20.0], [50.0, 35.0, 20.0], FOLD, FOLD], false);
    seven.pinch = true;
    let mut crossed = pose([UP, UP, FOLD, FOLD], false);
    crossed.dirs = [-14.0, 12.0, -6.0, -14.0];
    vec![
        (1, "1", pose([UP, FOLD, FOLD, FOLD], false)),
        (2, "2", pose([UP, UP, FOLD, FOLD], false)),
        (3, "3 (index-middle-ring)", pose([UP, UP, UP, FOLD], false)),
        (3, "3 (middle-ring-pinky)", pose([FOLD, UP, UP, UP], false)),
        (4, "4", pose([UP, UP, UP, UP], false)),
        (5, "5", pose([UP, UP, UP, UP], true)),
        (6, "6", pose([FOLD, FOLD, FOLD, UP], true)),
        (7, "7", seven),
        (8, "8", pose([UP, FOLD, FOLD, FOLD], true)),
        (9, "9", pose([HOOK, FOLD, FOLD, FOLD], false)),
        (10, "10 (fist)", pose([FOLD, FOLD, FOLD, FOLD], false)),
        (10, "10 (crossed)", crossed),
    ]
}

#[test]
fn every_number_under_rotation_scale_mirror_noise() {
    let r = Rules::default();
    let mut total = 0;
    let mut bad = vec![];
    for (want, name, p) in cases() {
        let pts = build(p);
        for (i, &deg) in [-35.0f32, -15.0, 0.0, 20.0, 40.0].iter().enumerate() {
            for &scale in &[60.0f32, 140.0] {
                for &mirror in &[false, true] {
                    for seed in 0..4u64 {
                        let h = to_image(&pts, deg, scale, mirror, 0.025, seed + 10 * i as u64);
                        let got = classify(&h, &r);
                        if got.as_ref().map(|g| g.value) != Some(want) {
                            let (f, t, _) = palmid::fingernum::finger_states(&h.points, &r);
                            eprintln!("{name} rot {deg} s {scale} m {mirror} seed {seed}: thumb {t:?} fingers {f:?}");
                        }
                        if got.as_ref().map(|g| g.value) != Some(want) { bad.push(format!("{name}: rot {deg}° scale {scale} mirror {mirror} seed {seed} → {:?}", got.map(|g| g.value))); }
                        total += 1;
                    }
                }
            }
        }
    }
    eprintln!("{total} synthetic poses, {} wrong", bad.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn half_folded_is_not_a_number_change() {
    // fingers half way between up and folded: rules must not flip to another number spuriously
    let r = Rules::default();
    let half = [40.0, 50.0, 30.0];
    let h = to_image(&build(pose([UP, half, FOLD, FOLD], false)), 0.0, 100.0, false, 0.0, 0);
    let v = classify(&h, &r).map(|g| g.value);
    assert!(matches!(v, Some(1) | Some(2) | None), "{v:?}");
}

#[test]
fn stabilizer_needs_a_majority() {
    let mut s = Stabilizer::new(8, 6);
    let seq = [Some(3), Some(3), None, Some(3), Some(3), Some(3), Some(3)];
    let out: Vec<_> = seq.iter().map(|&v| s.push(v)).collect();
    assert_eq!(out, vec![None, None, None, None, None, None, Some(3)]);
    for _ in 0..3 { s.push(Some(5)); }
    assert_eq!(s.push(Some(5)), None); // 4 × 3 and 4 × 5 in the window: neither holds
}

/// Real landmarks (opencv_zoo colour pipeline) from a JSON-lines file: {"frame","points","label"?}.
/// PALMID_FINGER_CASES=<file.jsonl>; prints the number per frame, and accuracy where labelled.
#[test]
fn real_landmarks_from_file() {
    let Ok(p) = std::env::var("PALMID_FINGER_CASES") else { eprintln!("skipped: set PALMID_FINGER_CASES"); return };
    let r = Rules::default();
    let (mut n, mut ok, mut labelled) = (0, 0, 0);
    for line in std::fs::read_to_string(p).unwrap().lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let points = v["points"].as_array().unwrap().iter().map(|q| [q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32, q[2].as_f64().unwrap() as f32]).collect();
        let got = classify(&HandPose { points, confidence: 1.0, handedness: 0.5 }, &r);
        n += 1;
        if let Some(l) = v["label"].as_u64() {
            labelled += 1;
            ok += (got.as_ref().map(|g| g.value as u64) == Some(l)) as usize;
        }
        eprintln!("frame {} → {:?}", v["frame"], got.map(|g| (g.value, g.rule, g.extended)));
    }
    eprintln!("{n} hands, {ok}/{labelled} labelled correct");
}

/// Follow one hand through a frame sequence with landmarks only (detector box on the first frame),
/// as the web app does when the detector drops out. PALMID_FINGER_TRACK=<dir with NNN.png and
/// boxes.json>|<first frame>|<box index>|<handpose.onnx>; needs ORT_DYLIB_PATH.
#[test]
fn track_through_frames() {
    let (Some(_), Ok(spec)) = (std::env::var_os("ORT_DYLIB_PATH"), std::env::var("PALMID_FINGER_TRACK")) else { eprintln!("skipped"); return };
    let s: Vec<&str> = spec.split('|').collect();
    let boxes: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/boxes.json", s[0])).unwrap()).unwrap();
    let first: usize = s[1].parse().unwrap();
    let k: usize = s[2].parse().unwrap();
    let mut hl = palmid::handpose::HandLandmarker::new(std::path::Path::new(s[3]), 2).unwrap();
    let load = |f: usize| {
        let i = image::open(format!("{}/{f:03}.png", s[0])).unwrap().to_luma8();
        palmid::imgproc::GrayImage { w: i.width() as usize, h: i.height() as usize, data: i.into_raw() }
    };
    let b = &boxes[first.to_string()][k];
    let mut bx = [0, 1, 2, 3].map(|j| b[j].as_f64().unwrap() as f32);
    // box consistency on detected frames: IoU(detector box, box from its own landmarks)
    let mut ious = vec![];
    for (f, bs) in boxes.as_object().unwrap() {
        for b in bs.as_array().unwrap() {
            let d = [0, 1, 2, 3].map(|j| b[j].as_f64().unwrap() as f32);
            if let Some(h) = hl.landmarks(&load(f.parse().unwrap()), d).unwrap() {
                let e = palmid::handpose::palm_box_from_landmarks(&h);
                ious.push((palmid::detector::iou(&d, &e), (e[2] - e[0]) / (d[2] - d[0]), ((d[0] + d[2]) - (e[0] + e[2])) / 2.0 / (e[2] - e[0]), ((d[1] + d[3]) - (e[1] + e[3])) / 2.0 / (e[2] - e[0]), (d[3]-d[1])/(d[2]-d[0])));
            }
        }
    }
    let n = ious.len() as f32;
    eprintln!("detector vs landmark box over {} hands: mean IoU {:.3}, side ratio {:.3}", ious.len(), ious.iter().map(|x| x.0).sum::<f32>() / n, ious.iter().map(|x| x.1).sum::<f32>() / n);
    eprintln!("centre offset (detector − landmark)/side: dx {:.3} dy {:.3}; detector aspect {:.3}", ious.iter().map(|x| x.2).sum::<f32>() / n, ious.iter().map(|x| x.3).sum::<f32>() / n, ious.iter().map(|x| x.4).sum::<f32>() / n);
    let r = Rules::default();
    let mut stab = Stabilizer::default();
    for f in first..=boxes.as_object().unwrap().len() {
        match hl.landmarks(&load(f), bx).unwrap() {
            Some(h) => {
                let g = classify(&h, &r);
                let (fs, t, l) = palmid::fingernum::finger_states(&h.points, &r);
                let d = |a: usize| { let (p, q) = (h.points[a], h.points[0]); ((p[0]-q[0]).powi(2)+(p[1]-q[1]).powi(2)+(p[2]-q[2]).powi(2)).sqrt() / l };
                eprintln!("  states: {}; thumb curl {:.0} reach {:.2} ext {}; tip/knuckle/pip dist: {}", fs.iter().map(|s| format!("mcp {:.0} curl {:.0}", s.mcp, s.curl)).collect::<Vec<_>>().join(" | "), t.curl, t.reach, t.extended,
                    [(8,5,6),(12,9,10),(16,13,14),(20,17,18)].iter().map(|&(a,b,c)| format!("{:.2}/{:.2}/{:.2}", d(a), d(b), d(c))).collect::<Vec<_>>().join(" "));
                eprintln!("frame {f}: conf {:.2} → {:?} held {:?}", h.confidence, g.as_ref().map(|g| (g.value, g.rule)), stab.push(g.map(|g| g.value)));
                bx = palmid::handpose::palm_box_from_landmarks(&h);
            }
            None => { eprintln!("frame {f}: lost"); break; }
        }
    }
}

#[test]
fn evaluate_counts_and_mcc() {
    use palmid::fingernum::{evaluate, Recording};
    let r = Rules::default();
    let mut recs = vec![];
    for (label, _, p) in cases() {
        for seed in 0..5u64 {
            let h = to_image(&build(p), 10.0, 100.0, false, 0.01, seed);
            recs.push(Recording { label, points: h.points, confidence: 0.95, frame: seed });
        }
    }
    // one deliberately wrong label
    recs[0].label = 2;
    let rep = evaluate(&recs, &r);
    assert_eq!(rep.frames, recs.len());
    assert_eq!(rep.confusion[2][1], 1);
    assert!((rep.accuracy - (recs.len() - 1) as f64 / recs.len() as f64).abs() < 1e-12);
    assert!(rep.mcc.unwrap() > 0.95 && rep.mcc.unwrap() < 1.0);
    eprintln!("accuracy {:.3}, MCC {:.3}", rep.accuracy, rep.mcc.unwrap());
}
