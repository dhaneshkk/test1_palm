//! Hand landmarks vs the Python reference (tools/handpose_ref.py). Needs ORT_DYLIB_PATH and
//! PALMID_HANDPOSE=<handpose_estimation_mediapipe_2023feb.onnx>; skipped otherwise.

use palmid::handpose::{aligned_palm_roi, HandLandmarker};
use palmid::imgproc::GrayImage;
use std::path::Path;

fn fx(n: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(n)
}

#[test]
fn landmarks_match_python() {
    let (Some(_), Ok(model)) = (std::env::var_os("ORT_DYLIB_PATH"), std::env::var("PALMID_HANDPOSE")) else {
        eprintln!("skipped: set ORT_DYLIB_PATH and PALMID_HANDPOSE");
        return;
    };
    let i = image::open(fx("gray.png")).unwrap().to_luma8();
    let g = GrayImage { w: i.width() as usize, h: i.height() as usize, data: i.into_raw() };
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fx("meta.json")).unwrap()).unwrap();
    let b: Vec<f32> = meta["box"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fx("handpose_ref.json")).unwrap()).unwrap();
    let mut hl = HandLandmarker::new(Path::new(&model), 2).unwrap();
    hl.min_confidence = 0.0;
    let pose = hl.landmarks(&g, [b[0], b[1], b[2], b[3]]).unwrap().expect("pose");
    let mut worst = 0f32;
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let (x, y) = (p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32);
        worst = worst.max(((pose.points[k][0] - x).powi(2) + (pose.points[k][1] - y).powi(2)).sqrt());
    }
    let dc = (pose.confidence - r["confidence"].as_f64().unwrap() as f32).abs();
    eprintln!("21 landmarks: worst distance {worst:.3} px, confidence {:.4} (Δ {dc:.4})", pose.confidence);
    assert!(worst < 1.5, "landmarks differ by {worst} px");
    assert!(dc < 0.02);
    let roi = aligned_palm_roi(&g, &pose, 128, 0.0);
    eprintln!("aligned ROI: {}", if roi.is_some() { "ok" } else { "outside frame" });
}
