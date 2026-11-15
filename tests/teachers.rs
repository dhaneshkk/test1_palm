//! Teacher parity vs OpenCV FaceDetectorYN and the notebook palm decode.
//! Runs only with PALMID_TEACHERS=<dir with yunet.onnx, palm.onnx> and
//! PALMID_TEACHER_CASES=<img>|<ref.json>;... (references from tools/teacher_ref.py). Skipped otherwise.

use palmid::detector::iou;
use palmid::imgproc::RgbImage;
use palmid::teachers::Teachers;

fn best_iou(b: &[f32; 4], set: &[[f32; 4]]) -> f32 {
    set.iter().map(|r| iou(b, r)).fold(0.0, f32::max)
}

#[test]
fn teachers_match_opencv() {
    let (Ok(dir), Ok(cases)) = (std::env::var("PALMID_TEACHERS"), std::env::var("PALMID_TEACHER_CASES")) else {
        eprintln!("skipped: set PALMID_TEACHERS and PALMID_TEACHER_CASES");
        return;
    };
    let mut t = Teachers::new(&std::path::Path::new(&dir).join("yunet.onnx"), &std::path::Path::new(&dir).join("palm.onnx"), 2).unwrap();
    for case in cases.split(';') {
        let (img, refp) = case.split_once('|').unwrap();
        let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refp).unwrap()).unwrap();
        let get = |k: &str| -> Vec<[f32; 4]> {
            r[k].as_array().unwrap().iter().map(|b| { let v: Vec<f32> = b.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect(); [v[0], v[1], v[2], v[3]] }).collect()
        };
        let rgb = RgbImage::load(std::path::Path::new(img)).unwrap();
        let (faces, palms) = (t.faces(&rgb).unwrap(), t.palms(&rgb).unwrap());
        let (rf, rp) = (get("faces"), get("palms"));
        eprintln!("{img}: faces rust {} ref {} | palms rust {} ref {}", faces.len(), rf.len(), palms.len(), rp.len());
        assert_eq!(faces.len(), rf.len());
        assert_eq!(palms.len(), rp.len());
        for f in &rf { let m = best_iou(f, &faces); eprintln!("  face IoU {m:.4}"); assert!(m > 0.9); }
        for p in &rp { let m = best_iou(p, &palms); eprintln!("  palm IoU {m:.4}"); assert!(m > 0.98); }
    }
}

#[test]
fn gray_matches_opencv_cvtcolor() {
    let Ok(cases) = std::env::var("PALMID_GRAY_CASES") else { eprintln!("skipped: set PALMID_GRAY_CASES=<img>|<cv2 gray png>"); return; };
    for case in cases.split(';') {
        let (img, refp) = case.split_once('|').unwrap();
        let g = RgbImage::load(std::path::Path::new(img)).unwrap().to_gray();
        let r = image::open(refp).unwrap().to_luma8().into_raw();
        let diff = g.data.iter().zip(&r).filter(|(a, b)| a != b).count();
        eprintln!("gray differing pixels: {diff}");
        assert_eq!(diff, 0);
    }
}
