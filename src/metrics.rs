//! MCC evaluation and Mondrian split-conformal threshold calibration on YOLO-format data
//! (`images/<split>/*.png|jpg`, `labels/<split>/*.txt` with `cls cx cy w h` normalised).

use crate::detector::{iou, Detector};
use crate::imgproc::GrayImage;
use crate::lean_mirror::{conf_rank, N_CAL_FORMAL_BOUND};
use anyhow::Result;
use serde::Serialize;
use std::path::{Path, PathBuf};

pub fn mcc(tp: u64, fp: u64, fn_: u64, tn: u64) -> f64 {
    let (tp, fp, fn_, tn) = (tp as f64, fp as f64, fn_ as f64, tn as f64);
    let d = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if d == 0.0 { 0.0 } else { (tp * tn - fp * fn_) / d }
}

pub struct Sample {
    pub image: PathBuf,
    pub boxes: Vec<(usize, [f32; 4])>,
}

pub fn load_split(root: &Path, split: &str) -> Result<Vec<Sample>> {
    let mut out = vec![];
    let mut paths: Vec<PathBuf> = std::fs::read_dir(root.join("images").join(split))?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for p in paths {
        let img = image::image_dimensions(&p)?;
        let (w, h) = (img.0 as f32, img.1 as f32);
        let lp = root.join("labels").join(split).join(p.file_stem().unwrap()).with_extension("txt");
        let mut boxes = vec![];
        if let Ok(txt) = std::fs::read_to_string(&lp) {
            for line in txt.lines().filter(|l| !l.trim().is_empty()) {
                let v: Vec<f32> = line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                if v.len() == 5 {
                    boxes.push((v[0] as usize, [(v[1] - v[3] / 2.0) * w, (v[2] - v[4] / 2.0) * h, (v[1] + v[3] / 2.0) * w, (v[2] + v[4] / 2.0) * h]));
                }
            }
        }
        out.push(Sample { image: p, boxes });
    }
    Ok(out)
}

#[derive(Serialize, Debug)]
pub struct ClassThreshold {
    pub class: String,
    pub n_cal: usize,
    pub tau: f32,
    pub source: String,
}

/// Nonconformity of a GT object = 1 − best same-class score with IoU ≥ 0.5 (0 if missed).
/// τ_c = 1 − q_c, q_c = k-th smallest, k = confRank(n) ⇒ P(object scored ≥ τ_c) ≥ 1 − 1/30 per class.
pub fn calibrate(det: &mut Detector, samples: &[Sample], fallback: f32) -> Result<Vec<ClassThreshold>> {
    let nc = det.meta.classes.len();
    let floor = vec![det.meta.score_floor; nc];
    let mut ncs: Vec<Vec<f32>> = vec![vec![]; nc];
    for s in samples {
        let dets = det.detect_with(&GrayImage::load(&s.image)?, &floor)?;
        for (c, gb) in &s.boxes {
            let best = dets.iter().filter(|d| d.cls == *c && iou(&d.bbox, gb) >= 0.5).map(|d| d.score).fold(0.0, f32::max);
            ncs[*c].push(1.0 - best);
        }
    }
    Ok((0..nc)
        .map(|c| {
            let v = &mut ncs[c];
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            let (tau, source) = if n >= N_CAL_FORMAL_BOUND {
                let q = 1.0 - v[conf_rank(n) - 1];
                if q >= det.meta.score_floor { (q, "sccp".into()) } else { (det.meta.score_floor, "sccp_floor(coverage not attainable)".into()) }
            } else {
                (fallback, format!("fallback(n={n}<{N_CAL_FORMAL_BOUND})"))
            };
            ClassThreshold { class: det.meta.classes[c].clone(), n_cal: n, tau, source }
        })
        .collect())
}

#[derive(Serialize, Debug)]
pub struct ClassMcc {
    pub class: String,
    pub mcc: f64,
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
    pub tn: u64,
}

/// Frame-level MCC per class: "class predicted present" vs "class labelled present".
pub fn evaluate(det: &mut Detector, samples: &[Sample]) -> Result<Vec<ClassMcc>> {
    let nc = det.meta.classes.len();
    let mut cm = vec![[0u64; 4]; nc];
    for s in samples {
        let dets = det.detect(&GrayImage::load(&s.image)?)?;
        for c in 0..nc {
            let y = s.boxes.iter().any(|(k, _)| *k == c);
            let yh = dets.iter().any(|d| d.cls == c);
            let idx = match (y, yh) { (true, true) => 0, (false, true) => 1, (true, false) => 2, (false, false) => 3 };
            cm[c][idx] += 1;
        }
    }
    Ok((0..nc)
        .map(|c| ClassMcc { class: det.meta.classes[c].clone(), mcc: mcc(cm[c][0], cm[c][1], cm[c][2], cm[c][3]), tp: cm[c][0], fp: cm[c][1], fn_: cm[c][2], tn: cm[c][3] })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::mcc;
    #[test]
    fn mcc_known_values() {
        assert!((mcc(10, 0, 0, 10) - 1.0).abs() < 1e-12);
        assert!((mcc(0, 10, 10, 0) + 1.0).abs() < 1e-12);
        assert_eq!(mcc(0, 0, 0, 10), 0.0);
    }
}
