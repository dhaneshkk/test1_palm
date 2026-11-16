//! 21-point hand landmarks (MediaPipe hand-landmark model, opencv_zoo
//! `handpose_estimation_mediapipe_2023feb.onnx`, Apache-2.0) on the grayscale frame, and a palm ROI
//! aligned to those landmarks.
//!
//! The palm box comes from the monochrome student detector, which has no palm keypoints, so the
//! hand is assumed upright (no rotation step): box shifted up by 0.4·h, enlarged 3×, padded to a
//! square, resized to 224 — as opencv_zoo's `mp_handpose.py` does after its rotation.
//! Measured on 54 hands: mean confidence 0.93 on grayscale vs 0.97 on colour; landmarks within ~5 %
//! of palm size of the colour result.

use crate::detector::init_ort;
use crate::imgproc::{resize_area, GrayImage};
use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use serde::Serialize;
use std::path::Path;

pub const HANDPOSE_URL: &str = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx";
const S: usize = 224;

/// MediaPipe `HAND_CONNECTIONS`: wrist 0; thumb 1–4; index 5–8; middle 9–12; ring 13–16; pinky 17–20.
pub const CONNECTIONS: [(usize, usize); 21] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
];

#[derive(Clone, Debug, Serialize)]
pub struct HandPose {
    /// (x, y, z) in frame pixels; z is relative depth (same scale as x, y)
    pub points: Vec<[f32; 3]>,
    pub confidence: f32,
    /// model's handedness score (≥ 0.5 ≈ right hand as seen by the camera)
    pub handedness: f32,
}

pub struct HandLandmarker {
    session: Session,
    pub min_confidence: f32,
}

impl HandLandmarker {
    pub fn new(onnx: &Path, threads: usize) -> Result<Self> {
        init_ort()?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(onnx)
            .with_context(|| format!("loading {}", onnx.display()))?;
        Ok(Self { session, min_confidence: 0.8 })
    }

    /// Landmarks for the hand around a palm box (x1,y1,x2,y2). `None` if below `min_confidence`.
    pub fn landmarks(&mut self, gray: &GrayImage, palm: [f32; 4]) -> Result<Option<HandPose>> {
        let (w, h) = (palm[2] - palm[0], palm[3] - palm[1]);
        let (y1, y2) = (palm[1] - 0.4 * h, palm[3] - 0.4 * h);
        let (cx, cy) = ((palm[0] + palm[2]) / 2.0, (y1 + y2) / 2.0);
        let (hw, hh) = (w * 3.0 / 2.0, h * 3.0 / 2.0);
        // numpy astype(int32) truncates toward zero, then clip to the frame
        let clip = |v: f32, hi: usize| (v as i64).clamp(0, hi as i64) as usize;
        let (bx0, by0, bx1, by1) = (clip(cx - hw, gray.w), clip(cy - hh, gray.h), clip(cx + hw, gray.w), clip(cy + hh, gray.h));
        if bx1 <= bx0 + 4 || by1 <= by0 + 4 {
            return Ok(None);
        }
        let crop = gray.crop(bx0, by0, bx1, by1);
        let side = crop.w.max(crop.h);
        let (left, top) = ((side - crop.w) / 2, (side - crop.h) / 2);
        let mut sq = GrayImage::new(side, side, 0);
        for y in 0..crop.h {
            sq.data[(y + top) * side + left..(y + top) * side + left + crop.w].copy_from_slice(&crop.data[y * crop.w..(y + 1) * crop.w]);
        }
        let small = resize_area(&sq, S, S);
        let mut t = Vec::with_capacity(S * S * 3);
        for &v in &small.data {
            let f = v as f32 / 255.0;
            t.extend_from_slice(&[f, f, f]); // gray replicated to RGB, NHWC
        }
        let out = self.session.run(ort::inputs!["input_1" => Tensor::from_array(([1usize, S, S, 3], t.into_boxed_slice()))?])?;
        let (_, lm) = out["Identity"].try_extract_tensor::<f32>()?;
        let (_, conf) = out["Identity_1"].try_extract_tensor::<f32>()?;
        let (_, hand) = out["Identity_2"].try_extract_tensor::<f32>()?;
        let confidence = conf[0];
        if confidence < self.min_confidence {
            return Ok(None);
        }
        let scale = (bx1 - bx0).max(by1 - by0) as f32 / S as f32;
        let (ccx, ccy) = ((bx0 + bx1) as f32 / 2.0, (by0 + by1) as f32 / 2.0);
        let points = (0..21)
            .map(|i| [(lm[i * 3] - S as f32 / 2.0) * scale + ccx, (lm[i * 3 + 1] - S as f32 / 2.0) * scale + ccy, lm[i * 3 + 2] * scale])
            .collect();
        Ok(Some(HandPose { points, confidence, handedness: hand[0] }))
    }
}

/// Square palm ROI aligned to the hand: centred 55 % of the way from the wrist (0) to the middle-finger
/// knuckle (9), its "down" axis pointing from the knuckles to the wrist, side = 0.85 × the index–pinky
/// knuckle distance (5–17). Bilinear sampling; `None` if the palm is too small or leaves the frame.
pub fn aligned_palm_roi(gray: &GrayImage, pose: &HandPose, out_side: usize, min_side_px: f32) -> Option<GrayImage> {
    let p = |i: usize| (pose.points[i][0], pose.points[i][1]);
    let (p0, p5, p9, p17) = (p(0), p(5), p(9), p(17));
    let (ax, ay) = (p9.0 - p0.0, p9.1 - p0.1);
    let len = (ax * ax + ay * ay).sqrt();
    let width = ((p5.0 - p17.0).powi(2) + (p5.1 - p17.1).powi(2)).sqrt();
    let side = 0.85 * width;
    if len < 1.0 || side < min_side_px {
        return None;
    }
    let (cx, cy) = (p0.0 + 0.55 * ax, p0.1 + 0.55 * ay);
    let (dy_x, dy_y) = (-ax / len, -ay / len); // "down": knuckles → wrist
    let (dx_x, dx_y) = (dy_y, -dy_x); // "right", perpendicular
    let half = side / 2.0;
    for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let x = cx + sx * half * dx_x + sy * half * dy_x;
        let y = cy + sx * half * dx_y + sy * half * dy_y;
        if x < 0.0 || y < 0.0 || x > (gray.w - 1) as f32 || y > (gray.h - 1) as f32 {
            return None;
        }
    }
    let mut out = GrayImage::new(out_side, out_side, 0);
    for v in 0..out_side {
        for u in 0..out_side {
            let a = ((u as f32 + 0.5) / out_side as f32 - 0.5) * side;
            let b = ((v as f32 + 0.5) / out_side as f32 - 0.5) * side;
            let x = (cx + a * dx_x + b * dy_x).clamp(0.0, (gray.w - 1) as f32);
            let y = (cy + a * dx_y + b * dy_y).clamp(0.0, (gray.h - 1) as f32);
            let (x0, y0) = (x.floor() as usize, y.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(gray.w - 1), (y0 + 1).min(gray.h - 1));
            let (fx, fy) = (x - x0 as f32, y - y0 as f32);
            let g = |xx: usize, yy: usize| gray.data[yy * gray.w + xx] as f32;
            let val = (g(x0, y0) * (1.0 - fx) + g(x1, y0) * fx) * (1.0 - fy) + (g(x0, y1) * (1.0 - fx) + g(x1, y1) * fx) * fy;
            out.data[v * out_side + u] = val.round().clamp(0.0, 255.0) as u8;
        }
    }
    Some(out)
}

/// Draw the skeleton (bones grey `bone`, joints `joint`) onto a grayscale frame.
pub fn draw_skeleton(img: &mut GrayImage, pose: &HandPose, bone: u8, joint: u8) {
    for &(a, b) in CONNECTIONS.iter() {
        let (pa, pb) = (pose.points[a], pose.points[b]);
        img.draw_line(pa[0], pa[1], pb[0], pb[1], bone);
    }
    for pt in &pose.points {
        img.draw_dot(pt[0], pt[1], 2, joint);
    }
}
