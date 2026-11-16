//! Dense face mesh on the grayscale frame: MediaPipe's face-landmark model (468 points; from the
//! Apache-2.0 `mediapipe` 0.10.14 wheel, converted TFLite → ONNX with tf2onnx — see
//! `assets/models/NOTICE-face_landmark.md`), plus a Delaunay triangulation of the points for drawing.
//!
//! Crop: square of side ⌊1.5 · max(w, h)⌋ centred on the face box, zero-padded, OpenCV-exact
//! INTER_LINEAR to 192×192, gray replicated to RGB, values in [0, 1]. Faces are assumed upright
//! (the face box carries no eye keypoints). On 16 face crops: presence 1.00 on grayscale, landmarks
//! within 1.4 % of the inter-ocular distance of the colour result.

use crate::detector::init_ort;
use crate::imgproc::{resize_linear, GrayImage};
use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use serde::Serialize;
use std::path::Path;

const S: usize = 192;

/// The converted model, embedded so the binaries need no download (2.4 MB, Apache-2.0).
pub const FACEMESH_ONNX: &[u8] = include_bytes!("../assets/models/face_landmark.onnx");

#[derive(Clone, Debug, Serialize)]
pub struct FaceMesh {
    /// 468 × (x, y, z) in frame pixels (z: relative depth, same scale)
    pub points: Vec<[f32; 3]>,
    /// face-presence probability (sigmoid of the model's flag output)
    pub presence: f32,
}

pub struct FaceMesher {
    session: Session,
    pub min_presence: f32,
}

impl FaceMesher {
    pub fn new(onnx: &Path, threads: usize) -> Result<Self> {
        Self::from_bytes(&std::fs::read(onnx).with_context(|| format!("reading {}", onnx.display()))?, threads)
    }

    /// Load the embedded model (or `PALMID_FACEMESH=<file.onnx>` when set).
    pub fn embedded(threads: usize) -> Result<Self> {
        match std::env::var_os("PALMID_FACEMESH") {
            Some(p) => Self::new(Path::new(&p), threads),
            None => Self::from_bytes(FACEMESH_ONNX, threads),
        }
    }

    pub fn from_bytes(bytes: &[u8], threads: usize) -> Result<Self> {
        init_ort()?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_memory(bytes)
            .context("loading face-landmark model")?;
        Ok(Self { session, min_presence: 0.5 })
    }

    pub fn mesh(&mut self, gray: &GrayImage, face: [f32; 4]) -> Result<Option<FaceMesh>> {
        let side = (1.5 * (face[2] - face[0]).max(face[3] - face[1])).floor() as i64;
        if side < 8 {
            return Ok(None);
        }
        let x0 = ((face[0] + face[2]) / 2.0 - side as f32 / 2.0).floor() as i64;
        let y0 = ((face[1] + face[3]) / 2.0 - side as f32 / 2.0).floor() as i64;
        let s = side as usize;
        let mut sq = GrayImage::new(s, s, 0);
        let (sx0, sy0) = (x0.max(0), y0.max(0));
        let (sx1, sy1) = ((x0 + side).min(gray.w as i64), (y0 + side).min(gray.h as i64));
        if sx1 <= sx0 || sy1 <= sy0 {
            return Ok(None);
        }
        for y in sy0..sy1 {
            let src = &gray.data[y as usize * gray.w + sx0 as usize..y as usize * gray.w + sx1 as usize];
            let d = (y - y0) as usize * s + (sx0 - x0) as usize;
            sq.data[d..d + src.len()].copy_from_slice(src);
        }
        let crop = resize_linear(&sq, S, S);
        let mut t = Vec::with_capacity(S * S * 3);
        for &v in &crop.data {
            let f = v as f32 / 255.0;
            t.extend_from_slice(&[f, f, f]);
        }
        let out = self.session.run(ort::inputs!["input_1" => Tensor::from_array(([1usize, S, S, 3], t.into_boxed_slice()))?])?;
        let (_, lm) = out["conv2d_21"].try_extract_tensor::<f32>()?;
        let (_, flag) = out["conv2d_31"].try_extract_tensor::<f32>()?;
        let presence = 1.0 / (1.0 + (-flag[0]).exp());
        if presence < self.min_presence {
            return Ok(None);
        }
        let k = side as f32 / S as f32;
        let points = (0..468).map(|i| [lm[i * 3] * k + x0 as f32, lm[i * 3 + 1] * k + y0 as f32, lm[i * 3 + 2] * k]).collect();
        Ok(Some(FaceMesh { points, presence }))
    }
}

/// Delaunay triangulation (Bowyer–Watson) of 2-D points; returns index triples.
pub fn delaunay(pts: &[(f64, f64)]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 {
        return vec![];
    }
    let (mut minx, mut miny, mut maxx, mut maxy) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for &(x, y) in pts {
        minx = minx.min(x);
        miny = miny.min(y);
        maxx = maxx.max(x);
        maxy = maxy.max(y);
    }
    let d = (maxx - minx).max(maxy - miny).max(1.0) * 20.0;
    let (mx, my) = ((minx + maxx) / 2.0, (miny + maxy) / 2.0);
    let mut v: Vec<(f64, f64)> = pts.to_vec();
    v.push((mx - d, my - d));
    v.push((mx + d, my - d));
    v.push((mx, my + d));
    let circum = |a: usize, b: usize, c: usize, v: &[(f64, f64)]| -> (f64, f64, f64) {
        let ((ax, ay), (bx, by), (cx, cy)) = (v[a], v[b], v[c]);
        let dd = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if dd.abs() < 1e-12 {
            return (0.0, 0.0, f64::INFINITY);
        }
        let (a2, b2, c2) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
        let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / dd;
        let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / dd;
        (ux, uy, (ax - ux).powi(2) + (ay - uy).powi(2))
    };
    let mut tris: Vec<([usize; 3], (f64, f64, f64))> = vec![([n, n + 1, n + 2], circum(n, n + 1, n + 2, &v))];
    for p in 0..n {
        let (px, py) = v[p];
        let mut bad = vec![];
        for (i, (_, (ux, uy, r2))) in tris.iter().enumerate() {
            if (px - ux).powi(2) + (py - uy).powi(2) < *r2 * (1.0 + 1e-12) {
                bad.push(i);
            }
        }
        // boundary of the cavity: edges belonging to exactly one bad triangle
        let mut edges: Vec<(usize, usize)> = vec![];
        for &i in &bad {
            let t = tris[i].0;
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = (a.min(b), a.max(b));
                if let Some(k) = edges.iter().position(|e| *e == key) {
                    edges.swap_remove(k);
                } else {
                    edges.push(key);
                }
            }
        }
        for &i in bad.iter().rev() {
            tris.swap_remove(i);
        }
        for (a, b) in edges {
            tris.push(([a, b, p], circum(a, b, p, &v)));
        }
    }
    let mut out: Vec<[usize; 3]> = tris.into_iter().map(|(t, _)| t).filter(|t| t.iter().all(|&k| k < n)).map(|mut t| { t.sort(); t }).collect();
    out.sort();
    out
}

/// Draw the triangulated mesh (edges `edge`, points `point`).
pub fn draw_mesh(img: &mut GrayImage, mesh: &FaceMesh, edge: u8, point: u8) {
    let pts: Vec<(f64, f64)> = mesh.points.iter().map(|p| (p[0] as f64, p[1] as f64)).collect();
    let mut drawn = std::collections::HashSet::new();
    for t in delaunay(&pts) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])] {
            if drawn.insert((a, b)) {
                img.draw_line(pts[a].0 as f32, pts[a].1 as f32, pts[b].0 as f32, pts[b].1 as f32, edge);
            }
        }
    }
    for p in &mesh.points {
        img.draw_dot(p[0], p[1], 0, point);
    }
}
