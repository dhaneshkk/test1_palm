//! Colour "teachers" that label frames for training the monochrome student:
//! * YuNet face detector (opencv_zoo `face_detection_yunet_2023mar.onnx`), decode as in OpenCV's
//!   `FaceDetectorYN::postProcess`;
//! * MediaPipe palm detector (opencv_zoo `palm_detection_mediapipe_2023feb.onnx`), 2016 SSD anchors.
//! Both need colour: on grayscale the palm detector found 5 palms where colour gave 54.

use crate::detector::{init_ort, iou};
use crate::imgproc::RgbImage;
use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

pub const YUNET_URL: &str = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx";
pub const PALM_URL: &str = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx";

fn session(path: &Path, threads: usize) -> Result<Session> {
    init_ort()?;
    Ok(Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))?)
}

/// Greedy NMS over (box, score) pairs, highest score first.
fn nms(mut c: Vec<([f32; 4], f32)>, thr: f32) -> Vec<([f32; 4], f32)> {
    c.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut keep: Vec<([f32; 4], f32)> = vec![];
    for d in c {
        if keep.iter().all(|k| iou(&k.0, &d.0) <= thr) {
            keep.push(d);
        }
    }
    keep
}

pub struct Teachers {
    yunet: Session,
    palm: Session,
    anchors: Vec<(f32, f32)>,
    pub face_thr: f32,
    pub palm_thr: f32,
}

impl Teachers {
    pub fn new(yunet: &Path, palm: &Path, threads: usize) -> Result<Self> {
        let mut anchors = Vec::with_capacity(2016);
        for (n, k) in [(24usize, 2usize), (12, 6)] {
            for y in 0..n {
                for x in 0..n {
                    for _ in 0..k {
                        anchors.push(((x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32));
                    }
                }
            }
        }
        Ok(Self { yunet: session(yunet, threads)?, palm: session(palm, threads)?, anchors, face_thr: 0.7, palm_thr: 0.6 })
    }

    /// Faces as x1,y1,x2,y2 in frame pixels. Frames larger than 640 are downscaled to fit.
    pub fn faces(&mut self, img: &RgbImage) -> Result<Vec<[f32; 4]>> {
        const S: usize = 640;
        let r = (S as f32 / img.w as f32).min(S as f32 / img.h as f32).min(1.0);
        let src = if r < 1.0 { img.resize_linear((img.w as f32 * r) as usize, (img.h as f32 * r) as usize) } else { img.clone() };
        // NCHW, BGR order, raw 0..255, zero padding bottom/right (as blobFromImage on the padded frame)
        let mut t = vec![0f32; 3 * S * S];
        for y in 0..src.h {
            for x in 0..src.w {
                let p = &src.data[(y * src.w + x) * 3..][..3];
                for (c, v) in [p[2], p[1], p[0]].into_iter().enumerate() {
                    t[c * S * S + y * S + x] = v as f32;
                }
            }
        }
        let out = self.yunet.run(ort::inputs!["input" => Tensor::from_array(([1usize, 3, S, S], t.into_boxed_slice()))?])?;
        let mut cands = vec![];
        for st in [8usize, 16, 32] {
            let (_, cls) = out[format!("cls_{st}").as_str()].try_extract_tensor::<f32>()?;
            let (_, obj) = out[format!("obj_{st}").as_str()].try_extract_tensor::<f32>()?;
            let (_, bb) = out[format!("bbox_{st}").as_str()].try_extract_tensor::<f32>()?;
            let cols = S / st;
            for i in 0..cls.len() {
                let score = (cls[i].clamp(0.0, 1.0) * obj[i].clamp(0.0, 1.0)).sqrt();
                if score < self.face_thr {
                    continue;
                }
                let (row, col) = ((i / cols) as f32, (i % cols) as f32);
                let (cx, cy) = ((col + bb[i * 4]) * st as f32, (row + bb[i * 4 + 1]) * st as f32);
                let (w, h) = (bb[i * 4 + 2].exp() * st as f32, bb[i * 4 + 3].exp() * st as f32);
                cands.push(([(cx - w / 2.0) / r, (cy - h / 2.0) / r, (cx + w / 2.0) / r, (cy + h / 2.0) / r], score));
            }
        }
        Ok(nms(cands, 0.3).into_iter().map(|(b, _)| b).collect())
    }

    /// Palms as x1,y1,x2,y2 in frame pixels (letterbox centred in 192×192, RGB/255, NHWC).
    pub fn palms(&mut self, img: &RgbImage) -> Result<Vec<[f32; 4]>> {
        const S: usize = 192;
        let r = S as f32 / img.w.max(img.h) as f32;
        let (nw, nh) = ((img.w as f32 * r) as usize, (img.h as f32 * r) as usize);
        let rs = img.resize_linear(nw, nh);
        let (top, left) = ((S - nh) / 2, (S - nw) / 2);
        let mut t = vec![0f32; S * S * 3];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..3 {
                    t[((y + top) * S + x + left) * 3 + c] = rs.data[(y * nw + x) * 3 + c] as f32 / 255.0;
                }
            }
        }
        let out = self.palm.run(ort::inputs!["input_1" => Tensor::from_array(([1usize, S, S, 3], t.into_boxed_slice()))?])?;
        let (_, reg) = out["Identity"].try_extract_tensor::<f32>()?;
        let (_, sc) = out["Identity_1"].try_extract_tensor::<f32>()?;
        let mut cands = vec![];
        for (i, &(ax, ay)) in self.anchors.iter().enumerate() {
            let score = 1.0 / (1.0 + (-(sc[i] as f64)).exp()) as f32;
            if score < self.palm_thr {
                continue;
            }
            let row = &reg[i * 18..i * 18 + 4];
            let (cx, cy) = (row[0] / S as f32 + ax, row[1] / S as f32 + ay);
            let (w, h) = (row[2] / S as f32, row[3] / S as f32);
            let b = [(cx - w / 2.0) * S as f32, (cy - h / 2.0) * S as f32, (cx + w / 2.0) * S as f32, (cy + h / 2.0) * S as f32];
            let b = [
                ((b[0] - left as f32) / r).clamp(0.0, img.w as f32 - 1.0),
                ((b[1] - top as f32) / r).clamp(0.0, img.h as f32 - 1.0),
                ((b[2] - left as f32) / r).clamp(0.0, img.w as f32 - 1.0),
                ((b[3] - top as f32) / r).clamp(0.0, img.h as f32 - 1.0),
            ];
            cands.push((b, score));
        }
        Ok(nms(cands, 0.3).into_iter().map(|(b, _)| b).collect())
    }
}
