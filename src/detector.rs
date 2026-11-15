//! Monochrome YOLOX-Nano student (INT8 QDQ ONNX exported by the notebook) + numpy-identical decode.
//! The exported graph has `decode_in_inference=False`, so the raw output is
//! `[1, N, 5+C]` = (reg dx, dy, log w, log h, obj·σ, cls·σ …) and grids/strides are applied here.

use crate::imgproc::{letterbox, GrayImage};
use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Sidecar written by the notebook next to the ONNX (`palmid_model.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMeta {
    pub classes: Vec<String>,
    pub input_size: usize,
    /// Per-class thresholds from Mondrian split-conformal calibration (or the fallback).
    pub tau: Vec<f32>,
    #[serde(default = "d_floor")]
    pub score_floor: f32,
    #[serde(default = "d_nms")]
    pub nms_iou: f32,
}
fn d_floor() -> f32 { 0.05 }
fn d_nms() -> f32 { 0.45 }

impl ModelMeta {
    pub fn default_face_palm() -> Self {
        Self { classes: vec!["face".into(), "palm".into()], input_size: 416, tau: vec![0.3, 0.3], score_floor: 0.05, nms_iou: 0.45 }
    }
    pub fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?)?)
    }
    pub fn class_id(&self, name: &str) -> Option<usize> {
        self.classes.iter().position(|c| c == name)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub cls: usize,
    pub score: f32,
    /// x1, y1, x2, y2 in original image pixels
    pub bbox: [f32; 4],
}

pub struct Detector {
    session: Session,
    pub meta: ModelMeta,
    grids: Vec<(f32, f32, f32)>, // (gx, gy, stride)
}

/// Point `ort` at libonnxruntime (load-dynamic). Honours ORT_DYLIB_PATH; otherwise tries common locations.
pub fn init_ort() -> Result<()> {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        let candidates = [
            "/usr/lib/libonnxruntime.so",
            "/usr/local/lib/libonnxruntime.so",
            "/opt/homebrew/lib/libonnxruntime.dylib",
            "/usr/local/lib/libonnxruntime.dylib",
        ];
        match candidates.iter().find(|p| Path::new(p).exists()) {
            Some(p) => std::env::set_var("ORT_DYLIB_PATH", p),
            None => return Err(anyhow!("set ORT_DYLIB_PATH to libonnxruntime (e.g. from `pip show onnxruntime` → capi/libonnxruntime.so.*)")),
        }
    }
    Ok(())
}

impl Detector {
    pub fn new(onnx: &Path, meta: ModelMeta, threads: usize) -> Result<Self> {
        init_ort()?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(onnx)
            .with_context(|| format!("loading {}", onnx.display()))?;
        let s = meta.input_size;
        let mut grids = Vec::new();
        for st in [8usize, 16, 32] {
            let n = s / st;
            for y in 0..n {
                for x in 0..n {
                    grids.push((x as f32, y as f32, st as f32));
                }
            }
        }
        Ok(Self { session, meta, grids })
    }

    /// Raw head output `[N, 5+C]` for one gray frame plus the letterbox ratio.
    pub fn raw(&mut self, gray: &GrayImage) -> Result<(Vec<f32>, usize, f32)> {
        let (chw, r) = letterbox(gray, self.meta.input_size);
        let (raw, stride) = self.run_tensor(chw)?;
        Ok((raw, stride, r))
    }

    pub fn grids(&self) -> &[(f32, f32, f32)] {
        &self.grids
    }

    /// Run the network on an already letterboxed CHW tensor (3·S·S, values 0..255).
    pub fn run_tensor(&mut self, chw: Vec<f32>) -> Result<(Vec<f32>, usize)> {
        let s = self.meta.input_size;
        let input = Tensor::from_array(([1usize, 3, s, s], chw.into_boxed_slice()))?;
        let outputs = self.session.run(ort::inputs!["images" => input])?;
        let (shape, data) = outputs["output"].try_extract_tensor::<f32>()?;
        let stride = *shape.last().ok_or_else(|| anyhow!("bad output shape"))? as usize;
        if stride != 5 + self.meta.classes.len() {
            return Err(anyhow!("model has {} classes, meta lists {}", stride - 5, self.meta.classes.len()));
        }
        Ok((data.to_vec(), stride))
    }

    pub fn detect_with(&mut self, gray: &GrayImage, tau: &[f32]) -> Result<Vec<Detection>> {
        let (raw, stride, r) = self.raw(gray)?;
        Ok(decode(&raw, stride, &self.grids, r, tau, self.meta.nms_iou))
    }

    pub fn detect(&mut self, gray: &GrayImage) -> Result<Vec<Detection>> {
        let tau = self.meta.tau.clone();
        self.detect_with(gray, &tau)
    }
}

pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if ua <= 0.0 { 0.0 } else { inter / ua }
}

/// Notebook `decode`: grid/stride decode → per-class τ filter → class-wise greedy NMS.
pub fn decode(raw: &[f32], stride: usize, grids: &[(f32, f32, f32)], r: f32, tau: &[f32], nms_iou: f32) -> Vec<Detection> {
    let nc = stride - 5;
    let mut cands: Vec<Vec<Detection>> = vec![Vec::new(); nc];
    for (i, &(gx, gy, st)) in grids.iter().enumerate() {
        let row = &raw[i * stride..(i + 1) * stride];
        let (mut cls, mut best) = (0usize, f32::MIN);
        for c in 0..nc {
            if row[5 + c] > best {
                best = row[5 + c];
                cls = c;
            }
        }
        let score = row[4] * best;
        if score < tau[cls] {
            continue;
        }
        let cx = (row[0] + gx) * st;
        let cy = (row[1] + gy) * st;
        let w = row[2].clamp(-10.0, 10.0).exp() * st;
        let h = row[3].clamp(-10.0, 10.0).exp() * st;
        cands[cls].push(Detection { cls, score, bbox: [(cx - w / 2.0) / r, (cy - h / 2.0) / r, (cx + w / 2.0) / r, (cy + h / 2.0) / r] });
    }
    let mut out = Vec::new();
    for mut v in cands {
        v.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let mut kept: Vec<Detection> = Vec::new();
        for d in v {
            if kept.iter().all(|k| iou(&k.bbox, &d.bbox) <= nms_iou) {
                kept.push(d);
            }
        }
        out.extend(kept);
    }
    out
}
