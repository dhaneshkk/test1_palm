//! Palmprint features: palm ROI → Competitive Code (Kong & Zhang 2004) → shift-tolerant angular distance.
//! A laptop webcam resolves palm principal lines and creases, not friction ridges (~500 dpi needed).

use crate::imgproc::{clahe, filter2d, gabor_kernel, gaussian_blur, resize_area, FloatImage, GrayImage};
use crate::lean_mirror::{ang_dist, resolution_gate};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PalmParams {
    pub roi: usize,
    pub code_stride: usize,
    pub gabor_ksize: usize,
    pub gabor_sigma: f64,
    pub gabor_lambda: f64,
    pub shift: isize,
    pub min_palm_side: f32,
}

impl Default for PalmParams {
    fn default() -> Self {
        Self { roi: 128, code_stride: 4, gabor_ksize: 17, gabor_sigma: 3.5, gabor_lambda: 9.0, shift: 2, min_palm_side: 96.0 }
    }
}

/// Orientation code grid: each cell ∈ {0..5} (6 Gabor orientations, 30° apart).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Code {
    pub h: usize,
    pub w: usize,
    pub cells: Vec<u8>,
}

pub struct PalmCoder {
    pub p: PalmParams,
    bank: Vec<FloatImage>,
}

impl PalmCoder {
    pub fn new(p: PalmParams) -> Self {
        let bank = (0..6)
            .map(|j| {
                let mut k = gabor_kernel(p.gabor_ksize, p.gabor_sigma, j as f64 * std::f64::consts::PI / 6.0, p.gabor_lambda, 0.7, 0.0);
                let mean = k.data.iter().map(|&v| v as f64).sum::<f64>() / k.data.len() as f64;
                k.data.iter_mut().for_each(|v| *v -= mean as f32); // zero-DC: illumination invariant
                k
            })
            .collect();
        Self { p, bank }
    }

    /// Central 70 % square of the palm box, CLAHE, resized to roi×roi. `None` below the resolution gate
    /// (Lean §5) or when the square leaves the frame.
    pub fn roi(&self, gray: &GrayImage, b: [f32; 4]) -> Option<GrayImage> {
        let side = (b[2] - b[0]).min(b[3] - b[1]);
        if !resolution_gate(side, self.p.min_palm_side) {
            return None;
        }
        let (cx, cy, half) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, 0.35 * side);
        let (x0, y0, x1, y1) = ((cx - half) as isize, (cy - half) as isize, (cx + half) as isize, (cy + half) as isize);
        if x0 < 0 || y0 < 0 || x1 as usize > gray.w || y1 as usize > gray.h || x1 <= x0 || y1 <= y0 {
            return None;
        }
        let crop = gray.crop(x0 as usize, y0 as usize, x1 as usize, y1 as usize);
        Some(resize_area(&clahe(&crop, 2.0, 8, 8), self.p.roi, self.p.roi))
    }

    /// Competitive code: per pixel, orientation of the most negative (dark-line) Gabor response.
    pub fn code(&self, roi: &GrayImage) -> Code {
        let blurred = gaussian_blur(roi, 1.0);
        let f = FloatImage { w: roi.w, h: roi.h, data: blurred.data.iter().map(|&v| v as f32).collect() };
        let resp: Vec<FloatImage> = self.bank.iter().map(|k| filter2d(&f, k)).collect();
        let m = self.p.gabor_ksize / 2;
        let st = self.p.code_stride;
        let ys: Vec<usize> = (m..roi.h - m).step_by(st).collect();
        let xs: Vec<usize> = (m..roi.w - m).step_by(st).collect();
        let mut cells = Vec::with_capacity(ys.len() * xs.len());
        for &y in &ys {
            for &x in &xs {
                let (mut best, mut arg) = (f32::INFINITY, 0u8);
                for (j, r) in resp.iter().enumerate() {
                    let v = r.data[y * roi.w + x];
                    if v < best {
                        best = v;
                        arg = j as u8; // first minimum, as numpy argmin
                    }
                }
                cells.push(arg);
            }
        }
        Code { h: ys.len(), w: xs.len(), cells }
    }

    /// Normalised angular distance in [0,1] (Lean §1: per-cell ≤ 3 ⇒ total ≤ 3n),
    /// minimised over ±shift translations of the code grid.
    pub fn distance(&self, a: &Code, b: &Code) -> f64 {
        code_distance(a, b, self.p.shift)
    }
}

pub fn code_distance(a: &Code, b: &Code, shift: isize) -> f64 {
    assert_eq!((a.h, a.w), (b.h, b.w), "code grids differ");
    let (h, w) = (a.h as isize, a.w as isize);
    let mut best = 1.0f64;
    for dy in -shift..=shift {
        for dx in -shift..=shift {
            let (ay0, ay1) = (dy.max(0), h + dy.min(0));
            let (ax0, ax1) = (dx.max(0), w + dx.min(0));
            let mut total = 0u64;
            let mut n = 0u64;
            for y in ay0..ay1 {
                for x in ax0..ax1 {
                    let av = a.cells[(y * w + x) as usize];
                    let bv = b.cells[((y - dy) * w + (x - dx)) as usize];
                    total += ang_dist(av, bv) as u64;
                    n += 1;
                }
            }
            if n > 0 {
                best = best.min(total as f64 / (3.0 * n as f64));
            }
        }
    }
    best
}
