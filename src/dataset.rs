//! Teacher-labelled dataset writer (YOLO txt format, grayscale PNGs).
//! Splits by interleaved time blocks (10 frames each: 6 train, 2 cal, 2 val) so every split
//! sees the whole session — a plain time split left the val set with no palms in practice.

use crate::imgproc::GrayImage;
use anyhow::Result;
use std::path::{Path, PathBuf};

pub const BLOCK: usize = 10;

pub fn split_for(index: usize) -> &'static str {
    match (index / BLOCK) % 5 {
        0 | 1 | 2 => "train",
        3 => "cal",
        _ => "val",
    }
}

pub struct DatasetWriter {
    pub root: PathBuf,
    pub count: usize,
    pub per_class: Vec<usize>,
}

impl DatasetWriter {
    pub fn create(root: &Path, n_classes: usize) -> Result<Self> {
        if root.exists() {
            std::fs::remove_dir_all(root)?;
        }
        for s in ["train", "cal", "val"] {
            std::fs::create_dir_all(root.join("images").join(s))?;
            std::fs::create_dir_all(root.join("labels").join(s))?;
        }
        Ok(Self { root: root.to_path_buf(), count: 0, per_class: vec![0; n_classes] })
    }

    /// `boxes`: (class, x1,y1,x2,y2) in pixels of `gray`.
    pub fn add(&mut self, gray: &GrayImage, boxes: &[(usize, [f32; 4])]) -> Result<&'static str> {
        let split = split_for(self.count);
        let stem = format!("{:05}", self.count);
        image::save_buffer(self.root.join("images").join(split).join(format!("{stem}.png")), &gray.data, gray.w as u32, gray.h as u32, image::ExtendedColorType::L8)?;
        let (w, h) = (gray.w as f32, gray.h as f32);
        let mut txt = String::new();
        for (c, b) in boxes {
            txt.push_str(&format!("{c} {:.6} {:.6} {:.6} {:.6}\n", (b[0] + b[2]) / 2.0 / w, (b[1] + b[3]) / 2.0 / h, (b[2] - b[0]) / w, (b[3] - b[1]) / h));
            self.per_class[*c] += 1;
        }
        std::fs::write(self.root.join("labels").join(split).join(format!("{stem}.txt")), txt)?;
        self.count += 1;
        Ok(split)
    }
}

#[cfg(test)]
mod tests {
    use super::split_for;
    #[test]
    fn interleaved_split_ratio() {
        let n = 300;
        let c = |s: &str| (0..n).filter(|&i| split_for(i) == s).count();
        assert_eq!((c("train"), c("cal"), c("val")), (180, 60, 60));
        // every 50-frame window contributes to all three splits
        for w in 0..n / 50 {
            let in_w = |s: &str| (w * 50..(w + 1) * 50).any(|i| split_for(i) == s);
            assert!(in_w("train") && in_w("cal") && in_w("val"));
        }
    }
}
