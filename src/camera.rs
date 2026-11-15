//! Frame sources with no OpenCV dependency: webcam/file/RTSP decoded by an `ffmpeg` subprocess
//! straight to 8-bit grayscale (`-pix_fmt gray`), or a directory of images.

use crate::imgproc::{GrayImage, RgbImage};
use anyhow::{anyhow, Context, Result};
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, ChildStdout, Command, Stdio};

pub enum Source {
    Ffmpeg { child: Child, out: ChildStdout, w: usize, h: usize, rgb: bool },
    Images { files: Vec<PathBuf>, i: usize },
}

impl Source {
    /// `spec`: `webcam` / `webcam:<device>` (platform default input), a video file / URL, or an image directory.
    pub fn open(spec: &str, w: usize, h: usize) -> Result<Self> {
        Self::open_with(spec, w, h, false)
    }

    /// Colour (RGB24) frames — needed by the teachers during data collection.
    pub fn open_rgb(spec: &str, w: usize, h: usize) -> Result<Self> {
        Self::open_with(spec, w, h, true)
    }

    fn open_with(spec: &str, w: usize, h: usize, rgb: bool) -> Result<Self> {
        let p = PathBuf::from(spec);
        if p.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| matches!(f.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(), Some("png" | "jpg" | "jpeg")))
                .collect();
            files.sort();
            return Ok(Source::Images { files, i: 0 });
        }
        let mut args: Vec<String> = vec!["-hide_banner".into(), "-loglevel".into(), "error".into()];
        if let Some(rest) = spec.strip_prefix("webcam") {
            let dev = rest.strip_prefix(':').filter(|d| !d.is_empty());
            let size = format!("{w}x{h}");
            if cfg!(target_os = "linux") {
                args.extend(["-f", "v4l2", "-video_size", &size, "-i", dev.unwrap_or("/dev/video0")].map(String::from));
            } else if cfg!(target_os = "macos") {
                args.extend(["-f", "avfoundation", "-framerate", "30", "-video_size", &size, "-i", dev.unwrap_or("0")].map(String::from));
            } else {
                let d = format!("video={}", dev.unwrap_or("Integrated Camera"));
                args.extend(["-f".to_string(), "dshow".into(), "-video_size".into(), size, "-i".into(), d]);
            }
        } else {
            args.extend(["-stream_loop", "-1", "-i", spec].map(String::from));
        }
        args.extend(["-vf", &format!("scale={w}:{h}"), "-f", "rawvideo", "-pix_fmt", if rgb { "rgb24" } else { "gray" }, "-"].map(String::from));
        let mut child = Command::new("ffmpeg").args(&args).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().context("spawning ffmpeg (is it installed?)")?;
        let out = child.stdout.take().ok_or_else(|| anyhow!("no ffmpeg stdout"))?;
        Ok(Source::Ffmpeg { child, out, w, h, rgb })
    }

    /// Next frame as grayscale.
    pub fn read(&mut self) -> Result<Option<GrayImage>> {
        Ok(match self {
            Source::Ffmpeg { rgb: false, out, w, h, .. } => {
                let mut buf = vec![0u8; *w * *h];
                read_full(out, &mut buf)?.then(|| GrayImage { w: *w, h: *h, data: buf })
            }
            _ => self.read_rgb()?.map(|f| f.to_gray()),
        })
    }

    /// Next frame as RGB (an ffmpeg source must have been opened with `open_rgb`).
    pub fn read_rgb(&mut self) -> Result<Option<RgbImage>> {
        match self {
            Source::Ffmpeg { rgb, out, w, h, .. } => {
                if !*rgb {
                    return Err(anyhow!("source opened in gray mode; use Source::open_rgb"));
                }
                let mut buf = vec![0u8; *w * *h * 3];
                Ok(read_full(out, &mut buf)?.then(|| RgbImage { w: *w, h: *h, data: buf }))
            }
            Source::Images { files, i } => {
                if *i >= files.len() {
                    return Ok(None);
                }
                *i += 1;
                Ok(Some(RgbImage::load(&files[*i - 1])?))
            }
        }
    }
}

fn read_full(out: &mut ChildStdout, buf: &mut [u8]) -> Result<bool> {
    match out.read_exact(buf) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e.into()),
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        if let Source::Ffmpeg { child, .. } = self {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
