//! Frame sources with no OpenCV dependency: webcam/file/RTSP decoded by an `ffmpeg` subprocess
//! straight to 8-bit grayscale (`-pix_fmt gray`), or a directory of images.

use crate::imgproc::GrayImage;
use anyhow::{anyhow, Context, Result};
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, ChildStdout, Command, Stdio};

pub enum Source {
    Ffmpeg { child: Child, out: ChildStdout, w: usize, h: usize },
    Images { files: Vec<PathBuf>, i: usize },
}

impl Source {
    /// `spec`: `webcam` / `webcam:<device>` (platform default input), a video file / URL, or an image directory.
    pub fn open(spec: &str, w: usize, h: usize) -> Result<Self> {
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
        args.extend(["-vf", &format!("scale={w}:{h}"), "-f", "rawvideo", "-pix_fmt", "gray", "-"].map(String::from));
        let mut child = Command::new("ffmpeg").args(&args).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().context("spawning ffmpeg (is it installed?)")?;
        let out = child.stdout.take().ok_or_else(|| anyhow!("no ffmpeg stdout"))?;
        Ok(Source::Ffmpeg { child, out, w, h })
    }

    pub fn read(&mut self) -> Result<Option<GrayImage>> {
        match self {
            Source::Ffmpeg { out, w, h, .. } => {
                let mut buf = vec![0u8; *w * *h];
                match out.read_exact(&mut buf) {
                    Ok(()) => Ok(Some(GrayImage { w: *w, h: *h, data: buf })),
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
                    Err(e) => Err(e.into()),
                }
            }
            Source::Images { files, i } => {
                if *i >= files.len() {
                    return Ok(None);
                }
                *i += 1;
                Ok(Some(GrayImage::load(&files[*i - 1])?))
            }
        }
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
