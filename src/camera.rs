//! Frame sources with no OpenCV dependency: webcam/file/RTSP decoded by an `ffmpeg` subprocess
//! (`dshow:<camera name>` runs Windows `ffmpeg.exe`, e.g. from WSL2),
//! straight to 8-bit grayscale (`-pix_fmt gray`), or a directory of images.

use crate::imgproc::{GrayImage, RgbImage};
use anyhow::{anyhow, Context, Result};
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, ChildStdout, Command, Stdio};

pub enum Source {
    Ffmpeg { child: Child, out: Option<ChildStdout>, w: usize, h: usize, rgb: bool },
    Images { files: Vec<PathBuf>, i: usize },
}

impl Source {
    /// `spec`: `webcam` / `webcam:<device>` (platform default input), `dshow:<camera name>` (Windows ffmpeg.exe,
    /// works from WSL2), a video file / URL, or an image directory.
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
        let mut binary = "ffmpeg".to_string();
        if let Some(name) = spec.strip_prefix("dshow:") {
            // Windows camera from WSL2 (or native Windows): run Windows ffmpeg directly and read raw frames
            // from its stdout — no pipe file, no network, no firewall. Override the binary with PALMID_FFMPEG_WIN.
            binary = std::env::var("PALMID_FFMPEG_WIN").unwrap_or_else(|_| if cfg!(windows) { "ffmpeg".into() } else { "ffmpeg.exe".into() });
            args.extend(["-f".to_string(), "dshow".into(), "-i".into(), format!("video={name}"), "-r".into(), "25".into()]);
        } else if let Some(rest) = spec.strip_prefix("webcam") {
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
        let mut child = Command::new(&binary)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawning {binary} (is it installed and on PATH?)"))?;
        let out = child.stdout.take().ok_or_else(|| anyhow!("no ffmpeg stdout"))?;
        Ok(Source::Ffmpeg { child, out: Some(out), w, h, rgb })
    }

    /// Next frame as grayscale.
    pub fn read(&mut self) -> Result<Option<GrayImage>> {
        Ok(match self {
            Source::Ffmpeg { rgb: false, out, w, h, .. } => {
                let mut buf = vec![0u8; *w * *h];
                read_full(out.as_mut().ok_or_else(|| anyhow!("source closed"))?, &mut buf)?.then(|| GrayImage { w: *w, h: *h, data: buf })
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
                Ok(read_full(out.as_mut().ok_or_else(|| anyhow!("source closed"))?, &mut buf)?.then(|| RgbImage { w: *w, h: *h, data: buf }))
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
        if let Source::Ffmpeg { child, out, .. } = self {
            // Close our end first: a Windows ffmpeg.exe behind WSL interop exits on the broken pipe
            // even if the kill signal does not reach it.
            drop(out.take());
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
