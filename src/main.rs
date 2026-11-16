use anyhow::{bail, Context, Result};
use axum::body::Bytes;
use clap::{Parser, Subcommand};
use palmid::camera::Source;
use palmid::detector::{Detector, ModelMeta};
use palmid::gallery::{Debounce, Gallery, GalleryConfig};
use palmid::imgproc::GrayImage;
use palmid::metrics;
use palmid::palmprint::{Code, PalmCoder, PalmParams};
use palmid::server::{router, AppState, Engine};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Parser)]
#[command(name = "palmid", version, about = "Monochrome face + palm detection and consent-based palmprint ID (CPU)")]
struct Cli {
    /// INT8 ONNX exported by the training notebook
    #[arg(long, global = true, default_value = "models/palmid_int8.onnx")]
    model: PathBuf,
    /// Model sidecar (classes, input size, per-class τ); defaults are used if missing
    #[arg(long, global = true, default_value = "models/palmid_model.json")]
    meta: PathBuf,
    /// Gallery of enrolled palm codes (JSON, codes only — no images)
    #[arg(long, global = true, default_value = "palm_gallery.json")]
    gallery: PathBuf,
    #[arg(long, global = true, default_value_t = 4)]
    threads: usize,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Detect faces and palms in images, print JSON
    Detect { images: Vec<PathBuf> },
    /// Enroll a consenting person from a camera, video or image folder
    Enroll {
        #[arg(long)]
        name: String,
        /// Must be `yes`: the person has agreed to palmprint enrollment on this machine
        #[arg(long)]
        consent: String,
        #[arg(long, default_value = "webcam")]
        source: String,
        #[arg(long, default_value_t = 40)]
        frames: usize,
    },
    /// Identify palms from a source (prints one JSON line per palm)
    Identify {
        #[arg(long, default_value = "webcam")]
        source: String,
        #[arg(long, default_value_t = 100)]
        max_frames: usize,
    },
    /// Remove a person and their codes from the gallery
    Delete { #[arg(long)] name: String },
    /// List enrolled people (names and thresholds only)
    List,
    /// Recalibrate per-class τ (Mondrian split-conformal) on a YOLO-format split and rewrite --meta
    Calibrate { #[arg(long)] data: PathBuf, #[arg(long, default_value = "cal")] split: String, #[arg(long, default_value_t = 0.30)] fallback: f32 },
    /// Frame-level MCC per class on a YOLO-format split
    Eval { #[arg(long)] data: PathBuf, #[arg(long, default_value = "val")] split: String },
    /// OpenAI-compatible /v1/chat/completions server
    Serve { #[arg(long, default_value_t = 8361)] port: u16 },
    /// Live camera loop + browser view at http://localhost:<port>/ + the same API
    Live {
        #[arg(long, default_value = "webcam")]
        source: String,
        #[arg(long, default_value_t = 8361)]
        port: u16,
        #[arg(long, default_value_t = 640)]
        width: usize,
        #[arg(long, default_value_t = 480)]
        height: usize,
    },
}

fn load_meta(p: &PathBuf) -> ModelMeta {
    ModelMeta::load(p).unwrap_or_else(|e| {
        eprintln!("[meta] {e:#} — using defaults (face, palm; τ = 0.30)");
        ModelMeta::default_face_palm()
    })
}

fn engine(cli: &Cli) -> Result<Engine> {
    Ok(Engine {
        det: Detector::new(&cli.model, load_meta(&cli.meta), cli.threads)?,
        coder: PalmCoder::new(PalmParams::default()),
        gallery: Gallery::open(&cli.gallery, GalleryConfig::default())?,
        debounce: Debounce::default(),
        debounce_n: 6,
        debounce_k: 3,
        hand: {
            let p = cli.model.with_file_name("handpose.onnx");
            if p.exists() { Some(palmid::handpose::HandLandmarker::new(&p, cli.threads)?) } else { None }
        },
        mesh: Some(palmid::facemesh::FaceMesher::embedded(cli.threads)?),
    })
}

/// Largest resolvable palm per frame → CompCode, until `n` codes or the source ends.
fn capture_codes(e: &mut Engine, source: &str, n: usize, max_frames: usize) -> Result<Vec<Code>> {
    let mut src = Source::open(source, 640, 480)?;
    let palm = e.det.meta.class_id("palm").context("model has no 'palm' class")?;
    let (mut codes, mut seen, mut small) = (vec![], 0usize, 0usize);
    while codes.len() < n && seen < max_frames {
        let Some(g) = src.read()? else { break };
        seen += 1;
        let best = e.det.detect(&g)?.into_iter().filter(|d| d.cls == palm).max_by(|a, b| {
            let area = |d: &palmid::Detection| (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]);
            area(a).partial_cmp(&area(b)).unwrap()
        });
        if let Some(d) = best {
            match e.coder.roi(&g, d.bbox) {
                Some(roi) => codes.push(e.coder.code(&roi)),
                None => small += 1,
            }
        }
        if seen % 30 == 0 {
            eprintln!("[capture] frames {seen}  palm codes {}/{n}  too-small {small}", codes.len());
        }
    }
    Ok(codes)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.cmd {
        Cmd::Detect { images } => {
            let mut e = engine(&cli)?;
            for p in images {
                let (r, _) = e.process(&GrayImage::load(p)?)?;
                println!("{}", serde_json::json!({"image": p, "result": r}));
            }
        }
        Cmd::Enroll { name, consent, source, frames } => {
            if consent != "yes" {
                bail!("--consent yes is required: only enroll people who have agreed to it");
            }
            let mut e = engine(&cli)?;
            eprintln!("[enroll] {name}: hold ONE open palm 25–40 cm from the camera, fingers spread, move slowly");
            let codes = capture_codes(&mut e, source, *frames, 3000)?;
            let out = e.gallery.enroll(name, &codes, true)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
            if !out.enrolled {
                std::process::exit(2);
            }
        }
        Cmd::Identify { source, max_frames } => {
            let mut e = engine(&cli)?;
            let mut src = Source::open(source, 640, 480)?;
            for _ in 0..*max_frames {
                let Some(g) = src.read()? else { break };
                let (r, _) = e.process(&g)?;
                for o in r.objects.iter().filter(|o| o.class == "palm") {
                    println!("{}", serde_json::to_string(o)?);
                }
            }
        }
        Cmd::Delete { name } => {
            let mut g = Gallery::open(&cli.gallery, GalleryConfig::default())?;
            println!("{}", if g.delete(name)? { format!("deleted {name}") } else { format!("{name} not enrolled") });
        }
        Cmd::List => {
            let g = Gallery::open(&cli.gallery, GalleryConfig::default())?;
            for (n, id) in &g.people {
                println!("{n}\ttau={:.4}\tchance={:.4}\tn_cal={}\t{}", id.tau, id.chance, id.n_cal, id.source);
            }
        }
        Cmd::Calibrate { data, split, fallback } => {
            let mut det = Detector::new(&cli.model, load_meta(&cli.meta), cli.threads)?;
            let th = metrics::calibrate(&mut det, &metrics::load_split(data, split)?, *fallback)?;
            println!("{}", serde_json::to_string_pretty(&th)?);
            let mut meta = det.meta.clone();
            meta.tau = th.iter().map(|t| t.tau).collect();
            std::fs::write(&cli.meta, serde_json::to_string_pretty(&meta)?)?;
            eprintln!("[calibrate] wrote {}", cli.meta.display());
        }
        Cmd::Eval { data, split } => {
            let mut det = Detector::new(&cli.model, load_meta(&cli.meta), cli.threads)?;
            println!("{}", serde_json::to_string_pretty(&metrics::evaluate(&mut det, &metrics::load_split(data, split)?)?)?);
        }
        Cmd::Serve { port } => {
            let state = AppState { engine: Arc::new(Mutex::new(engine(&cli)?)), live: None };
            let l = tokio::net::TcpListener::bind(("0.0.0.0", *port)).await?;
            eprintln!("[serve] http://0.0.0.0:{port}/v1/chat/completions");
            axum::serve(l, router(state)).await?;
        }
        Cmd::Live { source, port, width, height } => {
            let eng = Arc::new(Mutex::new(engine(&cli)?));
            let (tx, rx) = tokio::sync::watch::channel(Bytes::new());
            let (eng2, source, (w, h)) = (eng.clone(), source.clone(), (*width, *height));
            std::thread::spawn(move || -> Result<()> {
                let mut src = Source::open(&source, w, h)?;
                let (mut n, t0) = (0u64, std::time::Instant::now());
                while let Some(g) = src.read()? {
                    let mut e = eng2.lock().unwrap();
                    let (r, dets) = e.process(&g)?;
                    let palm = e.det.meta.class_id("palm");
                    drop(e);
                    let _ = tx.send(Bytes::from(palmid::server::annotate_with(&g, &dets, palm, &r.objects).to_jpeg(80)));
                    n += 1;
                    for o in r.objects.iter().filter(|o| o.identity.as_deref().is_some_and(|i| i != "pending")) {
                        eprintln!("[live] palm → {}", o.identity.as_ref().unwrap());
                    }
                    if n % 60 == 0 {
                        eprintln!("[live] {:.1} fps, last frame {:.1} ms", n as f64 / t0.elapsed().as_secs_f64(), r.latency_ms);
                    }
                }
                Ok(())
            });
            let l = tokio::net::TcpListener::bind(("0.0.0.0", *port)).await?;
            eprintln!("[live] open http://localhost:{port}/  (API: /v1/chat/completions)");
            axum::serve(l, router(AppState { engine: eng, live: Some(rx) })).await?;
        }
    }
    Ok(())
}
