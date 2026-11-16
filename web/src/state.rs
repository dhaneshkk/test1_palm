//! Shared application state. One camera thread owns the frame source and runs whichever mode is
//! active (infer / collect / enroll). The training job is a supervised Python child process whose
//! JSON progress lines update the status. Every change bumps a `watch` version that the page's
//! live regions wait on.

use anyhow::{Context, Result, anyhow, bail};
use palmid::camera::Source;
use palmid::dataset::DatasetWriter;
use palmid::detector::{Detection, Detector, ModelMeta};
use palmid::gallery::{Debounce, EnrollOutcome, Gallery, GalleryConfig};
use palmid::imgproc::{GrayImage, RgbImage};
use palmid::lean_mirror::{Phase, can_transition};
use palmid::metrics;
use palmid::facemesh::FaceMesher;
use palmid::handpose::{HandLandmarker, HANDPOSE_URL};
use palmid::palmprint::{Code, PalmCoder, PalmParams, RoiMode};
use palmid::server::{palm_codes, palm_codes_for, ObjectResult};
use palmid::teachers::{PALM_URL, Teachers, YUNET_URL};
use serde::Serialize;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::io::AsyncBufReadExt;
use tokio::sync::watch;

#[derive(Clone, Debug)]
pub struct Config {
    pub work: PathBuf,
    pub source: String,
    pub width: usize,
    pub height: usize,
    pub python: String,
    pub trainer: PathBuf,
    pub threads: usize,
}

impl Config {
    pub fn from_env() -> Self {
        let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        Self {
            work: PathBuf::from(env("PALMID_WORK", "palmid_work")),
            source: env("PALMID_SOURCE", "webcam"),
            width: env("PALMID_WIDTH", "640").parse().unwrap_or(640),
            height: env("PALMID_HEIGHT", "480").parse().unwrap_or(480),
            python: env("PALMID_PYTHON", "python3"),
            trainer: PathBuf::from(env("PALMID_TRAINER", "trainer/train_student.py")),
            threads: env("PALMID_THREADS", "4").parse().unwrap_or(4),
        }
    }
    pub fn models(&self) -> PathBuf { self.work.join("models") }
    pub fn dataset(&self) -> PathBuf { self.work.join("dataset") }
    pub fn onnx(&self) -> PathBuf { self.models().join("palmid_int8.onnx") }
    pub fn meta(&self) -> PathBuf { self.models().join("palmid_model.json") }
    pub fn gallery(&self) -> PathBuf { self.work.join("palm_gallery.json") }
    pub fn finger_recordings(&self) -> PathBuf { self.work.join("finger_recordings.jsonl") }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CameraStatus {
    pub source: String,
    pub running: bool,
    pub fps: f64,
    pub frame_ms: f64,
    pub frames: u64,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CollectStatus {
    pub active: bool,
    pub frames: usize,
    pub target: usize,
    pub faces: usize,
    pub palms: usize,
    pub split: [usize; 3],
    pub message: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClassMetric {
    pub class: String,
    pub tau: f32,
    pub tau_source: String,
    pub n_cal: usize,
    pub mcc: Option<f64>,
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
    pub tn: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainStatus {
    pub phase: Phase,
    pub epoch: usize,
    pub epochs: usize,
    pub iter: usize,
    pub total: usize,
    pub last_loss: Option<f64>,
    pub lr: Option<f64>,
    pub elapsed_s: f64,
    pub epoch_losses: Vec<f64>,
    pub iter_losses: Vec<f64>,
    pub log: VecDeque<String>,
    pub error: Option<String>,
    pub metrics: Vec<ClassMetric>,
}

impl Default for TrainStatus {
    fn default() -> Self {
        Self { phase: Phase::Idle, epoch: 0, epochs: 0, iter: 0, total: 0, last_loss: None, lr: None, elapsed_s: 0.0,
               epoch_losses: vec![], iter_losses: vec![], log: VecDeque::new(), error: None, metrics: vec![] }
    }
}

/// Labelled finger-number recordings, for measuring the rules on real hands.
#[derive(Clone, Debug, Default, Serialize)]
pub struct FingerRecStatus {
    /// number being recorded and frames still to capture
    pub active: Option<(u8, usize)>,
    /// recorded frames per number (index 1–10)
    pub counts: [usize; 11],
    pub report: Option<palmid::fingernum::FingerReport>,
}

/// Frames with a hand captured per "Record" press (about 3–6 s at the usual frame rates).
pub const FINGER_REC_FRAMES: usize = 60;

#[derive(Clone, Debug, Default, Serialize)]
pub struct EnrollStatus {
    pub active: bool,
    pub name: String,
    pub codes: usize,
    pub target: usize,
    pub too_small: usize,
    pub last: Option<(String, EnrollOutcome)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Person {
    pub name: String,
    pub tau: f64,
    pub chance: f64,
    pub n_cal: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Status {
    pub camera: CameraStatus,
    pub objects: Vec<ObjectResult>,
    pub collect: CollectStatus,
    pub train: TrainStatus,
    pub enroll: EnrollStatus,
    pub model_loaded: Option<String>,
    /// 21-point hand-landmark model: "loaded", "downloading…" or an error
    pub hand_model: Option<String>,
    /// 468-point face-mesh model (embedded): "loaded" or an error
    pub face_model: Option<String>,
    /// Chinese finger number held steady (≥ 6 of the last 8 frames) by the most confident hand
    pub finger_number: Option<u8>,
    pub finger_rec: FingerRecStatus,
    pub people: Vec<Person>,
}

/// Give up enrolling after this many frames (~30–60 s) if not enough palm views were captured.
const ENROLL_FRAME_BUDGET: usize = 450;

enum Mode {
    Infer,
    Collect { writer: DatasetWriter, target: usize },
    Enroll { name: String, target: usize, codes: Vec<Code>, seen: usize, roi: Option<RoiMode> },
}

pub struct Runtime {
    pub det: Option<Detector>,
    pub coder: PalmCoder,
    pub gallery: Gallery,
    pub debounce: Debounce,
    pub hand: Option<HandLandmarker>,
    pub mesh: Option<FaceMesher>,
    /// palm box re-derived from the last frame's landmarks, to follow a hand the detector misses
    pub track: Option<[f32; 4]>,
}

impl Runtime {
    /// Detect, and for each palm compute its code and open-set identity (debounced).
    pub fn process(&mut self, gray: &GrayImage) -> Result<(Vec<ObjectResult>, Vec<Detection>)> {
        let Some(det) = self.det.as_mut() else { return Ok((vec![], vec![])) };
        let dets = det.detect(gray)?;
        let palm = det.meta.class_id("palm");
        let mut out = vec![];
        for d in &dets {
            let mut o = ObjectResult::new(det.meta.classes[d.cls].clone(), d.score, d.bbox);
            if Some(d.cls) == palm {
                let (wb, wa) = self.gallery.modes_needed();
                let (codes, pose) = palm_codes_for(gray, d.bbox, &self.coder, self.hand.as_mut(), wb || !wa, wa)?;
                o.hand = pose;
                o.finger_number = o.hand.as_ref().and_then(|h| palmid::fingernum::classify(h, &palmid::fingernum::Rules::default()));
                if codes.boxed.is_none() && codes.aligned.is_none() {
                    o.note = Some("move closer".into());
                } else {
                    let id = self.gallery.identify_codes(&codes);
                    let stable = self.debounce.update(&id.identity, 6, 3);
                    o.identity = Some(if stable { id.identity } else { "pending".into() });
                    o.candidates = Some(id.candidates);
                }
            } else if let Some(m) = self.mesh.as_mut().filter(|_| det.meta.classes[d.cls] == "face") {
                o.face_mesh = m.mesh(gray, d.bbox)?;
            }
            out.push(o);
        }
        // Follow the hand with landmarks alone when the detector drops it — palm detectors rarely
        // fire on fists, pinches or hooked fingers (finger numbers 7, 9, 10). The landmark model's
        // own confidence (≥ 0.8) ends the track.
        let best = out.iter().filter_map(|o| o.hand.as_ref()).max_by(|a, b| a.confidence.total_cmp(&b.confidence));
        self.track = match (best, self.track, self.hand.as_mut()) {
            (Some(h), _, _) => Some(palmid::handpose::palm_box_from_landmarks(h)),
            (None, Some(b), Some(hl)) => match hl.landmarks(gray, b)? {
                Some(h) => {
                    let nb = palmid::handpose::palm_box_from_landmarks(&h);
                    let mut o = ObjectResult::new("palm".into(), h.confidence, nb);
                    o.note = Some("tracked by landmarks".into());
                    o.finger_number = palmid::fingernum::classify(&h, &palmid::fingernum::Rules::default());
                    o.hand = Some(h);
                    out.push(o);
                    Some(nb)
                }
                None => None,
            },
            _ => None,
        };
        Ok((out, dets))
    }
}

pub struct Inner {
    pub cfg: Config,
    status: Mutex<Status>,
    pub rt: Mutex<Runtime>,
    mode: Mutex<Mode>,
    teachers: Mutex<Option<Teachers>>,
    frame: Mutex<Arc<Vec<u8>>>,
    status_tx: watch::Sender<u64>,
    frame_tx: watch::Sender<u64>,
    job: Mutex<Option<tokio::task::AbortHandle>>,
    child_pid: Mutex<Option<u32>>,
}

/// Router-shared handle (`context::<App>(cx)`).
#[derive(Clone)]
pub struct App(pub Arc<Inner>);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl App {
    pub fn new(cfg: Config) -> Result<Self> {
        std::fs::create_dir_all(cfg.models())?;
        let gallery = Gallery::open(&cfg.gallery(), GalleryConfig::default())?;
        let mut status = Status { camera: CameraStatus { source: cfg.source.clone(), ..Default::default() }, ..Default::default() };
        let det = if cfg.onnx().exists() {
            let meta = ModelMeta::load(&cfg.meta()).unwrap_or_else(|_| ModelMeta::default_face_palm());
            match Detector::new(&cfg.onnx(), meta, cfg.threads) {
                Ok(d) => {
                    status.model_loaded = Some(cfg.onnx().display().to_string());
                    status.train.phase = Phase::Ready;
                    Some(d)
                }
                Err(e) => {
                    status.train.error = Some(format!("existing model failed to load: {e:#}"));
                    None
                }
            }
        } else {
            None
        };
        let app = App(Arc::new(Inner {
            rt: Mutex::new(Runtime { det, coder: PalmCoder::new(PalmParams::default()), gallery, debounce: Debounce::default(), hand: None, mesh: None, track: None }),
            cfg,
            status: Mutex::new(status),
            mode: Mutex::new(Mode::Infer),
            teachers: Mutex::new(None),
            frame: Mutex::new(Arc::new(vec![])),
            status_tx: watch::channel(0).0,
            frame_tx: watch::channel(0).0,
            job: Mutex::new(None),
            child_pid: Mutex::new(None),
        }));
        app.refresh_people();
        app.load_finger_counts();
        app.spawn_hand_model();
        app.spawn_face_model();
        Ok(app)
    }

    /// Load (downloading on first run) the 21-point hand-landmark model in the background.
    fn spawn_hand_model(&self) {
        let app = self.clone();
        std::thread::spawn(move || {
            let path = app.0.cfg.models().join("handpose.onnx");
            if !path.exists() {
                app.update(|s| s.hand_model = Some("downloading…".into()));
                let ok = std::process::Command::new("curl").args(["-fsSL", "-o"]).arg(&path).arg(HANDPOSE_URL).status().map(|s| s.success()).unwrap_or(false);
                if !ok {
                    let _ = std::fs::remove_file(&path);
                    app.update(|s| s.hand_model = Some("download failed — palms work without landmarks".into()));
                    return;
                }
            }
            match HandLandmarker::new(&path, app.0.cfg.threads) {
                Ok(h) => {
                    lock(&app.0.rt).hand = Some(h);
                    app.update(|s| s.hand_model = Some("loaded".into()));
                }
                Err(e) => app.update(|s| s.hand_model = Some(format!("failed: {e:#}"))),
            }
        });
    }

    /// Load the embedded 468-point face-mesh model in the background.
    fn spawn_face_model(&self) {
        let app = self.clone();
        std::thread::spawn(move || match FaceMesher::embedded(app.0.cfg.threads) {
            Ok(m) => {
                lock(&app.0.rt).mesh = Some(m);
                app.update(|s| s.face_model = Some("loaded".into()));
            }
            Err(e) => app.update(|s| s.face_model = Some(format!("failed: {e:#}"))),
        });
    }

    // ── finger-number recordings ───────────────────────────────────────────────
    pub fn start_finger_recording(&self, label: u8) -> Result<String> {
        anyhow::ensure!((1..=10).contains(&label), "number must be 1–10");
        anyhow::ensure!(lock(&self.0.rt).hand.is_some(), "the hand-landmark model is not loaded yet");
        self.update(|s| s.finger_rec.active = Some((label, FINGER_REC_FRAMES)));
        Ok(format!("recording {FINGER_REC_FRAMES} frames of “{label}” — hold the gesture, move it around a little"))
    }

    fn record_finger_frame(&self, label: u8, h: &palmid::handpose::HandPose, frame: u64) -> Result<()> {
        use std::io::Write;
        let rec = palmid::fingernum::Recording { label, points: h.points.clone(), confidence: h.confidence, frame };
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.0.cfg.finger_recordings())?;
        writeln!(f, "{}", serde_json::to_string(&rec)?)?;
        self.update(|s| {
            s.finger_rec.counts[label as usize] += 1;
            s.finger_rec.active = match s.finger_rec.active { Some((l, left)) if left > 1 => Some((l, left - 1)), _ => None };
        });
        Ok(())
    }

    /// Score the rules on everything recorded so far (frame level, per-number and multiclass MCC).
    pub fn score_finger_recordings(&self) -> Result<String> {
        let path = self.0.cfg.finger_recordings();
        anyhow::ensure!(path.exists(), "nothing recorded yet");
        let recs = palmid::fingernum::read_recordings(&path)?;
        let rep = palmid::fingernum::evaluate(&recs, &palmid::fingernum::Rules::default());
        let msg = format!("{} frames: accuracy {:.1} %, MCC {}", rep.frames, 100.0 * rep.accuracy, rep.mcc.map(|m| format!("{m:.3}")).unwrap_or_else(|| "n/a (record at least two numbers)".into()));
        self.update(|s| s.finger_rec.report = Some(rep));
        Ok(msg)
    }

    fn load_finger_counts(&self) {
        if let Ok(recs) = palmid::fingernum::read_recordings(&self.0.cfg.finger_recordings()) {
            self.update(|s| recs.iter().for_each(|r| s.finger_rec.counts[(r.label as usize).min(10)] += 1));
        }
    }

    pub fn status(&self) -> Status {
        lock(&self.0.status).clone()
    }
    pub fn subscribe_status(&self) -> watch::Receiver<u64> {
        self.0.status_tx.subscribe()
    }
    pub fn subscribe_frames(&self) -> watch::Receiver<u64> {
        self.0.frame_tx.subscribe()
    }
    pub fn frame_jpeg(&self) -> Arc<Vec<u8>> {
        lock(&self.0.frame).clone()
    }
    fn update(&self, f: impl FnOnce(&mut Status)) {
        f(&mut lock(&self.0.status));
        self.0.status_tx.send_modify(|v| *v += 1);
    }
    fn log(&self, msg: impl Into<String>) {
        let msg = msg.into();
        self.update(|s| {
            s.train.log.push_back(msg);
            while s.train.log.len() > 40 {
                s.train.log.pop_front();
            }
        });
    }
    /// Phase changes go through the Lean-mirrored transition relation.
    fn set_phase(&self, to: Phase) -> Result<()> {
        let mut st = lock(&self.0.status);
        let from = st.train.phase;
        if !can_transition(from, to) {
            bail!("cannot go from {from:?} to {to:?}");
        }
        st.train.phase = to;
        drop(st);
        self.0.status_tx.send_modify(|v| *v += 1);
        Ok(())
    }

    fn refresh_people(&self) {
        let people = lock(&self.0.rt).gallery.people.iter().map(|(n, p)| Person { name: n.clone(), tau: p.tau, chance: p.chance, n_cal: p.n_cal }).collect();
        self.update(|s| s.people = people);
    }

    // ── camera thread ──────────────────────────────────────────────────────────
    pub fn spawn_camera(&self) {
        let app = self.clone();
        std::thread::Builder::new().name("camera".into()).spawn(move || loop {
            if let Err(e) = app.camera_session() {
                app.update(|s| { s.camera.running = false; s.camera.error = Some(format!("{e:#}")); });
            }
            std::thread::sleep(Duration::from_secs(2));
        }).expect("camera thread");
    }

    fn camera_session(&self) -> Result<()> {
        let cfg = &self.0.cfg;
        let mut src = Source::open_rgb(&cfg.source, cfg.width, cfg.height).with_context(|| format!("opening {}", cfg.source))?;
        self.update(|s| { s.camera.running = true; s.camera.error = None; });
        let (mut n, t0, mut last_pub) = (0u64, Instant::now(), Instant::now() - Duration::from_secs(1));
        let mut fingers = palmid::fingernum::Stabilizer::default();
        while let Some(rgb) = src.read_rgb()? {
            let t = Instant::now();
            let gray = rgb.to_gray();
            let (objects, boxes) = self.step(&rgb, &gray)?;
            let best = objects.iter().filter(|o| o.hand.is_some()).max_by(|a, b| a.hand.as_ref().unwrap().confidence.total_cmp(&b.hand.as_ref().unwrap().confidence));
            let held = fingers.push(best.and_then(|o| o.finger_number.as_ref()).map(|f| f.value));
            let rec = lock(&self.0.status).finger_rec.active; // guard dropped here: record_finger_frame locks status again
            if let (Some((label, _)), Some(h)) = (rec, best.and_then(|o| o.hand.as_ref())) {
                self.record_finger_frame(label, h, n)?;
            }
            let mut vis = gray.clone();
            for (b, v, w) in boxes {
                vis.draw_rect(b, v, w);
            }
            for o in &objects {
                if let Some(h) = &o.hand {
                    palmid::handpose::draw_skeleton(&mut vis, h, 180, 255);
                }
                if let Some(m) = &o.face_mesh {
                    palmid::facemesh::draw_mesh(&mut vis, m, 150, 255);
                }
            }
            *lock(&self.0.frame) = Arc::new(vis.to_jpeg(75));
            self.0.frame_tx.send_modify(|v| *v += 1);
            n += 1;
            if last_pub.elapsed() > Duration::from_millis(250) {
                last_pub = Instant::now();
                let fps = n as f64 / t0.elapsed().as_secs_f64();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                self.update(|s| { s.camera.fps = fps; s.camera.frame_ms = ms; s.camera.frames = n; s.objects = objects; s.finger_number = held; });
            }
        }
        bail!("source ended")
    }

    /// One frame in the active mode. Returns display objects and boxes to draw (box, grey, thickness).
    fn step(&self, rgb: &RgbImage, gray: &GrayImage) -> Result<(Vec<ObjectResult>, Vec<([f32; 4], u8, usize)>)> {
        let mut mode = lock(&self.0.mode);
        match &mut *mode {
            Mode::Infer => {
                drop(mode);
                let (objs, dets) = lock(&self.0.rt).process(gray)?;
                let palm_cls = objs.iter().zip(&dets).find(|(o, _)| o.class == "palm").map(|(_, d)| d.cls);
                Ok((objs, dets.iter().map(|d| (d.bbox, if Some(d.cls) == palm_cls { 200 } else { 255 }, if Some(d.cls) == palm_cls { 4 } else { 2 })).collect()))
            }
            Mode::Collect { writer, target } => {
                let mut tg = lock(&self.0.teachers);
                let t = tg.as_mut().ok_or_else(|| anyhow!("teachers not loaded"))?;
                let (faces, palms) = (t.faces(rgb)?, t.palms(rgb)?);
                drop(tg);
                let boxes: Vec<(usize, [f32; 4])> = faces.iter().map(|b| (0, *b)).chain(palms.iter().map(|b| (1, *b))).collect();
                writer.add(gray, &boxes)?;
                let (count, target, pc) = (writer.count, *target, writer.per_class.clone());
                let si = ["train", "cal", "val"].iter().position(|s| *s == palmid::dataset::split_for(count - 1)).unwrap();
                let done = count >= target;
                if done {
                    *mode = Mode::Infer;
                }
                drop(mode);
                self.update(|s| {
                    s.collect.frames = count;
                    s.collect.faces = pc[0];
                    s.collect.palms = pc[1];
                    s.collect.split[si] += 1;
                    if done {
                        s.collect.active = false;
                        s.collect.message = Some(if pc[1] < 30 {
                            format!("done — only {} palm labels; show open palms more and collect again", pc[1])
                        } else if pc[1] >= count || pc[0] >= count {
                            "done — but every frame had a face or palm; MCC needs some frames without them (step away / hands down for part of the run)".into()
                        } else {
                            "done — ready to train".into()
                        });
                    }
                });
                if done {
                    let _ = self.set_phase(Phase::Collected);
                }
                let objs = boxes
                    .iter()
                    .map(|(c, b)| {
                        let mut o = ObjectResult::new(["face", "palm"][*c].into(), 1.0, *b);
                        o.note = Some("teacher".into());
                        o
                    })
                    .collect();
                Ok((objs, boxes.iter().map(|(c, b)| (*b, if *c == 1 { 200 } else { 255 }, 2)).collect()))
            }
            Mode::Enroll { name, target, codes, seen, roi } => {
                *seen += 1;
                let seen_now = *seen;
                let mut rt = lock(&self.0.rt);
                let Runtime { det, coder, hand, .. } = &mut *rt;
                let det = det.as_mut().ok_or_else(|| anyhow!("no model"))?;
                let palm = det.meta.class_id("palm").ok_or_else(|| anyhow!("model has no palm class"))?;
                let dets = det.detect(gray)?;
                let best = dets.iter().filter(|d| d.cls == palm).max_by(|a, b| area(a).partial_cmp(&area(b)).unwrap()).cloned();
                let mut small = false;
                let mut pose_drawn = None;
                if let Some(d) = &best {
                    let (pc, pose) = palm_codes(gray, d.bbox, coder, hand.as_mut())?;
                    pose_drawn = pose;
                    // the first usable frame fixes the ROI mode for this person (landmarks preferred)
                    if roi.is_none() {
                        *roi = pc.preferred().map(|(m, _)| m);
                    }
                    match roi.and_then(|m| pc.get(m).cloned()) {
                        Some(c) => codes.push(c),
                        None => small = true,
                    }
                }
                let (n, target_v, name_v, roi_v) = (codes.len(), *target, name.clone(), roi.unwrap_or_default());
                let mut outcome = None;
                if n >= target_v {
                    let all = std::mem::take(codes);
                    *mode = Mode::Infer;
                    outcome = Some(rt.gallery.enroll_with(&name_v, &all, true, roi_v)?);
                } else if seen_now >= ENROLL_FRAME_BUDGET {
                    *mode = Mode::Infer;
                    outcome = Some(EnrollOutcome {
                        enrolled: false,
                        reason: Some(format!("only {n} usable palm views in {seen_now} frames — move closer, add light, or train the model longer")),
                        tau: 0.0, chance: 0.0, n_cal: 0, genuine_median: None, source: "timeout".into(), roi: roi_v,
                    });
                }
                drop(rt);
                drop(mode);
                self.update(|s| {
                    s.enroll.codes = n;
                    s.enroll.too_small += small as usize;
                    if let Some(o) = &outcome {
                        s.enroll.active = false;
                        s.enroll.last = Some((name_v.clone(), o.clone()));
                    }
                });
                if outcome.is_some() {
                    self.refresh_people();
                }
                let mut objs = vec![];
                if let Some(d) = &best {
                    let mut o = ObjectResult::new("palm".into(), d.score, d.bbox);
                    o.hand = pose_drawn;
                    objs.push(o);
                }
                Ok((objs, best.map(|d| vec![(d.bbox, 200, 4)]).unwrap_or_default()))
            }
        }
    }

    // ── actions ─────────────────────────────────────────────────────────────────
    pub async fn start_collect(&self, target: usize) -> Result<String> {
        if !(20..=5000).contains(&target) {
            bail!("frames must be between 20 and 5000");
        }
        if lock(&self.0.teachers).is_none() {
            let m = self.0.cfg.models();
            let (y, p) = (m.join("yunet.onnx"), m.join("palm_teacher.onnx"));
            for (path, url) in [(&y, YUNET_URL), (&p, PALM_URL)] {
                if !path.exists() {
                    self.log(format!("downloading {}", path.file_name().unwrap().to_string_lossy()));
                    let ok = tokio::process::Command::new("curl").args(["-fsSL", "-o"]).arg(path).arg(url).status().await?.success();
                    if !ok {
                        bail!("download failed: {url}");
                    }
                }
            }
            let threads = self.0.cfg.threads;
            let t = tokio::task::spawn_blocking(move || Teachers::new(&y, &p, threads)).await??;
            *lock(&self.0.teachers) = Some(t);
        }
        if !matches!(*lock(&self.0.mode), Mode::Infer) {
            bail!("busy (collecting or enrolling)");
        }
        self.set_phase(Phase::Collecting)?;
        let writer = DatasetWriter::create(&self.0.cfg.dataset(), 2)?;
        *lock(&self.0.mode) = Mode::Collect { writer, target };
        self.update(|s| s.collect = CollectStatus { active: true, target, ..Default::default() });
        Ok(format!("collecting {target} frames"))
    }

    pub fn start_enroll(&self, name: String, consent: bool) -> Result<String> {
        let name = name.trim().to_string();
        if !consent {
            bail!("tick the consent box: only enroll people who agreed to it");
        }
        if name.is_empty() || name.len() > 64 || !name.chars().all(|c| c.is_alphanumeric() || " _-.".contains(c)) {
            bail!("name: 1–64 letters, digits, space, _ - .");
        }
        if lock(&self.0.rt).det.is_none() {
            bail!("train or load a model first");
        }
        let mut mode = lock(&self.0.mode);
        if !matches!(*mode, Mode::Infer) {
            bail!("busy (collecting or enrolling)");
        }
        *mode = Mode::Enroll { name: name.clone(), target: 40, codes: vec![], seen: 0, roi: None };
        drop(mode);
        self.update(|s| s.enroll = EnrollStatus { active: true, name: name.clone(), target: 40, ..Default::default() });
        Ok(format!("enrolling {name}: hold one open palm 25–40 cm from the camera"))
    }

    pub fn delete_person(&self, name: &str) -> Result<bool> {
        let had = lock(&self.0.rt).gallery.delete(name)?;
        self.refresh_people();
        Ok(had)
    }

    pub fn cancel_training(&self) -> bool {
        if let Some(h) = lock(&self.0.job).take() {
            h.abort();
            if let Some(pid) = lock(&self.0.child_pid).take() {
                let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
            }
            let _ = self.set_phase(Phase::Failed);
            self.update(|s| s.train.error = Some("cancelled".into()));
            true
        } else {
            false
        }
    }

    pub fn start_training(&self, epochs: usize) -> Result<String> {
        if !(1..=300).contains(&epochs) {
            bail!("epochs must be between 1 and 300");
        }
        if !matches!(*lock(&self.0.mode), Mode::Infer) {
            bail!("wait for collection / enrollment to finish");
        }
        let train_dir = self.0.cfg.dataset().join("images/train");
        let n = std::fs::read_dir(&train_dir).map(|d| d.count()).unwrap_or(0);
        if n < 8 {
            bail!("collect data first ({n} training frames)");
        }
        self.set_phase(Phase::Training)?;
        self.update(|s| { let log = std::mem::take(&mut s.train.log); s.train = TrainStatus { phase: Phase::Training, epochs, log, ..Default::default() }; });
        let app = self.clone();
        let handle = tokio::spawn(async move {
            if let Err(e) = app.run_training(epochs).await {
                app.log(format!("error: {e:#}"));
                let _ = app.set_phase(Phase::Failed);
                app.update(|s| s.train.error = Some(format!("{e:#}")));
            }
            *lock(&app.0.job) = None;
        });
        *lock(&self.0.job) = Some(handle.abort_handle());
        Ok(format!("training {epochs} epochs with SAM"))
    }

    async fn run_training(&self, epochs: usize) -> Result<()> {
        let cfg = self.0.cfg.clone();
        let out_dir = cfg.work.join("train_out");
        let mut child = tokio::process::Command::new(&cfg.python)
            .arg(&cfg.trainer)
            .args(["--data".as_ref(), cfg.dataset().as_os_str(), "--out".as_ref(), out_dir.as_os_str()])
            .args(["--epochs", &epochs.to_string(), "--threads", &cfg.threads.to_string()])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("starting {} {}", cfg.python, cfg.trainer.display()))?;
        *lock(&self.0.child_pid) = child.id();
        let mut lines = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
        let stderr = child.stderr.take().unwrap();
        let err_tail = tokio::spawn(async move {
            let mut l = tokio::io::BufReader::new(stderr).lines();
            let mut tail = VecDeque::new();
            while let Ok(Some(line)) = l.next_line().await {
                tail.push_back(line);
                if tail.len() > 20 { tail.pop_front(); }
            }
            tail.into_iter().collect::<Vec<_>>().join("\n")
        });
        let mut done: Option<(PathBuf, PathBuf)> = None;
        while let Some(line) = lines.next_line().await? {
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            match ev["event"].as_str().unwrap_or("") {
                "start" => self.log(format!("trainer started on {} — {} train / {} cal frames, labels {}", ev["device"].as_str().unwrap_or("?"), ev["train"], ev["cal"], ev["labels"])),
                "status" => self.log(ev["message"].as_str().unwrap_or("").to_string()),
                "iter" => self.update(|s| {
                    s.train.iter = ev["iter"].as_u64().unwrap_or(0) as usize;
                    s.train.total = ev["total"].as_u64().unwrap_or(0) as usize;
                    s.train.last_loss = ev["loss"].as_f64();
                    s.train.lr = ev["lr"].as_f64();
                    s.train.elapsed_s = ev["seconds"].as_f64().unwrap_or(0.0);
                    if let Some(l) = ev["loss"].as_f64() { s.train.iter_losses.push(l); }
                }),
                "epoch" => {
                    self.update(|s| { s.train.epoch = ev["epoch"].as_u64().unwrap_or(0) as usize; if let Some(l) = ev["loss"].as_f64() { s.train.epoch_losses.push(l); } });
                    self.log(format!("epoch {}/{}  loss {}  {}s", ev["epoch"], ev["epochs"], ev["loss"], ev["seconds"]));
                }
                "export" => self.log(format!("export: {}", ev["stage"].as_str().unwrap_or(""))),
                "done" => done = Some((PathBuf::from(ev["onnx"].as_str().unwrap_or("")), PathBuf::from(ev["meta"].as_str().unwrap_or("")))),
                "error" => bail!("trainer: {}", ev["message"].as_str().unwrap_or("?")),
                _ => {}
            }
        }
        let status = child.wait().await?;
        *lock(&self.0.child_pid) = None;
        let Some((onnx, meta)) = done else {
            bail!("trainer exited ({status}) without a model:\n{}", err_tail.await.unwrap_or_default());
        };
        self.set_phase(Phase::Calibrating)?;
        self.log("calibrating per-class τ (Mondrian split-conformal) and evaluating MCC in Rust");
        let app = self.clone();
        let (det, metrics) = tokio::task::spawn_blocking(move || app.calibrate_and_eval(&onnx, &meta)).await??;
        // Lean §6 `swap_requires_calibrated`: the live detector is replaced only on Calibrating → Ready.
        self.set_phase(Phase::Ready)?;
        lock(&self.0.rt).det = Some(det);
        lock(&self.0.rt).debounce = Debounce::default();
        let summary = metrics.iter().map(|m| format!("{} MCC {} (τ {:.3}, {})", m.class, m.mcc.map_or("n/a".into(), |v| format!("{v:.3}")), m.tau, m.tau_source)).collect::<Vec<_>>().join("; ");
        self.update(|s| { s.train.metrics = metrics; s.model_loaded = Some(cfg.onnx().display().to_string()); });
        self.log(format!("model live: {summary}"));
        Ok(())
    }

    fn calibrate_and_eval(&self, onnx: &Path, meta_path: &Path) -> Result<(Detector, Vec<ClassMetric>)> {
        let cfg = &self.0.cfg;
        let mut det = Detector::new(onnx, ModelMeta::load(meta_path)?, cfg.threads)?;
        let cal = metrics::load_split(&cfg.dataset(), "cal")?;
        let val = metrics::load_split(&cfg.dataset(), "val")?;
        let th = metrics::calibrate(&mut det, &cal, 0.30)?;
        det.meta.tau = th.iter().map(|t| t.tau).collect();
        let mcc = metrics::evaluate(&mut det, &val)?;
        std::fs::copy(onnx, cfg.onnx())?;
        std::fs::write(cfg.meta(), serde_json::to_string_pretty(&det.meta)?)?;
        let det = Detector::new(&cfg.onnx(), det.meta.clone(), cfg.threads)?;
        let metrics = th.into_iter().zip(mcc).map(|(t, m)| ClassMetric { class: t.class, tau: t.tau, tau_source: t.source, n_cal: t.n_cal, mcc: m.mcc, tp: m.tp, fp: m.fp, fn_: m.fn_, tn: m.tn }).collect();
        Ok((det, metrics))
    }
}

fn area(d: &Detection) -> f32 {
    (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
}
