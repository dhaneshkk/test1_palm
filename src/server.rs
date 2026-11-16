//! Engine (detector + palmprint + gallery + debounce) and the OpenAI-compatible HTTP API:
//! * `POST /v1/chat/completions` — user message `image_url` parts (data: URIs) → JSON results
//! * `GET  /v1/models`
//! * `GET  /live.mjpg`, `GET /` — annotated monochrome live view (when started with `live`)

use crate::detector::{Detection, Detector};
use crate::gallery::{Debounce, Gallery, Identification};
use crate::imgproc::GrayImage;
use crate::palmprint::PalmCoder;
use anyhow::Result;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::header;
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use base64::Engine as _;
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;

pub const MODEL_ID: &str = "palmid-mono-yolox-nano-int8";

#[derive(Serialize, Clone, Debug)]
pub struct ObjectResult {
    pub class: String,
    pub score: f32,
    pub bbox: [f32; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<crate::gallery::Candidate>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// 21 hand landmarks for palm detections, when the hand-landmark model is loaded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hand: Option<crate::handpose::HandPose>,
    /// 468-point face mesh for face detections, when the face-mesh model is loaded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub face_mesh: Option<crate::facemesh::FaceMesh>,
}

impl ObjectResult {
    pub fn new(class: String, score: f32, bbox: [f32; 4]) -> Self {
        Self { class, score, bbox, identity: None, candidates: None, note: None, hand: None, face_mesh: None }
    }
}

/// Codes for a palm box: box ROI always, landmark-aligned ROI when a hand pose is found.
pub fn palm_codes(
    gray: &GrayImage,
    bbox: [f32; 4],
    coder: &PalmCoder,
    hand: Option<&mut crate::handpose::HandLandmarker>,
) -> Result<(crate::palmprint::PalmCodes, Option<crate::handpose::HandPose>)> {
    palm_codes_for(gray, bbox, coder, hand, true, true)
}

/// As `palm_codes`, computing only the requested ROI modes (the hand pose is still estimated, for display).
pub fn palm_codes_for(
    gray: &GrayImage,
    bbox: [f32; 4],
    coder: &PalmCoder,
    hand: Option<&mut crate::handpose::HandLandmarker>,
    want_box: bool,
    want_aligned: bool,
) -> Result<(crate::palmprint::PalmCodes, Option<crate::handpose::HandPose>)> {
    let pose = match hand {
        Some(h) => h.landmarks(gray, bbox)?,
        None => None,
    };
    let boxed = if want_box { coder.roi(gray, bbox).map(|r| coder.code(&r)) } else { None };
    let aligned = if want_aligned { pose.as_ref().and_then(|p| coder.aligned_roi(gray, p)).map(|r| coder.code(&r)) } else { None };
    Ok((crate::palmprint::PalmCodes { boxed, aligned }, pose))
}

#[derive(Serialize, Clone, Debug)]
pub struct FrameResult {
    pub latency_ms: f64,
    pub objects: Vec<ObjectResult>,
}

pub struct Engine {
    pub det: Detector,
    pub coder: PalmCoder,
    pub gallery: Gallery,
    pub debounce: Debounce,
    pub debounce_n: u32,
    pub debounce_k: u32,
    pub hand: Option<crate::handpose::HandLandmarker>,
    pub mesh: Option<crate::facemesh::FaceMesher>,
}

impl Engine {
    pub fn process(&mut self, gray: &GrayImage) -> Result<(FrameResult, Vec<Detection>)> {
        let t = Instant::now();
        let dets = self.det.detect(gray)?;
        let palm = self.det.meta.class_id("palm");
        let mut objects = Vec::new();
        for d in &dets {
            let mut o = ObjectResult::new(self.det.meta.classes[d.cls].clone(), d.score, d.bbox);
            if Some(d.cls) == palm {
                let (wb, wa) = self.gallery.modes_needed();
                let (codes, pose) = palm_codes_for(gray, d.bbox, &self.coder, self.hand.as_mut(), wb || !wa, wa)?;
                o.hand = pose;
                if codes.boxed.is_none() && codes.aligned.is_none() {
                    o.note = Some("palm too small or cut off — move closer".into());
                } else {
                    let Identification { identity, candidates } = self.gallery.identify_codes(&codes);
                    let stable = self.debounce.update(&identity, self.debounce_n, self.debounce_k);
                    o.identity = Some(if stable { identity } else { "pending".into() });
                    o.candidates = Some(candidates);
                }
            } else if let Some(m) = self.mesh.as_mut().filter(|_| self.det.meta.classes[d.cls] == "face") {
                o.face_mesh = m.mesh(gray, d.bbox)?;
            }
            objects.push(o);
        }
        Ok((FrameResult { latency_ms: t.elapsed().as_secs_f64() * 1e3, objects }, dets))
    }
}

/// Annotate a frame: faces white, palms light grey (thick), for the MJPEG view.
pub fn annotate(gray: &GrayImage, dets: &[Detection], palm_cls: Option<usize>) -> GrayImage {
    annotate_with(gray, dets, palm_cls, &[])
}

/// As `annotate`, plus hand skeletons and face meshes.
pub fn annotate_with(gray: &GrayImage, dets: &[Detection], palm_cls: Option<usize>, objects: &[ObjectResult]) -> GrayImage {
    let mut vis = gray.clone();
    for o in objects {
        if let Some(h) = &o.hand {
            crate::handpose::draw_skeleton(&mut vis, h, 180, 255);
        }
        if let Some(m) = &o.face_mesh {
            crate::facemesh::draw_mesh(&mut vis, m, 150, 255);
        }
    }
    for d in dets {
        let (v, t) = if Some(d.cls) == palm_cls { (200, 4) } else { (255, 2) };
        vis.draw_rect(d.bbox, v, t);
    }
    vis
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Engine>>,
    pub live: Option<watch::Receiver<Bytes>>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

async fn models() -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "innoura"}]}))
}

fn decode_data_uri(url: &str) -> Option<GrayImage> {
    let b64 = url.split_once(',')?.1;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
    image::load_from_memory(&bytes).ok().map(|i| GrayImage::from_dynamic(&i))
}

async fn chat(State(st): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
    let mut results = Vec::new();
    let mut errors = Vec::new();
    for m in body["messages"].as_array().into_iter().flatten() {
        if m["role"] != "user" {
            continue;
        }
        for part in m["content"].as_array().into_iter().flatten() {
            if part["type"] == "image_url" {
                let url = part["image_url"]["url"].as_str().unwrap_or("");
                match decode_data_uri(url) {
                    Some(g) => match st.engine.lock().unwrap().process(&g) {
                        Ok((r, _)) => results.push(r),
                        Err(e) => errors.push(e.to_string()),
                    },
                    None => errors.push("image_url must be a data: URI with a PNG/JPEG".into()),
                }
            }
        }
    }
    let content = if results.is_empty() && errors.is_empty() {
        let e = st.engine.lock().unwrap();
        json!({"enrolled": e.gallery.people.keys().collect::<Vec<_>>(), "thresholds": e.det.meta.tau, "classes": e.det.meta.classes})
    } else {
        json!({"results": results, "errors": errors})
    };
    Json(json!({
        "id": format!("chatcmpl-{}", now()), "object": "chat.completion", "created": now(), "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content.to_string()}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }))
}

async fn index() -> Html<&'static str> {
    Html("<html><body style='background:#111;color:#ddd;font-family:sans-serif'><h3>palmid — monochrome live view</h3><img src='/live.mjpg' style='max-width:100%'></body></html>")
}

async fn mjpeg(State(st): State<AppState>) -> Response {
    let Some(rx) = st.live.clone() else {
        return (axum::http::StatusCode::NOT_FOUND, "live view only in `palmid live`").into_response();
    };
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        if rx.changed().await.is_err() {
            return None;
        }
        let jpg = rx.borrow_and_update().clone();
        let mut part = Vec::with_capacity(jpg.len() + 100);
        part.extend_from_slice(format!("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", jpg.len()).as_bytes());
        part.extend_from_slice(&jpg);
        part.extend_from_slice(b"\r\n");
        Some((Ok::<_, std::io::Error>(Bytes::from(part)), rx))
    });
    Response::builder()
        .header(header::CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame")
        .body(Body::from_stream(stream))
        .unwrap()
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/live.mjpg", get(mjpeg))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(state)
}
