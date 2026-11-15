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
}

impl Engine {
    pub fn process(&mut self, gray: &GrayImage) -> Result<(FrameResult, Vec<Detection>)> {
        let t = Instant::now();
        let dets = self.det.detect(gray)?;
        let palm = self.det.meta.class_id("palm");
        let mut objects = Vec::new();
        for d in &dets {
            let mut o = ObjectResult { class: self.det.meta.classes[d.cls].clone(), score: d.score, bbox: d.bbox, identity: None, candidates: None, note: None };
            if Some(d.cls) == palm {
                match self.coder.roi(gray, d.bbox) {
                    None => o.note = Some("palm too small or cut off — move closer".into()),
                    Some(roi) => {
                        let Identification { identity, candidates } = self.gallery.identify(&self.coder.code(&roi));
                        let stable = self.debounce.update(&identity, self.debounce_n, self.debounce_k);
                        o.identity = Some(if stable { identity } else { "pending".into() });
                        o.candidates = Some(candidates);
                    }
                }
            }
            objects.push(o);
        }
        Ok((FrameResult { latency_ms: t.elapsed().as_secs_f64() * 1e3, objects }, dets))
    }
}

/// Annotate a frame: faces white, palms light grey (thick), for the MJPEG view.
pub fn annotate(gray: &GrayImage, dets: &[Detection], palm_cls: Option<usize>) -> GrayImage {
    let mut vis = gray.clone();
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
