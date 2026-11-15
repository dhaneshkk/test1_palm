//! OpenAI-compatible inference API, sharing the live runtime (same model, gallery, debounce).

use crate::state::App;
use base64::Engine as _;
use palmid::imgproc::GrayImage;
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};
use topcoat::{
    Result,
    context::{Cx, context},
    router::{content::Json, route},
};

const MODEL_ID: &str = "palmid-mono-yolox-nano-int8";

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[route(GET "/v1/models")]
async fn models(cx: &Cx) -> Result<Json<Value>> {
    Ok(Json(json!({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "innoura"}]})))
}

#[route(POST "/v1/chat/completions")]
async fn chat(cx: &Cx, Json(body): Json<Value>) -> Result<Json<Value>> {
    let app = context::<App>(cx).clone();
    let mut images = vec![];
    for m in body["messages"].as_array().into_iter().flatten().filter(|m| m["role"] == "user") {
        for p in m["content"].as_array().into_iter().flatten().filter(|p| p["type"] == "image_url") {
            images.push(p["image_url"]["url"].as_str().unwrap_or("").to_string());
        }
    }
    let content = tokio::task::spawn_blocking(move || {
        let mut results = vec![];
        let mut errors = vec![];
        for url in images {
            let decoded = url
                .split_once(',')
                .filter(|(h, _)| h.starts_with("data:"))
                .and_then(|(_, b)| base64::engine::general_purpose::STANDARD.decode(b).ok())
                .and_then(|bytes| image::load_from_memory(&bytes).ok());
            match decoded {
                None => errors.push("image_url must be a data: URI with PNG/JPEG".to_string()),
                Some(img) => {
                    let t = std::time::Instant::now();
                    match app.0.rt.lock().unwrap_or_else(|e| e.into_inner()).process(&GrayImage::from_dynamic(&img)) {
                        Ok((objects, _)) => results.push(json!({"objects": objects, "latency_ms": t.elapsed().as_secs_f64() * 1e3})),
                        Err(e) => errors.push(format!("{e:#}")),
                    }
                }
            }
        }
        if results.is_empty() && errors.is_empty() {
            let s = app.status();
            json!({"phase": s.train.phase, "model": s.model_loaded, "enrolled": s.people.iter().map(|p| &p.name).collect::<Vec<_>>(), "metrics": s.train.metrics})
        } else {
            json!({"results": results, "errors": errors})
        }
    })
    .await?;
    Ok(Json(json!({
        "id": format!("chatcmpl-{}", now()), "object": "chat.completion", "created": now(), "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content.to_string()}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })))
}
