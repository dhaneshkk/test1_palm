//! Dashboard: live camera feed, live status (camera, collection, training with loss chart, metrics,
//! enrollment, gallery) and the controls that drive them. Live regions re-render over the
//! runtime WebSocket whenever the shared state changes.

use crate::state::{App, Status};
use base64::Engine as _;
use palmid::lean_mirror::Phase;
use std::time::Duration;
use topcoat::{
    Result,
    context::{Cx, context},
    router::{response::{IntoResponse, Response}, route, Body},
    runtime::{Event, connected, procedure, shard, signal},
    view::{View, emit, live, view},
};

fn report(r: anyhow::Result<String>) -> Result<String> {
    Ok(match r {
        Ok(m) => m,
        Err(e) => format!("⚠ {e:#}"),
    })
}

// ── actions (browser → server) ────────────────────────────────────────────────
#[procedure]
async fn collect(cx: &Cx, frames: String) -> Result<String> {
    let app = context::<App>(cx).clone();
    report(match frames.trim().parse::<usize>() {
        Ok(n) => app.start_collect(n).await,
        Err(_) => Err(anyhow::anyhow!("frames must be a number")),
    })
}

#[procedure]
async fn train(cx: &Cx, epochs: String) -> Result<String> {
    let app = context::<App>(cx);
    report(epochs.trim().parse::<usize>().map_err(|_| anyhow::anyhow!("epochs must be a number")).and_then(|e| app.start_training(e)))
}

#[procedure]
async fn cancel(cx: &Cx) -> Result<String> {
    Ok(if context::<App>(cx).cancel_training() { "training cancelled".into() } else { "no training running".into() })
}

#[procedure]
async fn enroll(cx: &Cx, name: String, consent: bool) -> Result<String> {
    report(context::<App>(cx).start_enroll(name, consent))
}

#[procedure]
async fn forget(cx: &Cx, name: String) -> Result<String> {
    report(context::<App>(cx).delete_person(name.trim()).map(|had| if had { format!("deleted {} and their codes", name.trim()) } else { format!("{} is not enrolled", name.trim()) }))
}

// ── stylesheet (served as a route so CSS needs no escaping) ───────────────────
pub struct Css;
impl IntoResponse for Css {
    fn into_response(self, _cx: &Cx) -> Result<Response> {
        Ok(Response::builder().header("Content-Type", "text/css; charset=utf-8").body(Body::from(include_str!("app.css")))?)
    }
}

#[route(GET "/app.css")]
async fn stylesheet(cx: &Cx) -> Result<Css> {
    Ok(Css)
}

// ── page ──────────────────────────────────────────────────────────────────────
pub async fn dashboard(cx: &Cx) -> Result<impl View> {
    Ok(view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>"palmid — monochrome face & palm"</title>
                <link rel="stylesheet" href="/app.css">
                <link rel="icon" href="data:,">
                topcoat::runtime::script()
            </head>
            <body>
                <header>
                    <h1>"palmid"</h1>
                    <span class="sub">"monochrome face + palm detection · consent-based palmprint ID · CPU"</span>
                </header>
                <main>
                    <section class="feed-col">
                        video()
                        <p class="hint">"Faces: thin white boxes. Palms: thick grey boxes. Everything below runs on the grayscale frame."</p>
                    </section>
                    <section class="side">
                        controls()
                        status()
                    </section>
                </main>
                <footer>"API: POST /v1/chat/completions (OpenAI-compatible, image_url parts) · GET /v1/models"</footer>
            </body>
        </html>
    })
}

#[topcoat::view::component]
async fn controls(cx: &Cx) -> Result<impl View> {
    let frames = signal(cx, || "300".to_string());
    let epochs = signal(cx, || "30".to_string());
    let name = signal(cx, String::new);
    let consent = signal(cx, || false);
    let forget_name = signal(cx, String::new);
    let msg = signal(cx, String::new);
    Ok(view! {
                        <div class="card controls">
                            <h2>"1 · Collect"</h2>
                            <p class="hint">"Colour teachers (YuNet, MediaPipe palm) label frames; grayscale copies become the dataset. Show your face and open palms at different distances, and spend part of the run with no face or palm in view (step aside, hands down) — those negatives are what make MCC meaningful."</p>
                            <label>"frames " <input size="6" :value=$(frames.get()) @input=$(|e: Event| frames.set(e.target.value))></label>
                            <button @click=$(async |_e| { let m = collect(frames.get()).await; msg.set(m); })>"Collect"</button>

                            <h2>"2 · Train"</h2>
                            <p class="hint">"YOLOX-Nano on grayscale with SAM (Python trainer), then per-class τ by split-conformal calibration and MCC in Rust."</p>
                            <label>"epochs " <input size="4" :value=$(epochs.get()) @input=$(|e: Event| epochs.set(e.target.value))></label>
                            <button @click=$(async |_e| { let m = train(epochs.get()).await; msg.set(m); })>"Train"</button>
                            <button class="ghost" @click=$(async |_e| { let m = cancel().await; msg.set(m); })>"Cancel"</button>

                            <h2>"3 · Enroll"</h2>
                            <p class="hint">"Only enroll people who agree. Stored: palm-line codes, never images."</p>
                            <label>"name " <input size="14" :value=$(name.get()) @input=$(|e: Event| name.set(e.target.value))></label>
                            <label class="check"><input type="checkbox" :checked=$(consent.get()) @change=$(|e: Event| consent.set(e.target.checked))>" this person consents"</label>
                            <button @click=$(async |_e| { let m = enroll(name.get(), consent.get()).await; msg.set(m); })>"Enroll"</button>
                            <label>"forget " <input size="14" :value=$(forget_name.get()) @input=$(|e: Event| forget_name.set(e.target.value))></label>
                            <button class="ghost" @click=$(async |_e| { let m = forget(forget_name.get()).await; msg.set(m); })>"Delete"</button>
                            <p class="msg">$(msg.get())</p>
                        </div>
    })
}

// ── live regions ──────────────────────────────────────────────────────────────
#[shard]
async fn video(cx: &Cx) -> Result<impl View> {
    Ok(live! {
        let app = context::<App>(cx).clone();
        let mut frames = app.subscribe_frames();
        loop {
            let jpg = app.frame_jpeg();
            let token = if jpg.is_empty() {
                emit! { <div class="feed empty">"waiting for camera…"</div> }?
            } else {
                let src = format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(&*jpg));
                emit! { <img class="feed" src=(src) alt="live monochrome camera"> }?
            };
            if !connected(cx) {
                break Ok(token);
            }
            frames.changed().await.ok();
            tokio::time::sleep(Duration::from_millis(80)).await; // ≤ ~12 fps to the browser
        }
    })
}

fn phase_label(p: Phase) -> &'static str {
    match p {
        Phase::Idle => "idle — collect data to begin",
        Phase::Collecting => "collecting",
        Phase::Collected => "data collected — ready to train",
        Phase::Training => "training (SAM)",
        Phase::Calibrating => "calibrating τ + MCC",
        Phase::Ready => "model live",
        Phase::Failed => "failed",
    }
}

fn loss_points(v: &[f64]) -> String {
    if v.len() < 2 {
        return String::new();
    }
    let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    let span = (hi - lo).max(1e-9);
    v.iter().enumerate().map(|(i, &x)| format!("{:.1},{:.1}", i as f64 / (v.len() - 1) as f64 * 300.0, 80.0 - (x - lo) / span * 76.0 - 2.0)).collect::<Vec<_>>().join(" ")
}

#[shard]
async fn status(cx: &Cx) -> Result<impl View> {
    Ok(live! {
        let app = context::<App>(cx).clone();
        let mut changed = app.subscribe_status();
        loop {
            let s: Status = app.status();
            let token = emit! { status_panels(s: s) }?;
            if !connected(cx) {
                break Ok(token);
            }
            changed.changed().await.ok();
            tokio::time::sleep(Duration::from_millis(250)).await; // ≤ 4 status renders / s
        }
    })
}

#[topcoat::view::component]
async fn status_panels(cx: &Cx, s: Status) -> Result<impl View> {
    let t = s.train.clone();
    let pct = if t.total > 0 { t.iter * 100 / t.total } else { 0 };
    let collect_pct = if s.collect.target > 0 { s.collect.frames * 100 / s.collect.target } else { 0 };
    let pts = loss_points(&t.iter_losses);
    let phase_class = format!("phase p{}", t.phase as u8);
    Ok(view! {
        <div class="card">
            <h2>"Camera"</h2>
            <div class="kv">
                <span>"source"</span><b>(s.camera.source.clone())</b>
                <span>"state"</span><b>if s.camera.running { "running" } else { "stopped" }</b>
                <span>"fps"</span><b>(format!("{:.1}", s.camera.fps))</b>
                <span>"frame"</span><b>(format!("{:.1} ms", s.camera.frame_ms))</b>
                <span>"model"</span><b>(s.model_loaded.clone().unwrap_or_else(|| "none yet".into()))</b>
            </div>
            if let Some(e) = &s.camera.error { <p class="err">(e.clone())</p> }
            <ul class="objs">
                for o in &s.objects {
                    <li>
                        <b>(o.class.clone())</b> " " (format!("{:.2}", o.score))
                        if let Some(id) = &o.identity { " → " <b class="id">(id.clone())</b> }
                        if let Some(n) = &o.note { " · " <i>(n.clone())</i> }
                    </li>
                }
            </ul>
        </div>
        <div class="card">
            <h2>"Pipeline " <span class=(phase_class)>(phase_label(t.phase))</span></h2>
            if s.collect.active || s.collect.frames > 0 {
                <p>"collect " (s.collect.frames) " / " (s.collect.target) " frames · faces " (s.collect.faces) " · palms " (s.collect.palms)
                   " · train/cal/val " (s.collect.split[0]) "/" (s.collect.split[1]) "/" (s.collect.split[2])</p>
                <div class="bar"><div style=(format!("width:{collect_pct}%"))></div></div>
                if let Some(m) = &s.collect.message { <p class="hint">(m.clone())</p> }
            }
            if t.epochs > 0 {
                <p>"epoch " (t.epoch) " / " (t.epochs) " · iter " (t.iter) " / " (t.total)
                   if let Some(l) = t.last_loss { " · loss " (format!("{l:.3}")) }
                   if let Some(lr) = t.lr { " · lr " (format!("{lr:.5}")) }
                   " · " (format!("{:.0}s", t.elapsed_s))</p>
                <div class="bar"><div style=(format!("width:{pct}%"))></div></div>
                <svg class="chart" viewBox="0 0 300 80" preserveAspectRatio="none" role="img" aria-label="training loss per iteration">
                    <polyline points=(pts) fill="none" stroke="currentColor" stroke-width="1.5"></polyline>
                </svg>
            }
            if let Some(e) = &t.error { <p class="err">(e.clone())</p> }
            if !t.metrics.is_empty() {
                <table>
                    <tr><th>"class"</th><th>"MCC (val)"</th><th>"τ"</th><th>"τ source"</th><th>"n cal"</th><th>"TP/FP/FN/TN"</th></tr>
                    for m in &t.metrics {
                        <tr>
                            <td>(m.class.clone())</td><td><b>(m.mcc.map_or("n/a".to_string(), |v| format!("{v:.3}")))</b></td><td>(format!("{:.3}", m.tau))</td>
                            <td>(m.tau_source.clone())</td><td>(m.n_cal)</td><td>(format!("{}/{}/{}/{}", m.tp, m.fp, m.fn_, m.tn))</td>
                        </tr>
                    }
                </table>
                <p class="hint">"MCC is frame-level presence vs the colour teachers' labels on the val split, not human ground truth. n/a = the split had no frames without that class (or none with it), so MCC is undefined."</p>
            }
            <pre class="log">for line in &t.log { (line.clone()) "\n" }</pre>
        </div>
        <div class="card">
            <h2>"Gallery"</h2>
            if s.enroll.active {
                <p>"enrolling " <b>(s.enroll.name.clone())</b> ": " (s.enroll.codes) " / " (s.enroll.target) " palm codes"
                   if s.enroll.too_small > 0 { " · " (s.enroll.too_small) " frames too small — move closer" }</p>
                <div class="bar"><div style=(format!("width:{}%", s.enroll.codes * 100 / s.enroll.target.max(1)))></div></div>
            }
            if let Some((n, o)) = &s.enroll.last {
                if o.enrolled {
                    <p class="ok">"enrolled " (n.clone()) " · τ " (format!("{:.3}", o.tau)) " vs chance " (format!("{:.3}", o.chance)) " · " (o.source.clone())</p>
                } else {
                    <p class="err">(n.clone()) ": not enrolled — " (o.reason.clone().unwrap_or_default()) " (τ " (format!("{:.3}", o.tau)) ", chance " (format!("{:.3}", o.chance)) ")"</p>
                }
            }
            if s.people.is_empty() {
                <p class="hint">"nobody enrolled; unenrolled palms show as unknown"</p>
            } else {
                <table>
                    <tr><th>"name"</th><th>"τ"</th><th>"chance"</th><th>"n cal"</th></tr>
                    for p in &s.people {
                        <tr><td>(p.name.clone())</td><td>(format!("{:.3}", p.tau))</td><td>(format!("{:.3}", p.chance))</td><td>(p.n_cal)</td></tr>
                    }
                </table>
            }
        </div>
    })
}
