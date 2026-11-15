//! palmid-web — Topcoat web app: collect (colour teachers) → train (SAM, Python) → calibrate + MCC (Rust)
//! → live monochrome inference and consent-based palmprint ID, with live status over WebSocket.

mod api;
mod state;
mod ui;

use state::{App, Config};
use topcoat::{
    asset::{AssetBundle, RouterBuilderAssetExt},
    router::{RouterBuilderDiscoverExt, module_router},
    runtime::RuntimeLayer,
};

#[tokio::main]
async fn main() {
    let cfg = Config::from_env();
    eprintln!("[palmid-web] work dir {} · source {} · trainer {} {}", cfg.work.display(), cfg.source, cfg.python, cfg.trainer.display());
    let app = App::new(cfg).expect("app state");
    app.spawn_camera();
    let assets = AssetBundle::load().expect("asset bundle missing: run `topcoat asset bundle -p palmid-web` (or `topcoat dev`)");
    topcoat::start(module_router!().assets(assets).with(app).discover().layer(RuntimeLayer::new()).build())
        .await
        .unwrap();
}

#[topcoat::router::page]
async fn page(cx: &topcoat::context::Cx) -> topcoat::Result<impl topcoat::view::View> {
    ui::dashboard(cx).await
}
