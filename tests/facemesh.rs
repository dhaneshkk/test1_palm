//! Delaunay vs scipy (committed reference); face mesh vs Python (local image only:
//! PALMID_FACEMESH_CASE=<gray.png>|<ref.json>|<x1,y1,x2,y2>, refs from tools/facemesh_ref.py;
//! PALMID_FACEMESH_OUT=<png> also writes the drawn mesh).

use palmid::facemesh::{delaunay, FaceMesher};
use palmid::imgproc::GrayImage;
use std::path::Path;

#[test]
fn delaunay_matches_scipy() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/delaunay_ref.json");
    let cases: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    for c in cases {
        let pts: Vec<(f64, f64)> = c["points"].as_array().unwrap().iter().map(|q| (q[0].as_f64().unwrap(), q[1].as_f64().unwrap())).collect();
        let want: Vec<[usize; 3]> = c["triangles"].as_array().unwrap().iter().map(|t| [t[0].as_u64().unwrap() as usize, t[1].as_u64().unwrap() as usize, t[2].as_u64().unwrap() as usize]).collect();
        let got = delaunay(&pts);
        eprintln!("{} points: {} triangles (scipy {})", pts.len(), got.len(), want.len());
        assert_eq!(got, want);
    }
}

#[test]
fn face_mesh_matches_python() {
    let (Some(_), Ok(case)) = (std::env::var_os("ORT_DYLIB_PATH"), std::env::var("PALMID_FACEMESH_CASE")) else {
        eprintln!("skipped: set ORT_DYLIB_PATH and PALMID_FACEMESH_CASE");
        return;
    };
    let parts: Vec<&str> = case.split('|').collect();
    let i = image::open(parts[0]).unwrap().to_luma8();
    let g = GrayImage { w: i.width() as usize, h: i.height() as usize, data: i.into_raw() };
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(parts[1]).unwrap()).unwrap();
    let b: Vec<f32> = parts[2].split(',').map(|v| v.trim().parse().unwrap()).collect();
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models/face_landmark.onnx");
    let mut fm = FaceMesher::new(&model, 2).unwrap();
    let mesh = fm.mesh(&g, [b[0], b[1], b[2], b[3]]).unwrap().expect("face");
    let worst = r["points"].as_array().unwrap().iter().enumerate().map(|(k, p)| {
        ((mesh.points[k][0] - p[0].as_f64().unwrap() as f32).powi(2) + (mesh.points[k][1] - p[1].as_f64().unwrap() as f32).powi(2)).sqrt()
    }).fold(0f32, f32::max);
    let tri = delaunay(&mesh.points.iter().map(|p| (p[0] as f64, p[1] as f64)).collect::<Vec<_>>());
    eprintln!("468 points: worst {worst:.4} px, presence {:.4}; triangles {} (python {})", mesh.presence, tri.len(), r["triangles"].as_array().unwrap().len());
    if let Ok(out) = std::env::var("PALMID_FACEMESH_OUT") {
        let t = std::time::Instant::now();
        let mut vis = g.clone();
        palmid::facemesh::draw_mesh(&mut vis, &mesh, 150, 255);
        eprintln!("triangulate + draw {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        image::GrayImage::from_raw(vis.w as u32, vis.h as u32, vis.data).unwrap().save(out).unwrap();
    }
    assert!(worst < 0.01);
}
