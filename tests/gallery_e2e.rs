//! End-to-end enrollment / open-set identification on synthetic palm-line images.
//! Mechanics check only (synthetic textures are not a biometric benchmark).

use palmid::gallery::{Gallery, GalleryConfig};
use palmid::imgproc::GrayImage;
use palmid::palmprint::{Code, PalmCoder, PalmParams};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A "palm": a few principal lines + many fine creases (each a rotated sinusoid), fixed per identity.
fn identity(seed: u64) -> Vec<[f32; 6]> {
    let mut r = Rng(seed * 2654435761 + 1);
    (0..14)
        .map(|i| {
            let major = i < 3;
            [r.f() * 128.0, (r.f() - 0.5) * 3.0, 0.02 + r.f() * 0.06, r.f() * 6.28, if major { 4.0 + r.f() * 6.0 } else { 1.5 + r.f() * 3.0 }, if major { 2.2 } else { 1.0 }]
        })
        .collect()
}

/// Render one capture: small translation, brightness/contrast change and sensor noise.
fn render(lines: &[[f32; 6]], sample: u64) -> GrayImage {
    let mut r = Rng(sample * 97 + 13);
    let (dx, dy) = ((r.f() - 0.5) * 3.0, (r.f() - 0.5) * 3.0);
    let (gain, bias) = (0.8 + r.f() * 0.4, (r.f() - 0.5) * 30.0);
    let mut img = GrayImage::new(128, 128, 0);
    for y in 0..128 {
        for x in 0..128 {
            let (xf, yf) = (x as f32 + dx, y as f32 + dy);
            let mut v = 170.0f32;
            for l in lines {
                let (y0, slope, freq, phase, amp, width) = (l[0], l[1], l[2], l[3], l[4], l[5]);
                let curve = y0 + slope * (xf - 64.0) + amp * (freq * xf + phase).sin();
                let d = (yf - curve).abs();
                v -= 70.0 * (-(d * d) / (2.0 * width * width)).exp();
            }
            let noise = (r.f() - 0.5) * 16.0;
            img.data[y * 128 + x] = (v * gain + bias + noise).clamp(0.0, 255.0) as u8;
        }
    }
    img
}

fn codes(coder: &PalmCoder, lines: &[[f32; 6]], from: u64, n: u64) -> Vec<Code> {
    (from..from + n).map(|s| coder.code(&render(lines, s))).collect()
}

#[test]
fn enroll_identify_reject() {
    let coder = PalmCoder::new(PalmParams::default());
    let path = std::env::temp_dir().join(format!("palmid_e2e_{}.json", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let mut g = Gallery::open(&path, GalleryConfig::default()).unwrap();
    let (a, b, c) = (identity(1), identity(2), identity(3));

    assert!(g.enroll("x", &codes(&coder, &a, 0, 40), false).is_err(), "no consent → refused");
    let ea = g.enroll("alice", &codes(&coder, &a, 0, 40), true).unwrap();
    let eb = g.enroll("bob", &codes(&coder, &b, 1000, 40), true).unwrap();
    eprintln!("alice {ea:?}\nbob {eb:?}");
    assert!(ea.enrolled && eb.enrolled);
    assert_eq!(ea.n_cal, 30);
    assert_eq!(ea.source, "sccp");

    let (mut tp_a, mut tp_b, mut fa) = (0, 0, 0);
    for s in 0..20u64 {
        tp_a += (g.identify(&coder.code(&render(&a, 500 + s))).identity == "alice") as u32;
        tp_b += (g.identify(&coder.code(&render(&b, 1500 + s))).identity == "bob") as u32;
        fa += (g.identify(&coder.code(&render(&c, 2500 + s))).identity != "unknown") as u32;
    }
    eprintln!("alice {tp_a}/20  bob {tp_b}/20  unenrolled accepted {fa}/20");
    assert!(tp_a >= 17 && tp_b >= 17, "genuine probes should be accepted (≈ 1 − α)");
    assert_eq!(fa, 0, "an unenrolled palm must come back unknown");

    // persistence + deletion
    let g2 = Gallery::open(&path, GalleryConfig::default()).unwrap();
    assert_eq!(g2.people.len(), 2);
    let mut g3 = g2;
    assert!(g3.delete("alice").unwrap());
    assert_eq!(Gallery::open(&path, GalleryConfig::default()).unwrap().people.len(), 1);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn noise_only_palm_is_refused() {
    // Pure noise = unresolved texture: genuine distances ≈ chance, so the gate must refuse.
    let coder = PalmCoder::new(PalmParams::default());
    let path = std::env::temp_dir().join(format!("palmid_noise_{}.json", std::process::id()));
    let mut g = Gallery::open(&path, GalleryConfig::default()).unwrap();
    let mut r = Rng(42);
    let noise: Vec<Code> = (0..40)
        .map(|_| {
            let mut img = GrayImage::new(128, 128, 0);
            img.data.iter_mut().for_each(|v| *v = (r.f() * 255.0) as u8);
            coder.code(&img)
        })
        .collect();
    let out = g.enroll("noise", &noise, true).unwrap();
    eprintln!("noise {out:?}");
    assert!(!out.enrolled);
    assert!(g.people.is_empty());
    let _ = std::fs::remove_file(&path);
}
