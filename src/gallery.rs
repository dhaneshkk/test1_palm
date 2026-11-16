//! Consent-based enrollment and open-set identification.
//! * Only people enrolled with explicit consent are in the gallery; anyone else is `unknown`.
//! * Stored: CompCode orientation grids (3-bit codes) + per-person threshold. Never images.
//! * Per-person τ = conformal quantile of genuine distances (Lean §3, needs ≥ 29 calibration codes).
//! * Chance gate: if τ is not clearly below the distance to structure-free (shuffled) codes, the palm
//!   texture was not resolved and a threshold would accept everyone — enrollment is refused.

use crate::lean_mirror::{accept, closer, conf_rank, step, N_CAL_FORMAL_BOUND};
use crate::palmprint::{code_distance, Code, PalmCodes, RoiMode};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Identity {
    pub templates: Vec<String>, // base-6 digit strings, one char per cell
    pub code_h: usize,
    pub code_w: usize,
    pub tau: f64,
    pub chance: f64,
    pub n_cal: usize,
    pub source: String,
    /// ROI cut used at enrollment; probes are coded the same way. Older galleries default to `box`.
    #[serde(default)]
    pub roi: RoiMode,
}

#[derive(Clone, Debug, Serialize)]
pub struct EnrollOutcome {
    pub enrolled: bool,
    pub reason: Option<String>,
    pub tau: f64,
    pub chance: f64,
    pub n_cal: usize,
    pub genuine_median: Option<f64>,
    pub source: String,
    pub roi: RoiMode,
}

#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub name: String,
    pub dist: f64,
    pub tau: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Identification {
    pub identity: String,
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GalleryConfig {
    pub gallery_templates: usize,
    pub id_margin: f64,
    pub shift: isize,
}
impl Default for GalleryConfig {
    fn default() -> Self {
        Self { gallery_templates: 10, id_margin: 0.06, shift: 2 }
    }
}

pub struct Gallery {
    pub path: PathBuf,
    pub cfg: GalleryConfig,
    pub people: BTreeMap<String, Identity>,
}

fn encode(c: &Code) -> String {
    c.cells.iter().map(|&v| (b'0' + v) as char).collect()
}
fn decode(s: &str, h: usize, w: usize) -> Code {
    Code { h, w, cells: s.bytes().map(|b| b - b'0').collect() }
}

/// splitmix64 — deterministic shuffles for the chance estimate (no external RNG crate).
struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

impl Gallery {
    pub fn open(path: &Path, cfg: GalleryConfig) -> Result<Self> {
        let people = if path.exists() { serde_json::from_str(&std::fs::read_to_string(path)?)? } else { BTreeMap::new() };
        Ok(Self { path: path.to_path_buf(), cfg, people })
    }

    pub fn save(&self) -> Result<()> {
        std::fs::write(&self.path, serde_json::to_string_pretty(&self.people)?)?;
        Ok(())
    }

    pub fn delete(&mut self, name: &str) -> Result<bool> {
        let had = self.people.remove(name).is_some();
        self.save()?;
        Ok(had)
    }

    /// Which ROI modes the enrolled people need: (box, landmarks).
    pub fn modes_needed(&self) -> (bool, bool) {
        (self.people.values().any(|p| p.roi == RoiMode::Box), self.people.values().any(|p| p.roi == RoiMode::Landmarks))
    }

    pub fn templates(&self, id: &Identity) -> Vec<Code> {
        id.templates.iter().map(|s| decode(s, id.code_h, id.code_w)).collect()
    }

    /// `consent` must be true: enrollment is only for people who agreed to it.
    pub fn enroll(&mut self, name: &str, codes: &[Code], consent: bool) -> Result<EnrollOutcome> {
        self.enroll_with(name, codes, consent, RoiMode::Box)
    }

    /// Enroll with codes all cut by `roi` (box or landmark-aligned).
    pub fn enroll_with(&mut self, name: &str, codes: &[Code], consent: bool, roi: RoiMode) -> Result<EnrollOutcome> {
        if !consent {
            bail!("enrollment requires the person's explicit consent");
        }
        let g = self.cfg.gallery_templates;
        if codes.len() <= g {
            bail!("need more than {g} palm codes (got {})", codes.len());
        }
        let (gal, cal) = codes.split_at(g);
        let mut genuine: Vec<f64> = cal
            .iter()
            .map(|c| gal.iter().map(|t| code_distance(c, t, self.cfg.shift)).fold(1.0, f64::min))
            .collect();
        genuine.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = genuine.len();
        let (tau, source) = if n >= N_CAL_FORMAL_BOUND {
            (genuine[conf_rank(n) - 1], "sccp".to_string())
        } else {
            (*genuine.last().unwrap(), format!("max-genuine(n={n}<{N_CAL_FORMAL_BOUND}, no guarantee)"))
        };
        let mut rng = SplitMix(0);
        let probes: Vec<&Code> = cal.iter().take(10).collect();
        let mut chance_d: Vec<f64> = probes
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut shuffled = gal[i % gal.len()].clone();
                rng.shuffle(&mut shuffled.cells);
                code_distance(c, &shuffled, self.cfg.shift)
            })
            .collect();
        let chance = median(&mut chance_d);
        let genuine_median = Some(genuine[n / 2]);
        if tau > chance - self.cfg.id_margin {
            return Ok(EnrollOutcome {
                enrolled: false,
                reason: Some("palm texture not resolved (move closer / more light / sharper focus)".into()),
                tau, chance, n_cal: n, genuine_median, source, roi,
            });
        }
        self.people.insert(
            name.to_string(),
            Identity { templates: gal.iter().map(encode).collect(), code_h: gal[0].h, code_w: gal[0].w, tau, chance, n_cal: n, source: source.clone(), roi },
        );
        self.save()?;
        Ok(EnrollOutcome { enrolled: true, reason: None, tau, chance, n_cal: n, genuine_median, source, roi })
    }

    /// Open-set: accept person p iff d_p ≤ τ_p; among accepted, smallest d_p/τ_p wins (division-free).
    pub fn identify(&self, code: &Code) -> Identification {
        self.identify_codes(&PalmCodes { boxed: Some(code.clone()), aligned: Some(code.clone()) })
    }

    /// As `identify`, comparing each person with the probe code cut the way they were enrolled.
    /// People whose ROI mode could not be computed for this frame are skipped.
    pub fn identify_codes(&self, codes: &PalmCodes) -> Identification {
        let mut best: Option<(String, f64, f64)> = None;
        let mut candidates = Vec::new();
        for (name, id) in &self.people {
            let Some(code) = codes.get(id.roi) else { continue };
            let d = self.templates(id).iter().map(|t| code_distance(code, t, self.cfg.shift)).fold(1.0, f64::min);
            candidates.push(Candidate { name: name.clone(), dist: d, tau: id.tau });
            if accept(d, id.tau) && best.as_ref().map_or(true, |(_, bd, bt)| closer(d, id.tau, *bd, *bt)) {
                best = Some((name.clone(), d, id.tau));
            }
        }
        Identification { identity: best.map(|b| b.0).unwrap_or_else(|| "unknown".into()), candidates }
    }
}

/// K-of-N announcement debounce (Lean §4): an identity is announced only after ≥ K supporting frames.
#[derive(Default)]
pub struct Debounce {
    counts: BTreeMap<String, u32>,
}
impl Debounce {
    pub fn update(&mut self, seen: &str, n: u32, k: u32) -> bool {
        for (key, c) in self.counts.iter_mut() {
            if key != seen {
                *c = step(n, *c, false);
            }
        }
        let c = self.counts.entry(seen.to_string()).or_insert(0);
        *c = step(n, *c, true);
        *c >= k
    }
}
