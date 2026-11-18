//! Second-factor verification of a claimed user: 1:1 palm match plus a random Chinese finger-number
//! challenge on one continuously tracked hand, a 5-attempt lockout, and a non-biometric fallback
//! (TOTP, RFC 6238, in any authenticator app).
//!
//! What the challenge does and doesn't stop: a printed palm or a still photo can't perform "7, then 3"
//! on cue, so it fails; a replayed or injected video of the right gestures could pass. This is a
//! webcam-level liveness hint, not certified presentation-attack detection.
//!
//! Lockout (lean/PalmID.lean §9): every failed palm session counts; the 5th consecutive failure locks
//! the user's palm factor for `lockout`. A success resets the count. The fallback code has its own
//! count and lock, so the code can't be brute-forced either. Every event goes to an audit log.

use crate::lean_mirror::accept;
use anyhow::{bail, ensure, Context, Result};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ── policy ──────────────────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize)]
pub struct Policy {
    /// consecutive failures that lock a factor
    pub max_failures: u32,
    /// how long a lock lasts
    pub lockout: Duration,
    /// time to complete a palm session
    pub session_timeout: Duration,
    /// finger numbers in the challenge (after the palm step)
    pub challenge_len: usize,
    /// palm frames that must match the claimed user (1:1) during the palm step
    pub palm_hits: u32,
    /// the hand may leave the view for at most this long once the session started
    pub max_gap: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            max_failures: 5,
            lockout: Duration::from_secs(15 * 60),
            session_timeout: Duration::from_secs(40),
            challenge_len: 2,
            palm_hits: 3,
            max_gap: Duration::from_millis(1500),
        }
    }
}

/// Numbers used in challenges: 5 (open hand) is the palm step itself, so it's left out.
pub const CHALLENGE_NUMBERS: [u8; 9] = [1, 2, 3, 4, 6, 7, 8, 9, 10];

pub fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut b = [0u8; N];
    getrandom::getrandom(&mut b).expect("OS random number generator");
    b
}

/// Uniform pick from 0..n without modulo bias.
fn random_below(n: u32) -> u32 {
    let zone = u32::MAX - (u32::MAX % n);
    loop {
        let v = u32::from_le_bytes(random_bytes::<4>());
        if v < zone {
            return v % n;
        }
    }
}

/// `len` numbers from CHALLENGE_NUMBERS, consecutive ones different (so each must be re-formed).
pub fn random_challenge(len: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(len);
    while out.len() < len {
        let n = CHALLENGE_NUMBERS[random_below(CHALLENGE_NUMBERS.len() as u32) as usize];
        if out.last() != Some(&n) {
            out.push(n);
        }
    }
    out
}

// ── TOTP (RFC 6238, HMAC-SHA1, 6 digits, 30 s) ─────────────────────────────────────────────────────

const B32: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

pub fn base32_encode(data: &[u8]) -> String {
    let (mut out, mut buf, mut bits) = (String::new(), 0u32, 0u32);
    for &b in data {
        buf = (buf << 8) | b as u32;
        bits += 8;
        while bits >= 5 {
            out.push(B32[((buf >> (bits - 5)) & 31) as usize] as char);
            bits -= 5;
        }
    }
    if bits > 0 {
        out.push(B32[((buf << (5 - bits)) & 31) as usize] as char);
    }
    out
}

pub fn base32_decode(s: &str) -> Option<Vec<u8>> {
    let (mut out, mut buf, mut bits) = (vec![], 0u32, 0u32);
    for c in s.chars().filter(|c| !c.is_whitespace() && *c != '=') {
        let v = B32.iter().position(|&x| x as char == c.to_ascii_uppercase())? as u32;
        buf = (buf << 5) | v;
        bits += 5;
        if bits >= 8 {
            out.push((buf >> (bits - 8)) as u8);
            bits -= 8;
        }
    }
    Some(out)
}

/// HOTP value (RFC 4226) for counter `step`, `digits` long.
pub fn hotp(secret: &[u8], step: u64, digits: u32) -> u32 {
    let mut mac = Hmac::<sha1::Sha1>::new_from_slice(secret).expect("any key length");
    mac.update(&step.to_be_bytes());
    let h = mac.finalize().into_bytes();
    let o = (h[19] & 0x0f) as usize;
    let bin = ((h[o] as u32 & 0x7f) << 24) | ((h[o + 1] as u32) << 16) | ((h[o + 2] as u32) << 8) | h[o + 3] as u32;
    bin % 10u32.pow(digits)
}

/// Accept a 6-digit code for the current 30 s step ±1 (clock drift), never twice for the same step.
/// Returns the step that matched.
pub fn totp_check(secret: &[u8], code: &str, now: u64, last_step: Option<u64>) -> Option<u64> {
    let code: u32 = code.trim().parse().ok().filter(|_| code.trim().len() == 6)?;
    let t = now / 30;
    [t.saturating_sub(1), t, t + 1]
        .into_iter()
        .filter(|&s| last_step.is_none_or(|l| s > l))
        .find(|&s| hotp(secret, s, 6) == code)
}

// ── persisted per-user state + audit log ───────────────────────────────────────────────────────────

/// Lockout counter for one factor (mirrors `Lock` in lean/PalmID.lean §9).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Lock {
    pub failures: u32,
    /// unix seconds; attempts are refused while now < locked_until
    pub locked_until: u64,
}

impl Lock {
    pub fn can_attempt(&self, now: u64) -> bool {
        now >= self.locked_until
    }
    pub fn fail(&mut self, now: u64, max: u32, lockout: u64) {
        self.failures += 1;
        if self.failures >= max {
            self.failures = 0;
            self.locked_until = now + lockout;
        }
    }
    pub fn succeed(&mut self) {
        self.failures = 0;
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UserAuth {
    pub palm: Lock,
    pub fallback: Lock,
    /// base32 TOTP secret (None until set up)
    pub totp_secret: Option<String>,
    pub totp_last_step: Option<u64>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct StoreFile {
    users: BTreeMap<String, UserAuth>,
}

pub struct AuthStore {
    path: PathBuf,
    log: PathBuf,
    data: StoreFile,
    pub policy: Policy,
}

#[derive(Clone, Debug, Serialize)]
pub struct AuthResult {
    pub user: String,
    pub ok: bool,
    /// "palm+challenge" or "fallback code"
    pub method: String,
    pub reason: String,
    pub at: u64,
    /// failures left before the factor locks (after this result)
    pub attempts_left: u32,
    pub locked_until: Option<u64>,
}

impl AuthStore {
    /// `path`: JSON state (TOTP secrets inside: keep it private); the audit log goes next to it.
    pub fn open(path: &Path, policy: Policy) -> Result<Self> {
        let data = if path.exists() { serde_json::from_str(&std::fs::read_to_string(path)?).with_context(|| format!("reading {}", path.display()))? } else { StoreFile::default() };
        Ok(Self { log: path.with_extension("log.jsonl"), path: path.to_path_buf(), data, policy })
    }

    /// Pick up changes made by another process (the CLI next to a running web app).
    fn reload(&mut self) {
        if let Ok(t) = std::fs::read_to_string(&self.path) {
            if let Ok(d) = serde_json::from_str(&t) {
                self.data = d;
            }
        }
    }

    fn save(&self) -> Result<()> {
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&self.data)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    pub fn audit(&self, user: &str, event: &str, detail: &str) {
        let line = serde_json::json!({"t": now_secs(), "user": user, "event": event, "detail": detail});
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.log) {
            let _ = writeln!(f, "{line}");
        }
    }

    pub fn user(&self, name: &str) -> UserAuth {
        self.data.users.get(name).cloned().unwrap_or_default()
    }

    pub fn users(&self) -> &BTreeMap<String, UserAuth> {
        &self.data.users
    }

    /// Refuse to start a palm session while the user's palm factor is locked.
    pub fn check_palm_allowed(&mut self, name: &str, now: u64) -> Result<()> {
        self.reload();
        let u = self.user(name);
        if !u.palm.can_attempt(now) {
            bail!("palm verification for {name} is locked for {} more seconds (too many failures); use the fallback code or ask an administrator", u.palm.locked_until - now);
        }
        Ok(())
    }

    /// Record a finished palm session.
    pub fn record_palm(&mut self, name: &str, ok: bool, reason: &str) -> Result<AuthResult> {
        self.reload();
        let now = now_secs();
        let (max, lock) = (self.policy.max_failures, self.policy.lockout.as_secs());
        let u = self.data.users.entry(name.to_string()).or_default();
        if ok {
            u.palm.succeed();
            u.fallback.succeed();
        } else {
            u.palm.fail(now, max, lock);
        }
        let r = AuthResult {
            user: name.into(),
            ok,
            method: "palm+challenge".into(),
            reason: reason.into(),
            at: now,
            attempts_left: max - u.palm.failures,
            locked_until: (!u.palm.can_attempt(now)).then_some(u.palm.locked_until),
        };
        self.save()?;
        self.audit(name, if ok { "palm_ok" } else { "palm_fail" }, reason);
        Ok(r)
    }

    /// New TOTP secret for `name` (replaces any old one). Returns (base32 secret, otpauth URI).
    pub fn totp_setup(&mut self, name: &str, issuer: &str) -> Result<(String, String)> {
        self.reload();
        ensure!(!name.trim().is_empty(), "name is empty");
        let secret = base32_encode(&random_bytes::<20>());
        let u = self.data.users.entry(name.to_string()).or_default();
        u.totp_secret = Some(secret.clone());
        u.totp_last_step = None;
        self.save()?;
        self.audit(name, "totp_setup", "");
        let enc = |s: &str| s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect::<String>();
        let uri = format!("otpauth://totp/{}:{}?secret={secret}&issuer={}&algorithm=SHA1&digits=6&period=30", enc(issuer), enc(name), enc(issuer));
        Ok((secret, uri))
    }

    /// Non-biometric fallback: a 6-digit TOTP code. Has its own 5-attempt lockout.
    pub fn fallback(&mut self, name: &str, code: &str) -> Result<AuthResult> {
        self.reload();
        let now = now_secs();
        let (max, lock) = (self.policy.max_failures, self.policy.lockout.as_secs());
        let u = self.data.users.get_mut(name).filter(|u| u.totp_secret.is_some()).ok_or_else(|| anyhow::anyhow!("no fallback code is set up for {name}"))?;
        if !u.fallback.can_attempt(now) {
            bail!("fallback for {name} is locked for {} more seconds (too many wrong codes)", u.fallback.locked_until - now);
        }
        let secret = base32_decode(u.totp_secret.as_deref().unwrap()).context("stored secret is not base32")?;
        let ok = match totp_check(&secret, code, now, u.totp_last_step) {
            Some(step) => {
                u.totp_last_step = Some(step);
                u.fallback.succeed();
                u.palm.succeed();
                true
            }
            None => {
                u.fallback.fail(now, max, lock);
                false
            }
        };
        let r = AuthResult {
            user: name.into(),
            ok,
            method: "fallback code".into(),
            reason: if ok { "code accepted".into() } else { "wrong, expired or already used code".into() },
            at: now,
            attempts_left: max - u.fallback.failures,
            locked_until: (!u.fallback.can_attempt(now)).then_some(u.fallback.locked_until),
        };
        self.save()?;
        self.audit(name, if ok { "fallback_ok" } else { "fallback_fail" }, "");
        Ok(r)
    }

    /// Administrator unlock: clears both counters and locks.
    pub fn unlock(&mut self, name: &str) -> Result<bool> {
        self.reload();
        let Some(u) = self.data.users.get_mut(name) else { return Ok(false) };
        u.palm = Lock::default();
        u.fallback = Lock::default();
        self.save()?;
        self.audit(name, "unlock", "administrator");
        Ok(true)
    }

    pub fn remove(&mut self, name: &str) -> Result<()> {
        self.reload();
        if self.data.users.remove(name).is_some() {
            self.save()?;
            self.audit(name, "removed", "");
        }
        Ok(())
    }
}

// ── one palm session: palm step, then the finger numbers, on one tracked hand ────────────────────────

#[derive(Clone, Debug, Serialize, PartialEq)]
pub enum Step {
    /// open hand whose palm matches the claimed user (1:1)
    Palm,
    /// hold this Chinese finger number
    Finger(u8),
}

impl Step {
    pub fn label(&self) -> String {
        match self {
            Step::Palm => "show your open palm".into(),
            Step::Finger(n) => format!("show {n}"),
        }
    }
}

/// What the camera saw in one frame, for the session.
#[derive(Clone, Copy, Debug, Default)]
pub struct Observation {
    /// a hand was seen (detected or followed by landmarks)
    pub hand: bool,
    /// finger number held steady (stabiliser output)
    pub held: Option<u8>,
    /// palm codes compared with the claimed user: Some(true) if within their threshold
    pub palm_match: Option<bool>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub enum Outcome {
    Running,
    Passed,
    Failed(String),
}

#[derive(Clone, Debug, Serialize)]
pub struct Session {
    pub id: String,
    pub user: String,
    pub steps: Vec<Step>,
    /// index of the current step (== steps.len() when passed)
    pub at: usize,
    pub palm_hits: u32,
    pub palm_misses: u32,
    pub outcome: Outcome,
    #[serde(skip)]
    started: std::time::Instant,
    #[serde(skip)]
    last_hand: Option<std::time::Instant>,
    #[serde(skip)]
    policy: Policy,
    /// a finger step only counts once the previous number was let go (no "held over" from before)
    #[serde(skip)]
    need_release: bool,
}

impl Session {
    pub fn new(user: &str, policy: Policy) -> Self {
        let mut steps = vec![Step::Palm];
        steps.extend(random_challenge(policy.challenge_len).into_iter().map(Step::Finger));
        Self {
            id: random_bytes::<12>().iter().map(|b| format!("{b:02x}")).collect(),
            user: user.into(),
            steps,
            at: 0,
            palm_hits: 0,
            palm_misses: 0,
            outcome: Outcome::Running,
            started: std::time::Instant::now(),
            last_hand: None,
            policy,
            need_release: false,
        }
    }

    pub fn seconds_left(&self) -> u64 {
        self.policy.session_timeout.saturating_sub(self.started.elapsed()).as_secs()
    }

    /// Advance with one frame; steps must be done in order on a hand that stays in view.
    pub fn observe(&mut self, o: Observation) -> &Outcome {
        self.observe_at(o, std::time::Instant::now())
    }

    pub fn observe_at(&mut self, o: Observation, now: std::time::Instant) -> &Outcome {
        if self.outcome != Outcome::Running {
            return &self.outcome;
        }
        if now.duration_since(self.started) > self.policy.session_timeout {
            self.outcome = Outcome::Failed(format!("timed out at “{}”", self.steps[self.at].label()));
            return &self.outcome;
        }
        if o.hand {
            self.last_hand = Some(now);
        } else if let Some(t) = self.last_hand {
            if now.duration_since(t) > self.policy.max_gap {
                self.outcome = Outcome::Failed("the hand left the view during the challenge".into());
            }
            return &self.outcome;
        }
        match self.steps[self.at] {
            Step::Palm => match o.palm_match {
                Some(true) => {
                    self.palm_hits += 1;
                    if self.palm_hits >= self.policy.palm_hits {
                        self.at += 1;
                        self.need_release = true;
                    }
                }
                Some(false) => self.palm_misses += 1,
                None => {}
            },
            Step::Finger(n) => {
                if self.need_release {
                    // the previous shape must be let go before the next one counts
                    let prev = match self.steps[self.at - 1] { Step::Palm => Some(5), Step::Finger(m) => Some(m) };
                    if o.held != prev {
                        self.need_release = false;
                    }
                }
                if !self.need_release && o.held == Some(n) {
                    self.at += 1;
                    self.need_release = true;
                }
            }
        }
        if self.at == self.steps.len() {
            self.outcome = Outcome::Passed;
        }
        &self.outcome
    }
}

/// 1:1 decision for one frame from the gallery's per-person distances.
pub fn palm_match(candidates: &[crate::gallery::Candidate], user: &str) -> Option<bool> {
    candidates.iter().find(|c| c.name == user).map(|c| accept(c.dist, c.tau))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc6238_and_rfc4226_vectors() {
        // RFC 4226 appendix D: secret "12345678901234567890", 6 digits
        let k = b"12345678901234567890";
        let want = [755224, 287082, 359152, 969429, 338314, 254676, 287922, 162583, 399871, 520489];
        for (i, w) in want.iter().enumerate() {
            assert_eq!(hotp(k, i as u64, 6), *w);
        }
        // RFC 6238 appendix B (SHA-1, 8 digits): T = 59 s → 94287082, T = 1111111109 → 07081804
        assert_eq!(hotp(k, 59 / 30, 8), 94287082);
        assert_eq!(hotp(k, 1111111109 / 30, 8), 7081804);
    }

    #[test]
    fn totp_window_and_no_replay() {
        let k = b"12345678901234567890";
        let now = 1_700_000_000;
        let code = format!("{:06}", hotp(k, now / 30, 6));
        let s = totp_check(k, &code, now, None).unwrap();
        assert_eq!(totp_check(k, &code, now, Some(s)), None, "same code twice");
        let next = format!("{:06}", hotp(k, now / 30 + 1, 6));
        assert!(totp_check(k, &next, now, None).is_some(), "one step of drift");
        let far = format!("{:06}", hotp(k, now / 30 + 3, 6));
        assert!(totp_check(k, &far, now, None).is_none());
        assert!(totp_check(k, "12345", now, None).is_none());
    }

    #[test]
    fn base32_round_trip() {
        for n in 0..40 {
            let d: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(base32_decode(&base32_encode(&d)).unwrap(), d);
        }
        assert_eq!(base32_encode(b"foobar"), "MZXW6YTBOI");
    }

    #[test]
    fn challenge_shape() {
        for _ in 0..500 {
            let c = random_challenge(3);
            assert_eq!(c.len(), 3);
            assert!(c.iter().all(|n| CHALLENGE_NUMBERS.contains(n)));
            assert!(c.windows(2).all(|w| w[0] != w[1]));
        }
    }

    #[test]
    fn lockout_after_five() {
        let mut l = Lock::default();
        for i in 0..4 {
            l.fail(100, 5, 900);
            assert!(l.can_attempt(100), "after {} failures", i + 1);
        }
        l.fail(100, 5, 900);
        assert!(!l.can_attempt(100) && !l.can_attempt(999) && l.can_attempt(1000));
        assert_eq!(l.failures, 0);
        let mut l = Lock::default();
        for _ in 0..4 { l.fail(5, 5, 900); }
        l.succeed();
        for _ in 0..4 { l.fail(5, 5, 900); }
        assert!(l.can_attempt(5), "a success resets the count");
    }

    /// Mirror of lean/PalmID.lean §9 (`failN_counts`, `max_failures_lock`) on a grid.
    #[test]
    fn lean_lockout_statements() {
        for max in 1..8u32 {
            for lock in [0u64, 1, 60, 900] {
                let mut l = Lock::default();
                for k in 1..=max {
                    l.fail(100, max, lock);
                    if k < max {
                        assert_eq!(l, Lock { failures: k, locked_until: 0 });
                    }
                }
                for t in 0..(100 + lock + 3) {
                    assert_eq!(l.can_attempt(t), t >= 100 + lock, "max {max} L {lock} t {t}");
                }
            }
        }
    }

    fn sess(steps: Vec<Step>) -> Session {
        let mut s = Session::new("asha", Policy::default());
        s.steps = steps;
        s
    }

    fn obs(hand: bool, held: Option<u8>, palm: Option<bool>) -> Observation {
        Observation { hand, held, palm_match: palm }
    }

    #[test]
    fn session_passes_only_in_order_on_one_hand() {
        let t0 = std::time::Instant::now();
        let ms = |m| t0 + Duration::from_millis(m);
        let mut s = sess(vec![Step::Palm, Step::Finger(7), Step::Finger(3)]);
        // showing 7 before the palm does nothing
        s.observe_at(obs(true, Some(7), None), ms(100));
        assert_eq!(s.at, 0);
        for k in 0..3 { s.observe_at(obs(true, Some(5), Some(true)), ms(200 + k * 100)); }
        assert_eq!(s.at, 1);
        // 3 before 7: no progress
        s.observe_at(obs(true, Some(3), None), ms(600));
        assert_eq!(s.at, 1);
        s.observe_at(obs(true, Some(7), None), ms(700));
        assert_eq!(s.at, 2);
        s.observe_at(obs(true, Some(3), None), ms(800));
        assert_eq!(s.outcome, Outcome::Passed);
    }

    #[test]
    fn session_needs_release_between_equal_shapes() {
        let t0 = std::time::Instant::now();
        let mut s = sess(vec![Step::Palm, Step::Finger(7), Step::Finger(7)]);
        for k in 0..3 { s.observe_at(obs(true, Some(5), Some(true)), t0 + Duration::from_millis(k * 100)); }
        s.observe_at(obs(true, Some(7), None), t0 + Duration::from_millis(400));
        s.observe_at(obs(true, Some(7), None), t0 + Duration::from_millis(500));
        assert_eq!(s.at, 2, "holding 7 must not satisfy the next 7");
        s.observe_at(obs(true, None, None), t0 + Duration::from_millis(600));
        s.observe_at(obs(true, Some(7), None), t0 + Duration::from_millis(700));
        assert_eq!(s.outcome, Outcome::Passed);
    }

    #[test]
    fn session_fails_on_gap_timeout_and_wrong_palm() {
        let t0 = std::time::Instant::now();
        let mut s = sess(vec![Step::Palm, Step::Finger(2)]);
        s.observe_at(obs(true, Some(5), Some(true)), t0);
        s.observe_at(obs(false, None, None), t0 + Duration::from_millis(3000));
        assert!(matches!(s.outcome, Outcome::Failed(ref r) if r.contains("left the view")));
        let mut s = sess(vec![Step::Palm, Step::Finger(2)]);
        for k in 0..50 { s.observe_at(obs(true, Some(5), Some(false)), t0 + Duration::from_millis(k * 100)); }
        assert_eq!(s.at, 0, "someone else's palm never passes");
        s.observe_at(obs(true, Some(5), Some(false)), t0 + Duration::from_secs(41));
        assert!(matches!(s.outcome, Outcome::Failed(ref r) if r.contains("timed out")));
    }
}
