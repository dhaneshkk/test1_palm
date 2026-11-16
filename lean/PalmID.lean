/-
VG360 mono face/palm + palmprint ID: formal guarantees (Lean 4.31, kernel-only: Nat + omega, no Mathlib).
CompCode: each code cell holds a dominant line orientation o ∈ {0..5} (6 Gabor filters, 30° apart).
-/

-- §1 Angular distance between two orientation codes on the 6-cycle.
def absDiff (a b : Nat) : Nat := if a ≤ b then b - a else a - b
def angDist (a b : Nat) : Nat := if absDiff a b ≤ 3 then absDiff a b else 6 - absDiff a b

theorem absDiff_symm (a b : Nat) : absDiff a b = absDiff b a := by
  unfold absDiff; split <;> split <;> omega

theorem angDist_symm (a b : Nat) : angDist a b = angDist b a := by
  unfold angDist; rw [absDiff_symm a b]

theorem angDist_self (a : Nat) : angDist a a = 0 := by
  unfold angDist absDiff; simp

/-- Per-cell distance never exceeds 3 for valid codes, so a template of n cells has total ≤ 3n:
    normalising by 3n gives a score in [0,1]. -/
theorem angDist_le_three (a b : Nat) (ha : a < 6) (hb : b < 6) : angDist a b ≤ 3 := by
  have : absDiff a b ≤ 5 := by unfold absDiff; split <;> omega
  unfold angDist; split <;> omega

def total : List (Nat × Nat) → Nat
  | [] => 0
  | (a, b) :: t => angDist a b + total t

theorem total_le (l : List (Nat × Nat)) (h : ∀ p ∈ l, p.1 < 6 ∧ p.2 < 6) : total l ≤ 3 * l.length := by
  induction l with
  | nil => simp [total]
  | cons p t ih =>
    have hp := h p (by simp)
    have ht : ∀ q ∈ t, q.1 < 6 ∧ q.2 < 6 := fun q hq => h q (by simp [hq])
    have := angDist_le_three p.1 p.2 hp.1 hp.2
    have := ih ht
    simp [total, List.length]; omega

-- §2 Open-set accept decision (scaled integers: dist·D ≤ τ·D, cross-multiplied, no division).
def accept (dist tau : Nat) : Prop := dist ≤ tau

/-- A looser threshold never rejects a probe that a stricter one accepted. -/
theorem accept_monotone (d t t' : Nat) (h : t ≤ t') (ha : accept d t) : accept d t' := by
  unfold accept at *; omega

/-- Best-of-gallery matching: if any template is accepted, the minimum distance is accepted. -/
theorem min_accept (d₁ d₂ t : Nat) (h : accept d₁ t ∨ accept d₂ t) : accept (min d₁ d₂) t := by
  unfold accept at *; omega

/-- Identity disambiguation uses dist·τ_other < dist_other·τ (ratio comparison, division-free):
    it is irreflexive, so a probe is never ranked strictly closer to a person than to itself. -/
def closerRel (d t d' t' : Nat) : Prop := d * t' < d' * t
theorem closerRel_irrefl (d t : Nat) : ¬ closerRel d t d t := by
  unfold closerRel; omega

-- §3 Mondrian split-conformal per identity / per class at α = 1/30:
--     rank k = ⌈(n+1)(1−α)⌉ = ((n+1)·29 + 29)/30 must satisfy k ≤ n.
def confRank (n : Nat) : Nat := ((n + 1) * 29 + 29) / 30
theorem conformal_feasible (n : Nat) (h : 29 ≤ n) : confRank n ≤ n := by
  unfold confRank; omega
theorem conformal_tight : confRank 28 > 28 := by decide

-- §4 K-of-N debounce for identity announcements (saturating counter).
def step (N cnt : Nat) (hit : Bool) : Nat := if hit then min (cnt + 1) N else cnt - 1
theorem step_bounded (N cnt : Nat) (hit : Bool) (h : cnt ≤ N) : step N cnt hit ≤ N := by
  unfold step; cases hit <;> simp <;> omega
theorem step_lipschitz (N cnt : Nat) (hit : Bool) : step N cnt hit ≤ cnt + 1 := by
  unfold step; cases hit <;> simp <;> omega
def run (N : Nat) : Nat → List Bool → Nat
  | c, [] => c
  | c, b :: bs => run N (step N c b) bs
theorem run_le (N : Nat) : ∀ (c : Nat) (bs : List Bool), run N c bs ≤ c + bs.length
  | c, [] => by simp [run]
  | c, b :: bs => by
      have h1 := step_lipschitz N c b
      have h2 := run_le N (step N c b) bs
      simp [run, List.length]; omega
theorem announce_needs_K_frames (N K : Nat) (bs : List Bool) (h : K ≤ run N 0 bs) : K ≤ bs.length := by
  have := run_le N 0 bs; omega

-- §5 Resolution gate: palm-line codes are computed only if the palm box side ≥ minSide px;
--     code grid cells = side / stride, so the gate guarantees ≥ minSide/stride cells per axis.
theorem gate_cells (side minSide stride : Nat) (h : minSide ≤ side) :
    minSide / stride ≤ side / stride := Nat.div_le_div_right h

-- §6 Web-app job phases: Idle 0, Collecting 1, Collected 2, Training 3, Calibrating 4, Ready 5, Failed 6.
def canTransition : Nat → Nat → Bool
  | 0, 1 | 2, 1 | 5, 1 | 6, 1 => true          -- (re)start collection when nothing is training
  | 1, 2 => true                                -- collection finished
  | 2, 3 | 5, 3 | 6, 3 => true                  -- train from collected data / retrain / retry
  | 3, 4 => true                                -- training done → calibrate
  | 4, 5 => true                                -- calibrated → model goes live
  | 1, 6 | 3, 6 | 4, 6 => true                  -- failure or cancel while a job runs
  | _, _ => false

/-- The live detector is replaced only on Calibrating → Ready: an uncalibrated model never goes live. -/
theorem swap_requires_calibrated (a : Nat) (h : canTransition a 5 = true) : a = 4 := by
  rcases a with _ | _ | _ | _ | _ | _ | _ | a <;> simp_all [canTransition]

/-- Training starts only from collected data, a previous model, or a failed run. -/
theorem training_requires_data (a : Nat) (h : canTransition a 3 = true) : a = 2 ∨ a = 5 ∨ a = 6 := by
  rcases a with _ | _ | _ | _ | _ | _ | _ | a <;> simp_all [canTransition]

theorem no_skip_calibration : canTransition 3 5 = false := by decide
theorem collection_waits_for_training : canTransition 3 1 = false ∧ canTransition 4 1 = false := by decide
theorem no_self_loops (a : Nat) : canTransition a a = false := by
  rcases a with _ | _ | _ | _ | _ | _ | _ | a <;> simp [canTransition]

/-! ## §7 Face-mesh crop (src/facemesh.rs, integer pixels)

side = ⌊3·m/2⌋ with m = max(w, h); x0 = ⌊(x1+x2)/2⌋ − ⌊side/2⌋. The square crop always contains the
face box along each axis, with at least ⌊m/4⌋ − 1 pixels of context on both sides. -/

def cropSide (w h : Int) : Int := 3 * max w h / 2
def cropOrigin (a b side : Int) : Int := (a + b) / 2 - side / 2

theorem crop_contains_box (x1 x2 y1 y2 : Int) (hh : y1 ≤ y2) :
    let s := cropSide (x2 - x1) (y2 - y1)
    cropOrigin x1 x2 s ≤ x1 ∧ x2 ≤ cropOrigin x1 x2 s + s := by
  simp only [cropSide, cropOrigin]
  constructor <;> omega

theorem crop_margin (x1 x2 y1 y2 : Int) :
    let m := max (x2 - x1) (y2 - y1)
    let s := cropSide (x2 - x1) (y2 - y1)
    x1 - cropOrigin x1 x2 s ≥ m / 4 - 1 ∧ cropOrigin x1 x2 s + s - x2 ≥ m / 4 - 1 := by
  simp only [cropSide, cropOrigin]
  constructor <;> omega

/-! ## §8 Finger-number stabiliser (src/fingernum.rs `Stabilizer`)

The window holds the last frames' readings (0 = no number). A number is reported when at least
`need` readings in a window of at most `w` frames equal it, with `w < 2·need`. Then two different
numbers can never both qualify, so the report is unique. -/

def countEq (l : List Nat) (a : Nat) : Nat := (l.filter (fun x => decide (x = a))).length

theorem countEq_cons (x : Nat) (l : List Nat) (c : Nat) :
    countEq (x :: l) c = (if x = c then 1 else 0) + countEq l c := by
  unfold countEq
  by_cases h : x = c
  · simp [h]
    omega
  · simp [h]

theorem count_two_le (l : List Nat) (a b : Nat) (hab : a ≠ b) :
    countEq l a + countEq l b ≤ l.length := by
  induction l with
  | nil => simp [countEq]
  | cons x l ih =>
    rw [countEq_cons, countEq_cons, List.length_cons]
    split <;> split <;> omega

theorem stable_unique (l : List Nat) (w need a b : Nat) (hl : l.length ≤ w) (hm : w < 2 * need)
    (ha : need ≤ countEq l a) (hb : need ≤ countEq l b) : a = b := by
  apply Classical.byContradiction
  intro h
  have := count_two_le l a b h
  omega
