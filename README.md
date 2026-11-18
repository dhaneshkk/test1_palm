# palmid

Monochrome laptop-camera **face + palm detection**, **consent-based palmprint identification**,
**hand landmarks with Chinese finger numbers 1–10** and a **468-point face mesh**, all on CPU.

**New here? Read the [user guide](USER_GUIDE.md)**: install, collect, train, enroll, finger numbers,
API and troubleshooting, step by step. This README covers how it works and how it is tested.

Three ways to use it:

| | What | Where |
|---|---|---|
| **Web app** | Collect → train → calibrate → live inference, with live status in the browser | [`web/`](web) (Rust, [Topcoat](https://github.com/tokio-rs/topcoat) 0.12) |
| **CLI + API** | Enroll, identify, live MJPEG view, OpenAI-compatible server, calibrate, evaluate | [`src/`](src) (Rust library + `palmid` binary) |
| **Notebook** | The original single-cell Colab pipeline | [`notebooks/`](notebooks) |

![palmid web dashboard: live monochrome feed, pipeline phase, training loss chart, per-class τ and MCC, trainer log](docs/web-dashboard.png)

*(The boxes and FPS text inside this frame are baked into the opencv_zoo demo clip used as a stand-in camera.)*

## How it works

```
 colour frames ─► colour teachers ─► grayscale copies + labels ─► YOLOX-Nano student (SAM) ─► INT8 ONNX
                  YuNet face,                                     trained on grayscale only
                  MediaPipe palm
                                                                                     │
 grayscale frame ─► CLAHE + letterbox ─► INT8 student ─► per-class conformal τ, NMS ◄┘
                                                │
                                                └─ palm box ─► ROI ─► CompCode ─► open-set ID (enrolled only)
```

**Why a student model.** MediaPipe's palm detector depends on skin colour. On grayscale it found 5
palms where colour gave 54 (same 70 frames), and CLAHE did not help. So colour frames are labelled by
the teachers and YOLOX-Nano is trained on the grayscale copies. Everything at run time is monochrome.

**Thresholds.** Each class gets a threshold from Mondrian split-conformal calibration at α = 1/30
(needs ≥ 29 calibration objects, proven in Lean). Evaluation is frame-level MCC per class.

## What it does and does not do

| Feature | What it does | Limits |
|---|---|---|
| **Faces** | detection, plus a 468-point mesh joined by Delaunay triangles | no face recognition; mesh assumes an upright face |
| **Palms** | detection and identification of **enrolled people only**, each enrolled with explicit consent (`--consent yes` in the CLI, a checkbox in the web app); anyone else is `unknown` | palm lines and creases only, **not friction ridges** (those need ~500 dpi contact scanners); no liveness check |
| **Gallery** | CompCode orientation codes (one digit 0–5 per cell), never images; delete a person at any time | — |
| **Hand landmarks** | 21 points per palm, hand skeleton on the live view, and a palm crop aligned to the hand for enrollment and matching | hand assumed roughly upright |
| **Finger numbers** | Chinese one-hand numbers 1–10 from the landmarks, held for 6 of 8 frames before shown; follows a hand by landmarks when the detector drops it (fists, pinches) | 6–9 not yet measured on labelled real hands (the web app can record and score them) |
| **Second factor** | verifies a **claimed** person: open palm matched 1:1, then a random Chinese finger-number challenge ("show 8, then 10") on the same tracked hand within 40 s; 5 failures lock the palm factor for 15 min; TOTP code (RFC 6238, any authenticator app) as the non-biometric fallback with its own 5-try lock; audit log | stops photos, **not** replayed or injected video: not certified liveness; use behind a first factor |
| **Safety gate** | refuses enrollment when a person's palm samples aren't clearly more alike than random palms (shuffled-code distance − 0.06), because any threshold would then accept everyone | — |

Details:

* **Hand landmarks:** MediaPipe's 21-point hand-landmark model (opencv_zoo, Apache-2.0) runs on the
  grayscale crop of each palm. The aligned crop is rotated along the wrist→middle-knuckle axis and
  scaled to the knuckle width. New enrollments use it when landmarks are available; people enrolled
  earlier keep the box crop, and each person's crop mode is stored so probes are cut the same way. The
  model downloads on first start of the web app; for the CLI, put `handpose.onnx` next to `--model`.
  On 54 grayscale hands: mean confidence 0.93 (colour 0.97), landmarks within about 5 % of palm size of
  the colour result.
* **Finger numbers** (`src/fingernum.rs`, geometric rules, no training): finger extension from 3-D
  joint angles in a hand-centred frame, so position, scale, in-plane rotation and left/right hand don't
  matter. 7 = thumb, index and middle tips pinched; 9 = hooked index; 10 = fist or crossed index and
  middle; 3 accepts index-middle-ring and middle-ring-little. Two numbers can never both pass the
  6-of-8 stabiliser (Lean §8). Checked: 960 synthetic poses all correct; on the opencv_zoo demo clip
  45 of 47 open hands read 5 (2 read 4, thumb close to the palm), and closing hands followed by
  landmarks read 10 once closed (half-closed frames give no number). **Record and score your own
  hands** in the web app (section 4) or with `palmid fingers-eval <recordings.jsonl>`: per-number
  recall, precision and MCC, and the multiclass MCC over 1–10 plus "no number".
* **Face mesh:** MediaPipe's face-landmark model (468 points, Apache-2.0), converted from the
  `mediapipe` 0.10.14 wheel to ONNX and **embedded in the binaries** (2.4 MB; `PALMID_FACEMESH=<file>`
  overrides it). On 16 face crops: presence 1.00 on grayscale, points within 1.4 % of the inter-ocular
  distance of the colour result. See `assets/models/NOTICE-face_landmark.md`.

The same palm-ridge code, face detector and face mesh are reused by
[palm-recognition-prototype](https://github.com/dhaneshkk/palm-recognition-prototype) (branch
`rust-inference`) as a Cargo dependency.

See [Limitations](#limitations) before relying on any of it.

## Quick start: web app

Requirements: Rust ≥ 1.99 (Topcoat 0.12), `ffmpeg`, Python 3 with the trainer requirements in a
virtual environment. Ubuntu 24.04 and Debian 12 refuse system-wide `pip install` (PEP 668).

```bash
sudo apt install -y python3-venv ffmpeg git curl
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU wheels (~200 MB, not ~2.5 GB CUDA)
pip install -r trainer/requirements.txt
export ORT_DYLIB_PATH=$(python3 -c "import onnxruntime,os,glob;print(glob.glob(os.path.dirname(onnxruntime.__file__)+'/capi/libonnxruntime.so*')[0])")
export PALMID_PYTHON=$PWD/.venv/bin/python3    # the web app runs the trainer with this interpreter

cargo install topcoat-cli --version 0.12.1 --locked
topcoat asset bundle -p palmid-web --release   # Topcoat's browser runtime
cargo run -p palmid-web --release              # open http://127.0.0.1:3000
```

Re-run `source .venv/bin/activate` and the two `export` lines in each new terminal.

To re-derive thresholds for an existing model without retraining (stop the web app first, then
restart it):

```bash
target/release/palmid --model palmid_work/models/palmid_int8.onnx --meta palmid_work/models/palmid_model.json \
  calibrate --data palmid_work/dataset --split cal
```

Then in the page:

1. **Collect** about 300 frames. Show your face and open palms at different distances and angles.
   Teacher models download from opencv_zoo the first time.
2. **Train** for 30+ epochs. You see epoch, iteration, loss, learning rate, a loss chart and the trainer
   log live. When training ends, Rust calibrates τ and computes MCC on the val split, then the new
   model goes live. **Cancel** stops the trainer.
3. **Enroll** a consenting person: tick the box and hold one open palm 25–40 cm from the camera until
   40 views are captured. It gives up with a message after about 450 frames without enough palm views.

**The camera is read by the server process**, through ffmpeg on the machine running `palmid-web`,
not by the browser. Run the server on the laptop whose camera you want.

| env | default | |
|---|---|---|
| `PALMID_SOURCE` | `webcam` | `webcam`, `webcam:/dev/video2`, `dshow:<camera name>` (Windows ffmpeg, e.g. from WSL2), a video file/URL (looped), or an image folder |
| `PALMID_WORK` | `palmid_work` | dataset, models, gallery |
| `PALMID_WIDTH` / `PALMID_HEIGHT` | `640` / `480` | capture size; with `dshow:` the camera is asked for this size. Use `1280`/`720` on an HD/FHD webcam for more palm-line detail |
| `PALMID_PYTHON` / `PALMID_TRAINER` | `python3` / `trainer/train_student.py` | training job |
| `PALMID_YOLOX`, `PALMID_YOLOX_WEIGHTS` | `~/.cache/palmid/…` | YOLOX checkout and COCO weights, fetched on first training run |
| `PALMID_THREADS` | `4` | onnxruntime and torch threads |
| `HOST` / `PORT` | `127.0.0.1` / `3000` | Topcoat |

### Windows + WSL2

* **Clone inside the Linux filesystem** (`~/test1_palm`), not under `/mnt/c/…`. Cargo builds and
  Python imports are many times slower across the Windows drive mount.
* **Camera:** WSL2 usually has no `/dev/video0` (check with `ls /dev/video*`). Let palmid run the
  **Windows** ffmpeg itself through WSL interop. No pipe file, no network, no firewall:

  ```bash
  # once, in Windows PowerShell: winget install Gyan.FFmpeg   (then open a new WSL terminal)
  ffmpeg.exe -hide_banner -list_devices true -f dshow -i dummy      # find the "(video)" camera name
  export PALMID_SOURCE="dshow:HP True Vision FHD Camera"             # your camera's name
  cargo run -p palmid-web --release
  ```

  If `ffmpeg.exe` isn't on the WSL `PATH`, set `PALMID_FFMPEG_WIN` to its full path, for example
  `/mnt/c/Users/<you>/AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe`. If the feed shows a crossed-out
  camera icon, the laptop's camera privacy key or shutter is on. If the camera reports "busy" after a
  crash, end leftover processes with `taskkill.exe /IM ffmpeg.exe /F`.

  A TCP stream also works (`PALMID_SOURCE=tcp://host:5056` with Windows ffmpeg serving
  `-f mpegts "tcp://0.0.0.0:5056?listen=1"`), but WSL's NAT network usually needs a firewall rule.
* Natively on Windows, `PALMID_SOURCE=dshow:<camera name>` works the same way.

### What the web app is made of

```
 browser ◄──── live regions (WebSocket) ──── palmid-web (Topcoat)
   │  buttons = Topcoat procedures             │
   ▼                                            ├─ camera thread: ffmpeg → RGB frame → active mode
 1 Collect ────────────────────────────────────►│    collect → teachers (Rust/ort) → gray PNG + YOLO labels
 2 Train ──────────────────────────────────────►│    Python child: trainer/train_student.py (SAM) → JSON progress lines
                                                │    → Rust: split-conformal τ + MCC on val → swap model
 3 Enroll ─────────────────────────────────────►│    enroll  → 40 palm codes → conformal τ + chance gate
   (live view) ◄────────────────────────────────│    infer   → INT8 student + palmprint ID
                                                └─ POST /v1/chat/completions (same live model)
```

* Training stays in Python because YOLOX training needs PyTorch. The app supervises the trainer and
  parses one JSON event per line (`start`, `iter`, `epoch`, `export`, `done`, `error`).
* The pipeline phase is a checked state machine (`lean_mirror::can_transition`, Lean §6). The live
  detector is replaced only on **Calibrating → Ready** (`swap_requires_calibrated`), so a model with
  uncalibrated thresholds never serves.
* Collection splits frames in interleaved 10-frame blocks (6 train : 2 cal : 2 val), so every split
  covers the whole session. A plain time split once left the val split with no palms.

## CLI and API

Requirements: stable Rust, `ffmpeg`, `onnxruntime` (in the venv above; set `ORT_DYLIB_PATH`). No OpenCV.

```bash
cargo build --release -p palmid
B=target/release/palmid
$B enroll --name dhanesh --consent yes --source webcam          # hold one open palm 25–40 cm away
$B list
$B live --source webcam --port 8361                             # open http://localhost:8361/
$B identify --source webcam --max-frames 100
$B serve --port 8361                                            # API only
$B delete --name dhanesh
$B detect some.jpg other.png
$B calibrate --data path/to/yolo_data --split cal               # re-derive per-class τ, rewrites --meta
$B eval --data path/to/yolo_data --split val                    # frame-level MCC per class
$B fingers-eval palmid_work/finger_recordings.jsonl             # score labelled finger-number recordings
$B --auth palmid_work/auth_state.json auth-status                # second factor: counts, locks, fallback
$B --gallery palmid_work/palm_gallery.json --auth palmid_work/auth_state.json auth-setup-code --name asha
$B --auth palmid_work/auth_state.json auth-unlock --name asha    # administrator unlock
```

The CLI reads `models/palmid_int8.onnx`, `models/palmid_model.json` and `palm_gallery.json` by
default. To use what the web app trained and enrolled:

```bash
$B --model palmid_work/models/palmid_int8.onnx --meta palmid_work/models/palmid_model.json \
   --gallery palmid_work/palm_gallery.json live
```

`--source` accepts `webcam`, `webcam:/dev/video2`, a video file/URL, or a folder of images.
Capture uses an `ffmpeg` subprocess (v4l2 on Linux, avfoundation on macOS, dshow on Windows); any
ffmpeg input URL also works, e.g. `tcp://host:5056` for a camera streamed from another machine.

### API (OpenAI-compatible)

Served by both `palmid serve` / `palmid live` and the web app.

```bash
curl -s localhost:8361/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "palmid-mono-yolox-nano-int8",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 frame.jpg)"'"}}]}]}'
```

`choices[0].message.content` is a JSON string: per object `class`, `score`, `bbox`; for palms
`identity` (`name` / `unknown` / `pending` until 3 of the last 6 frames agree) with per-person
distances, `hand` (21 landmarks) and `finger_number` (value, rule, extended fingers); for faces
`face_mesh` (468 points and presence).

Second factor (web app): `POST /v1/auth/start {"user": …}`, `POST /v1/auth/status`,
`POST /v1/auth/fallback {"user": …, "code": "123456"}`, `POST /v1/auth/cancel`. See the
[user guide §10](USER_GUIDE.md#10-second-factor-verification). A request with no image returns the current status (phase, model, enrolled names, metrics).

## Training without the web app

```bash
python3 trainer/train_student.py --data DATASET --out OUT --epochs 30      # SAM; prints JSON progress
$B --model OUT/palmid_int8.onnx --meta OUT/palmid_model.json calibrate --data DATASET --split cal
$B --model OUT/palmid_int8.onnx --meta OUT/palmid_model.json eval --data DATASET --split val
```

`DATASET` is YOLO format: `images/{train,cal,val}/*.png` (grayscale) and `labels/{split}/*.txt`
(`cls cx cy w h`, normalised). The web app's **Collect** writes exactly this under `palmid_work/dataset`.

The Colab notebook ([`notebooks/VG360_PalmID_Mono_Webcam.ipynb`](notebooks)) does capture, teacher
labelling, SAM training and export in one cell, and writes `palmid_int8.onnx` + `palmid_model.json` for
the Rust side. Its own enrollment gallery (`.npz`) is not compatible with the Rust gallery; enroll
again with the CLI or web app.

**Training budget matters.** In testing on a 2-core CPU, 8 epochs on 180 training frames (88 SAM
iterations at batch 16, about 30 minutes) was not enough. The palm detector could not reach the
conformal recall target. When that happens the threshold now falls back to 0.30 (shown as
`fallback(coverage not attainable …)`) instead of the score floor, which flooded the view with false
boxes. Use more frames and epochs, a smaller batch, or a GPU.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | `palmid` library and CLI: preprocessing, detector, teachers, palmprint, gallery, hand landmarks, finger numbers, second-factor auth (TOTP, lockout, challenge), face mesh + Delaunay, metrics, dataset writer, camera, API server, Lean mirrors |
| `web/` | `palmid-web` Topcoat app (state, camera loop, training job, UI, API) |
| `trainer/` | `train_student.py` (SAM YOLOX-Nano trainer) and `requirements.txt` |
| `notebooks/` | original single-cell Colab notebook |
| `lean/` | `PalmID.lean`, the formal guarantees |
| `tests/` | OpenCV / onnxruntime parity, gallery end to end, teacher parity |
| `tools/` | fixture generators (`make_fixtures.py`, `teacher_ref.py`, `handpose_ref.py`, `facemesh_ref.py`), `convert_facemesh.sh` |
| `assets/models/` | embedded face-landmark ONNX and its licence notice |
| `models/` | default location for CLI models (not committed) |
| `docs/` | screenshot |
| `USER_GUIDE.md` | step-by-step guide for running the app |

## Formal guarantees and tests

`lean/PalmID.lean`: 29 theorems, Lean 4.31, kernel only (Nat/Int + `omega`, no Mathlib, no `sorry`):

| § | Theorems | Used by |
|---|---|---|
| 1 | `angDist_symm`, `angDist_self`, `angDist_le_three`, `total_le` | CompCode distance ∈ [0,1] |
| 2 | `accept_monotone`, `min_accept`, `closerRel_irrefl` | open-set accept / best-match ranking |
| 3 | `conformal_feasible`, `conformal_tight` | per-person and per-class τ need ≥ 29 calibration samples at α = 1/30 |
| 4 | `step_bounded`, `step_lipschitz`, `run_le`, `announce_needs_K_frames` | identity debounce |
| 5 | `gate_cells` | palm resolution gate |
| 6 | `swap_requires_calibrated`, `training_requires_data`, `no_skip_calibration`, `collection_waits_for_training`, `no_self_loops` | web-app job phases |
| 7 | `crop_contains_box`, `crop_margin` | face-mesh crop always contains the face box, with ≥ ⌊m/4⌋ − 1 px context per side |
| 8 | `countEq_cons`, `count_two_le`, `stable_unique` | finger-number stabiliser: with window < 2·need, at most one number can be reported |
| 9 | `failN_counts`, `fail_reaching_max_locks`, `max_failures_lock`, `succeed_keeps_lock` | second-factor lockout: failures are counted one by one, the 5th locks the factor for L, nothing is attempted before it expires, a success doesn't lift a lock |

`src/lean_mirror.rs` holds the integer mirrors the runtime actually calls, with exhaustive or
brute-force checks of each theorem statement.

`cargo test -p palmid` runs:

| Test | Result |
|---|---|
| CLAHE vs OpenCV | bit-exact |
| letterbox (CLAHE + INTER_LINEAR + pad) vs notebook | bit-exact |
| palm ROI (CLAHE + INTER_AREA) vs notebook | bit-exact |
| Gaussian blur vs OpenCV | within ±1 grey level (0.65 % of pixels) |
| CompCode vs notebook | 99.74 % of cells identical, distance 0.0017 |
| INT8 model output vs onnxruntime-Python, same machine | identical (max abs Δ = 0) |
| decode + NMS vs notebook | identical detections |
| colour teachers vs OpenCV FaceDetectorYN / notebook palm decode | IoU 1.0000 |
| RGB→gray vs `cvtColor` | identical over all 2²⁴ colours |
| 21 hand landmarks vs the opencv_zoo procedure (upright variant) | identical (worst 0.000 px; needs `PALMID_HANDPOSE`) |
| 468-point face mesh vs onnxruntime-Python (`tools/facemesh_ref.py`) | identical (worst 0.000 px; needs `PALMID_FACEMESH_CASE`) |
| Delaunay (Bowyer–Watson) vs `scipy.spatial.Delaunay`, 12 / 68 / 468 random points | identical triangle sets |
| finger numbers on 960 synthetic hand poses (rotation ±40°, scale, mirror, landmark noise) | all correct |
| fast CompCode (responses only at the 28×28 sample points) vs full filter2d | bit-identical |
| enrollment / identification (synthetic palm-line images) | 20/20 genuine accepted ×2, 0/20 unenrolled accepted, noise refused |

* Model tests need `ORT_DYLIB_PATH` and are skipped without it.
* Teacher tests need `PALMID_TEACHERS` (folder with `yunet.onnx`, `palm.onnx`) and
  `PALMID_TEACHER_CASES` / `PALMID_GRAY_CASES` (image | reference pairs from `tools/teacher_ref.py`).
  They are skipped otherwise, so CI doesn't run them.
* The face-mesh parity test needs `PALMID_FACEMESH_CASE=<gray.png>|<ref.json>|<x1,y1,x2,y2>` (refs
  from `tools/facemesh_ref.py`); `PALMID_FACEMESH_OUT=<png>` also writes the drawn mesh. On a face
  with near-cocircular points the Rust and scipy triangulations can differ by a triangle (901 vs 902
  on the test face); on generic points they are identical.
* `tests/fixtures/test_model_int8.onnx` is a barely trained smoke-test model for parity tests only,
  not a usable detector. Regenerate fixtures with `tools/make_fixtures.py`.

**CI** (GitHub Actions) builds and tests the whole workspace on current stable Rust, builds the
Topcoat asset bundle, and checks every Lean theorem for `sorry`. It regenerates the onnxruntime
reference fixtures on the runner first, because INT8 kernels differ across CPUs (next section).

## Limitations

* **Accuracy on real palms is unmeasured.** Public palmprint datasets (Tongji, IITD, MPD) weren't
  reachable from the build environment. The tests check mechanics and OpenCV parity, not biometric
  accuracy.
* **No liveness check:** a printed palm photo could be accepted. Don't use this for access control as is.
* **MCC is measured against the colour teachers' labels**, not human ground truth. It shows `n/a` when
  the val split has no frames without the class (or none with it); MCC is undefined then. Spend part
  of **Collect** with no face or palm in view.
* **The second factor is a webcam layer, not certified liveness.** The finger-number challenge defeats
  photos and printouts; a replayed or injected video can defeat it. NIST SP 800-63B-4 expects
  presentation-attack detection and a false match rate of 1 in 10 000 or better for biometric
  factors; neither is demonstrated here. For that, add a certified palm-vein sensor.
* **Finger numbers 6–9 are unverified on real hands**, and every hand-based feature assumes a roughly
  upright hand (the landmark model's crop has no rotation).
* **The web UI has no login.** It binds to `127.0.0.1` by default. Keep it that way or put it behind
  authentication before exposing it, because it can enroll and delete people.
* **INT8 results depend on the CPU.** The same INT8 model and input gave raw outputs up to 3.1 apart on
  the build sandbox's Xeon and on a GitHub runner (different onnxruntime kernels and versions). Final
  detections matched, but scores near a threshold can flip. **Run `palmid calibrate` on the
  deployment machine** (for example the N100 edge box) so the thresholds fit the kernels that will
  run there.
* **Test runs used a looping 70-frame demo clip** as the camera, so their val frames repeat training
  frames. Numbers from those runs say the pipeline works, not how well the model generalises.

## Measured

On a 2-vCPU Xeon sandbox (no GPU):

| | |
|---|---|
| CLI, release build, 4 threads | about 25–45 ms per 640×480 frame end to end (detection + palm code); live loop about 24 fps from a video file |
| with hand landmarks | about 88 ms per frame with ~1.2 palms per frame (detection + 21 landmarks + palm code) |
| face mesh | Delaunay triangulation + drawing of 468 points: 0.7 ms per face (release) |
| web app, collection | about 3–4 frames/s with both colour teachers (100 frames in 33 s) |
| web app, training | 3 epochs on 60 frames: 66 s; 8 epochs on 180 frames, 2 threads: about 30 min |
| API on the live model | 90–140 ms per request including JPEG/PNG decode |
