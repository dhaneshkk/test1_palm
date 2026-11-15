# palmid

Monochrome laptop-camera **face + palm detection** and **consent-based palmprint identification**,
running on CPU. Three ways to use it:

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

* **Faces:** detection only. No face recognition.
* **Palms:** identification of **enrolled people only**, each enrolled with explicit consent
  (`--consent yes` in the CLI, a consent checkbox in the web app). Anyone else is `unknown`.
  The gallery stores CompCode orientation codes (one digit 0–5 per cell), never images, and a person
  can be deleted at any time.
* **Resolution:** a webcam resolves palm principal lines and creases, **not friction ridges**
  (those need ~500 dpi contact scanners).
* **Safety gate:** if a person's genuine distances aren't clearly below chance (shuffled-code
  distance − 0.06), the palm texture wasn't resolved and any threshold would accept everyone, so
  enrollment is **refused** with a "move closer / more light" message.

See [Limitations](#limitations) before relying on any of it.

## Quick start: web app

Requirements: Rust ≥ 1.99 (Topcoat 0.12), `ffmpeg`, Python 3 with the trainer requirements.

```bash
pip install -r trainer/requirements.txt        # torch, onnx, onnxruntime, opencv, …
export ORT_DYLIB_PATH=$(python -c "import onnxruntime,os,glob;print(glob.glob(os.path.dirname(onnxruntime.__file__)+'/capi/libonnxruntime.so*')[0])")
cargo install topcoat-cli --version 0.12.1 --locked
topcoat asset bundle -p palmid-web --release   # Topcoat's browser runtime
cargo run -p palmid-web --release              # open http://127.0.0.1:3000
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
| `PALMID_SOURCE` | `webcam` | `webcam`, `webcam:/dev/video2`, a video file/URL (looped), or an image folder |
| `PALMID_WORK` | `palmid_work` | dataset, models, gallery |
| `PALMID_WIDTH` / `PALMID_HEIGHT` | `640` / `480` | capture size |
| `PALMID_PYTHON` / `PALMID_TRAINER` | `python3` / `trainer/train_student.py` | training job |
| `PALMID_YOLOX`, `PALMID_YOLOX_WEIGHTS` | `~/.cache/palmid/…` | YOLOX checkout and COCO weights, fetched on first training run |
| `PALMID_THREADS` | `4` | onnxruntime and torch threads |
| `HOST` / `PORT` | `127.0.0.1` / `3000` | Topcoat |

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

Requirements: stable Rust, `ffmpeg`, `onnxruntime` (set `ORT_DYLIB_PATH` as above). No OpenCV.

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
```

The CLI reads `models/palmid_int8.onnx`, `models/palmid_model.json` and `palm_gallery.json` by
default. To use what the web app trained and enrolled:

```bash
$B --model palmid_work/models/palmid_int8.onnx --meta palmid_work/models/palmid_model.json \
   --gallery palmid_work/palm_gallery.json live
```

`--source` accepts `webcam`, `webcam:/dev/video2`, a video file/URL, or a folder of images.
Capture uses an `ffmpeg` subprocess (v4l2 on Linux, avfoundation on macOS, dshow on Windows).

### API (OpenAI-compatible)

Served by both `palmid serve` / `palmid live` and the web app.

```bash
curl -s localhost:8361/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "palmid-mono-yolox-nano-int8",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 frame.jpg)"'"}}]}]}'
```

`choices[0].message.content` is a JSON string: per object `class`, `score`, `bbox`, and for palms
`identity` (`name` / `unknown` / `pending` until 3 of the last 6 frames agree) plus per-person
distances. A request with no image returns the current status (phase, model, enrolled names, metrics).

## Training without the web app

```bash
python trainer/train_student.py --data DATASET --out OUT --epochs 30      # SAM; prints JSON progress
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
conformal recall target, so its threshold fell to the floor (shown as `sccp_floor(coverage not
attainable)`) and palm MCC was 0. Use more frames and epochs, a smaller batch, or a GPU.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | `palmid` library and CLI: preprocessing, detector, teachers, palmprint, gallery, metrics, dataset writer, camera, API server, Lean mirrors |
| `web/` | `palmid-web` Topcoat app (state, camera loop, training job, UI, API) |
| `trainer/` | `train_student.py` (SAM YOLOX-Nano trainer) and `requirements.txt` |
| `notebooks/` | original single-cell Colab notebook |
| `lean/` | `PalmID.lean`, the formal guarantees |
| `tests/` | OpenCV / onnxruntime parity, gallery end to end, teacher parity |
| `tools/` | fixture generators (`make_fixtures.py`, `teacher_ref.py`) |
| `models/` | default location for CLI models (not committed) |
| `docs/` | screenshot |

## Formal guarantees and tests

`lean/PalmID.lean`: 20 theorems, Lean 4.31, kernel only (Nat + `omega`, no Mathlib, no `sorry`):

| § | Theorems | Used by |
|---|---|---|
| 1 | `angDist_symm`, `angDist_self`, `angDist_le_three`, `total_le` | CompCode distance ∈ [0,1] |
| 2 | `accept_monotone`, `min_accept`, `closerRel_irrefl` | open-set accept / best-match ranking |
| 3 | `conformal_feasible`, `conformal_tight` | per-person and per-class τ need ≥ 29 calibration samples at α = 1/30 |
| 4 | `step_bounded`, `step_lipschitz`, `run_le`, `announce_needs_K_frames` | identity debounce |
| 5 | `gate_cells` | palm resolution gate |
| 6 | `swap_requires_calibrated`, `training_requires_data`, `no_skip_calibration`, `collection_waits_for_training`, `no_self_loops` | web-app job phases |

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
| enrollment / identification (synthetic palm-line images) | 20/20 genuine accepted ×2, 0/20 unenrolled accepted, noise refused |

* Model tests need `ORT_DYLIB_PATH` and are skipped without it.
* Teacher tests need `PALMID_TEACHERS` (folder with `yunet.onnx`, `palm.onnx`) and
  `PALMID_TEACHER_CASES` / `PALMID_GRAY_CASES` (image | reference pairs from `tools/teacher_ref.py`).
  They are skipped otherwise, so CI doesn't run them.
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
* **MCC is measured against the colour teachers' labels**, not human ground truth.
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
| web app, collection | about 3–4 frames/s with both colour teachers (100 frames in 33 s) |
| web app, training | 3 epochs on 60 frames: 66 s; 8 epochs on 180 frames, 2 threads: about 30 min |
| API on the live model | 90–140 ms per request including JPEG/PNG decode |
