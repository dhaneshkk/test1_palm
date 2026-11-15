# palmid

Monochrome laptop-camera **face + palm detection** and **consent-based palmprint identification**, on CPU.
This is the Rust runtime of the `VG360_PalmID_Mono_Webcam` notebook; training stays in the notebook.

```
 notebook (Python, Colab)                          palmid (Rust, this repo)
 ───────────────────────                           ────────────────────────
 colour webcam frames                              grayscale frames (ffmpeg / images / API)
   → YuNet face + MediaPipe palm (teachers)          → CLAHE + letterbox (bit-exact with OpenCV)
   → grayscale copies                                → YOLOX-Nano INT8 (onnxruntime)
   → YOLOX-Nano trained with SAM                     → per-class conformal τ, NMS
   → INT8 ONNX + per-class τ  ───────────────────►   → palm ROI → CompCode → open-set ID
     palmid_int8.onnx, palmid_model.json             → OpenAI-compatible API, live MJPEG view
```

Why a student model: MediaPipe's palm detector depends on skin colour. On grayscale it found 5 palms
where colour gave 54 (same 70 frames), and CLAHE did not help. The notebook labels colour frames with
the teachers and trains YOLOX-Nano on the grayscale copies, so runtime is fully monochrome.

## What it does and does not do

* **Faces:** detection only. No face recognition.
* **Palms:** identification of **enrolled people only**, each enrolled with explicit consent
  (`--consent yes`). Anyone else is reported as `unknown`. The gallery stores CompCode orientation
  codes (one digit 0–5 per cell), never images. `palmid delete --name X` removes a person.
* **Resolution:** a webcam resolves palm principal lines and creases, **not friction ridges**
  (those need ~500 dpi contact scanners). Matching uses CompCode (6-orientation Gabor competitive code).
* **Safety gate:** if a person's genuine distances are not clearly below chance (shuffled-code
  distance − 0.06), the texture wasn't resolved and any threshold would accept everyone, so enrollment
  is **refused** with a "move closer / more light" message.
* **No liveness check:** a printed palm photo could be accepted. Don't use this for access control as is.
* **Accuracy on real palms is unmeasured.** Public palmprint datasets (Tongji, IITD, MPD) weren't
  reachable from the build environment. The tests check mechanics and OpenCV parity, not biometric accuracy.

## Build

```bash
pip install onnxruntime        # provides libonnxruntime.so (loaded at run time)
export ORT_DYLIB_PATH=$(python -c "import onnxruntime,os,glob;print(glob.glob(os.path.dirname(onnxruntime.__file__)+'/capi/libonnxruntime.so*')[0])")
cargo build --release
```

Webcam capture uses an `ffmpeg` subprocess (v4l2 on Linux, avfoundation on macOS, dshow on Windows).
No OpenCV needed.

## Use

Train in the notebook (`notebooks/VG360_PalmID_Mono_Webcam.ipynb`), then copy
`palmid_int8.onnx` and `palmid_model.json` into `models/`.

```bash
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

`--source` accepts `webcam`, `webcam:/dev/video2`, a video file/URL, or a folder of images.

### API (OpenAI-compatible)

```bash
curl -s localhost:8361/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "palmid-mono-yolox-nano-int8",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 frame.jpg)"'"}}]}]}'
```

`choices[0].message.content` is a JSON string: per object `class`, `score`, `bbox`, and for palms
`identity` (`name` / `unknown` / `pending` until 3 of the last 6 frames agree) plus per-person distances.

## Formal guarantees and tests

`lean/PalmID.lean`: 15 theorems, Lean 4.31, kernel only (Nat + `omega`, no Mathlib, no `sorry`):

| § | Theorems | Used by |
|---|---|---|
| 1 | `angDist_symm`, `angDist_self`, `angDist_le_three`, `total_le` | CompCode distance ∈ [0,1] |
| 2 | `accept_monotone`, `min_accept`, `closerRel_irrefl` | open-set accept / best-match ranking |
| 3 | `conformal_feasible`, `conformal_tight` | per-person and per-class τ need ≥ 29 calibration samples at α = 1/30 |
| 4 | `step_bounded`, `step_lipschitz`, `run_le`, `announce_needs_K_frames` | identity debounce |
| 5 | `gate_cells` | palm resolution gate |

`src/lean_mirror.rs` holds the integer mirrors that the runtime calls, with exhaustive or brute-force
checks of each theorem statement.

`cargo test` runs:

| Test | Result on the reference fixtures |
|---|---|
| CLAHE vs OpenCV | bit-exact |
| letterbox (CLAHE + INTER_LINEAR + pad) vs notebook | bit-exact |
| palm ROI (CLAHE + INTER_AREA) vs notebook | bit-exact |
| Gaussian blur vs OpenCV | within ±1 grey level (0.65 % of pixels) |
| CompCode vs notebook | 99.74 % of cells identical, distance 0.0017 |
| INT8 model output vs onnxruntime-Python | identical (max abs Δ = 0) |
| decode + NMS vs notebook | identical detections |
| enrollment / identification (synthetic palm-line images) | 20/20 genuine accepted ×2, 0/20 unenrolled accepted, noise refused |

Model tests need `ORT_DYLIB_PATH`; they are skipped without it. `tests/fixtures/test_model_int8.onnx`
is a barely trained smoke-test model for parity tests only, not a usable detector. Regenerate
fixtures with `tools/make_fixtures.py`.

Training (SAM, MCC, Mondrian conformal calibration) lives in the notebook. The Rust `eval` and
`calibrate` commands re-run the MCC and conformal steps on your own labelled data.

## Measured

On a 2-vCPU Xeon sandbox, release build, 4 threads: about 25–45 ms per 640×480 frame end to end
(detection plus palm code), and the live loop ran at about 24 fps from a video file.
