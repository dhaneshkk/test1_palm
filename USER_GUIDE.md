# palmid user guide

This guide is for the person running palmid on a laptop: setting it up, training it on your own
camera, enrolling people, reading the screen and fixing common problems. For how it works inside, see
the [README](README.md).

* [1. What palmid does](#1-what-palmid-does)
* [2. Before you start](#2-before-you-start)
* [3. Install](#3-install)
* [4. Start the web app](#4-start-the-web-app)
* [5. The screen](#5-the-screen)
* [6. Step 1: Collect](#6-step-1-collect)
* [7. Step 2: Train](#7-step-2-train)
* [8. Step 3: Enroll a person](#8-step-3-enroll-a-person)
* [9. Live identification](#9-live-identification)
* [10. Second-factor verification](#10-second-factor-verification)
* [11. Chinese finger numbers](#11-chinese-finger-numbers)
* [12. Face mesh](#12-face-mesh)
* [13. Using the API](#13-using-the-api)
* [14. Command line instead of the web app](#14-command-line-instead-of-the-web-app)
* [15. Troubleshooting](#15-troubleshooting)
* [16. Privacy and safe use](#16-privacy-and-safe-use)

---

## 1. What palmid does

palmid watches a **black-and-white** (grayscale) camera feed and:

| It does | It does not |
|---|---|
| Find **faces** and draw a 468-point mesh over each one | Recognise faces. Faces are never identified |
| Find **palms**, draw the 21-point hand skeleton | Read fingerprints or friction ridges (a webcam is far too coarse) |
| **Identify enrolled people by their palm lines**, if they agreed to enroll | Identify anyone who isn't enrolled: they show as `unknown` |
| Read **Chinese one-hand numbers 1–10** from your hand | Check that a palm is real (a printed photo could pass) |
| Verify a **claimed person** as a second login factor: palm 1:1 + random finger numbers, lockout, fallback code | Work as access control on its own (no certified liveness) |
| Answer the same questions over an OpenAI-style HTTP API | |

Everything runs on the CPU. No images of palms are stored: the gallery keeps only number codes of the
palm lines, and a person can be deleted at any time.

## 2. Before you start

You need:

* A laptop with a webcam (an HD/FHD webcam gives better palm detail).
* Linux, macOS, or Windows with **WSL2** (Ubuntu). About 4 GB free disk.
* Good, even light. Palm lines are faint; most problems come from dim or back-lit hands.
* An hour for the first run: install, about 10 minutes of collecting, then training (CPU training is slow; see §7).

## 3. Install

Open a terminal in the folder where you want palmid. On WSL2, use your Linux home folder (`~`),
**not** `/mnt/c/...`; the Windows drive is many times slower.

```bash
sudo apt install -y python3-venv ffmpeg git curl build-essential
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh     # Rust; then open a new terminal
rustup update                                                      # needs Rust 1.99 or newer

git clone https://github.com/dhaneshkk/test1_palm && cd test1_palm
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r trainer/requirements.txt

cargo install topcoat-cli --version 0.12.1 --locked
topcoat asset bundle -p palmid-web --release
```

**Every time you open a new terminal**, run these three lines from the `test1_palm` folder:

```bash
source .venv/bin/activate
export ORT_DYLIB_PATH=$(python3 -c "import onnxruntime,os,glob;print(glob.glob(os.path.dirname(onnxruntime.__file__)+'/capi/libonnxruntime.so*')[0])")
export PALMID_PYTHON=$PWD/.venv/bin/python3
```

### Camera on WSL2 (Windows)

WSL2 usually can't see the webcam directly. palmid can use the Windows copy of ffmpeg instead:

1. In Windows PowerShell, once: `winget install Gyan.FFmpeg`, then open a new WSL terminal.
2. Find your camera's name:
   ```bash
   ffmpeg.exe -hide_banner -list_devices true -f dshow -i dummy
   ```
   Look for the line ending in `(video)`, for example `"HP True Vision FHD Camera"`.
3. Tell palmid to use it, at a resolution that shows palm lines:
   ```bash
   export PALMID_SOURCE="dshow:HP True Vision FHD Camera"
   export PALMID_WIDTH=1280 PALMID_HEIGHT=720
   ```

## 4. Start the web app

```bash
cargo run -p palmid-web --release
```

The first build takes several minutes. When you see the listening address, open
**http://127.0.0.1:3000** in your browser. The camera is opened by this program, not by the browser,
so the browser doesn't ask for camera permission.

Useful settings (set them before `cargo run`):

| Setting | Default | Use it to |
|---|---|---|
| `PALMID_SOURCE` | `webcam` | pick a camera: `webcam:/dev/video2`, `dshow:<name>`, a video file, or a folder of images |
| `PALMID_WIDTH`, `PALMID_HEIGHT` | 640, 480 | capture larger frames (1280, 720 recommended on HD webcams) |
| `PALMID_WORK` | `palmid_work` | keep data, models and the gallery somewhere else |
| `PALMID_THREADS` | 4 | match your CPU core count |
| `PORT` | 3000 | change the web port |

## 5. The screen

| Area | What it shows |
|---|---|
| **Video** (left) | the grayscale feed. Faces: thin white box and the triangle mesh. Palms: thick grey box, hand skeleton, and the person's name if identified |
| **Controls** (right, top) | the four steps: Collect, Train, Enroll, Finger numbers. Messages appear in yellow under them |
| **Camera** | source, frames per second, time per frame, which models are loaded, the current **finger number**, and a line per detected object |
| **Pipeline** | the current phase (idle, collecting, training, calibrating, model live), progress bars, the training loss chart and, after training, the quality table |
| **Gallery** | enrolled people and their personal thresholds |

"hand landmarks" and "face mesh" in the Camera panel should say **loaded** a few seconds after start.
The hand model downloads on first start, so it needs internet once.

## 6. Step 1: Collect

palmid learns to find faces and palms in **your** camera's grayscale picture. Collecting records
frames and labels them automatically with colour models.

1. Type a number of frames (300 is a good start) and press **Collect**.
2. While it runs:
   * show your face at different distances and angles;
   * show **open palms**, both hands, near and far, tilted a little;
   * spend about a fifth of the time with **no face and no palm in view** (step aside, hands down).
     These empty frames are what let the quality numbers (MCC) mean something.
3. The Pipeline panel counts frames, faces and palms. Collection runs at about 3–4 frames a second.

You can collect more than once; new frames are added.

## 7. Step 2: Train

1. Type a number of epochs (30 or more) and press **Train**.
2. Watch the Pipeline panel: epoch and iteration, loss (should fall), learning rate and the trainer's log.
3. When training ends, palmid sets a threshold for faces and palms on part of the data it kept aside,
   measures quality on another part, and switches to the new model. The phase shows **model live**.
4. **Cancel** stops training; the previous model keeps running.

**How long?** On a 2-core laptop CPU, 8 epochs on 180 frames took about 30 minutes, and that was too
little: palms were missed. Plan for more frames and epochs, or train on a machine with a GPU (§14).

**Reading the quality table:**

| Column | Meaning |
|---|---|
| MCC (val) | agreement between palmid and the colour labels on frames it didn't train on: 1 is perfect, 0 is guessing. `n/a` means the check frames had no example (or no frame without one) of that class, so collect again with empty frames |
| τ | the detection threshold chosen for that class |
| τ source | how it was chosen. `fallback(coverage not attainable …)` means the model is too weak to meet the target, and a safe fixed 0.30 was used: train more |
| TP/FP/FN/TN | correct finds / false finds / misses / correct empties |

## 8. Step 3: Enroll a person

**Only enroll people who have agreed.**

1. Type the person's name and tick **this person consents**.
2. Press **Enroll**. The person holds **one open palm** facing the camera, 25–40 cm away, fingers
   relaxed and apart, in good light. Move it slightly so palmid sees a few angles.
3. palmid collects 40 palm views, then works out a personal threshold. The Gallery panel shows the new
   person.

If it can't get 40 good views in about 450 frames, it stops with a message. Move closer or add light
and try again.

**"Enrollment refused … move closer / more light"** means the palm lines weren't sharp enough: this
person's palm samples didn't look more alike than random palms. Accepting them would let anyone in, so
palmid refuses. Use 1280×720 capture, more light, and hold the palm closer.

**Delete a person:** type the name next to **forget** and press **Delete**. Their codes are removed.

## 9. Live identification

Once someone is enrolled, a palm in view shows one of:

| Label | Meaning |
|---|---|
| a **name** | the palm matched that person in at least 3 of the last 6 frames |
| `pending` | a match is forming but hasn't held for 3 of 6 frames yet |
| `unknown` | no enrolled person matched. Everyone not enrolled sees this |

Hold the palm the same way as during enrollment: same hand, open, similar distance.

## 10. Second-factor verification

palmid can check that the person at the camera is a **claimed, enrolled person**, as the second step
of a login after your system has checked the first factor (for example a password).

**What it checks:**

1. **Palm, 1:1.** The person shows an open palm; it must match *that person's* enrolled palm codes in
   3 frames (it is compared with nobody else).
2. **Random finger numbers.** Then, on the same hand, they show the numbers palmid asks for, in order,
   for example "show 8, then show 10". The numbers change every time.
3. **One hand, in view, in time.** Exactly one hand may be in view; if it leaves the view for more
   than 1.5 seconds, or 40 seconds pass, the attempt fails.

A printed palm or a photo can't do step 2, so it fails. A replayed or injected **video** of the right
gestures could pass: this is a webcam-level check, not certified liveness detection (see §16).

**Do it in the web app** (section **5 · Verify**):

1. Type the person's name and press **Verify**. The steps appear in the Camera panel with a countdown;
   each step is crossed out when done.
2. The result shows **✔ verified** or **✖ not verified** with the reason and the tries left.
3. **Cancel** stops an attempt; it counts as a failed one.

**Lockout.** 5 failed attempts in a row lock the person's palm check for **15 minutes**. A successful
check (palm or code) resets the count. The Gallery panel shows each person's failures and locks.

**Fallback code (non-biometric).** For people whose palm doesn't work (injury, gloves, lighting) or who
are locked out:

1. Once per person: type the name and press **Set up code**. palmid shows a secret and an
   `otpauth://` link. Add it to any authenticator app (Google Authenticator, Microsoft
   Authenticator, Aegis, 1Password …) by typing the secret or turning the link into a QR code. It is
   shown once.
2. To use it: type the name and the 6-digit code from the app, press **Use code**.

The code has its own limit: 5 wrong codes lock it for 15 minutes. A code works once.

**Administrator tasks** (command line, from the `test1_palm` folder):

```bash
B=target/release/palmid
A="--gallery palmid_work/palm_gallery.json --auth palmid_work/auth_state.json"
$B $A auth-status                    # failures, locks, fallback set-up per person
$B $A auth-unlock --name asha        # clear asha's counts and locks
$B $A auth-setup-code --name asha    # new fallback secret (replaces the old one)
$B $A auth-code --name asha --code 123456
```

**From another application** (same computer): `POST /v1/auth/start {"user": "asha"}` starts an
attempt, `POST /v1/auth/status` returns the steps and progress or the last result (`last.ok`),
`POST /v1/auth/fallback {"user": "asha", "code": "123456"}` checks a code, `POST /v1/auth/cancel` stops.

Every start, result, code use, set-up and unlock is written to `palmid_work/auth_state.log.jsonl`.
`palmid_work/auth_state.json` holds the counters and the **fallback secrets**: anyone who can read it
can generate codes, so keep the folder private (palmid makes the file readable by your user only).

## 11. Chinese finger numbers

Show one hand to the camera with a number from 1 to 10. The **finger number** line in the Camera panel
shows it once you hold it steady for about 6 of 8 frames.

| Number | Hand shape |
|---|---|
| 1 | index finger up |
| 2 | index and middle up |
| 3 | index, middle and ring up — or middle, ring and little up |
| 4 | four fingers up, thumb folded in |
| 5 | open hand, all five spread |
| 6 | thumb and little finger out, others folded ("call me") |
| 7 | thumb, index and middle fingertips pinched together, others folded |
| 8 | thumb and index out in an "L", others folded |
| 9 | index finger hooked (bent at the middle joints), others folded |
| 10 | fist — or index and middle fingers crossed |

Tips:

* Keep the hand roughly upright and the palm (or back of the hand) facing the camera.
* Fists and pinches are hard for the palm finder; palmid keeps following the hand from the previous
  frame, so **start with an open hand, then form the number**.
* Half-made shapes show no number rather than a wrong one.

### Measure how well it reads your hands

1. In **4 · Finger numbers**, type a number and press **Record**. Hold that shape until the Camera panel
   says the 60 frames are done; move the hand around a little.
2. Repeat for each number you care about (at least two numbers, so the score can be computed).
3. Press **Score**. A table shows, per number, how often it was read correctly (recall), how often a
   reading of it was right (precision) and MCC, plus an overall line.

Only hand-point coordinates are saved, in `palmid_work/finger_recordings.jsonl`. Delete that file to
start over.

## 12. Face mesh

Every detected face gets a mesh of 468 points joined into triangles. It is for display only; palmid
never identifies faces. It works best with the face upright and facing the camera.

## 13. Using the API

The web app (and `palmid serve`) answers OpenAI-style requests at `/v1/chat/completions`. Send an
image; get back what palmid sees:

```bash
curl -s localhost:3000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "palmid-mono-yolox-nano-int8",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 photo.jpg)"'"}}]}]}'
```

The answer's `choices[0].message.content` is JSON text with one entry per object: `class`, `score`,
`bbox`, and where present `identity`, `hand` (21 points), `finger_number`, `face_mesh` (468 points).
Send a message with no image to get the current status instead.

## 14. Command line instead of the web app

```bash
cargo build --release -p palmid
B=target/release/palmid
W="--model palmid_work/models/palmid_int8.onnx --meta palmid_work/models/palmid_model.json --gallery palmid_work/palm_gallery.json"

$B $W live --source webcam --port 8361        # live view at http://localhost:8361/
$B $W enroll --name asha --consent yes --source webcam
$B $W identify --source webcam --max-frames 100
$B $W list
$B $W delete --name asha
$B $W serve --port 8361                       # API only
$B fingers-eval palmid_work/finger_recordings.jsonl   # score finger-number recordings
```

**Training on another machine** (for example with a GPU): copy `palmid_work/dataset` there, run

```bash
python3 trainer/train_student.py --data palmid_work/dataset --out out --epochs 60
```

then copy `out/palmid_int8.onnx` and `out/palmid_model.json` back into `palmid_work/models/` and
re-derive the thresholds **on the laptop** (results differ slightly between CPUs):

```bash
$B --model palmid_work/models/palmid_int8.onnx --meta palmid_work/models/palmid_model.json \
   calibrate --data palmid_work/dataset --split cal
```

## 15. Troubleshooting

| Problem | What to do |
|---|---|
| Video says "waiting for camera" | check `PALMID_SOURCE`. On WSL2 use `dshow:<camera name>` (§3). On Linux, `ls /dev/video*` lists cameras |
| Feed shows a crossed-out camera icon | the laptop's camera privacy key or shutter is on |
| Camera "busy" after a crash | Windows: `taskkill.exe /IM ffmpeg.exe /F`. Close other apps using the camera |
| `ffmpeg.exe` not found (WSL2) | `export PALMID_FFMPEG_WIN=/mnt/c/Users/<you>/AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe` |
| Build says Rust is too old | `rustup update`, then check `rustc --version` is 1.99 or newer |
| `pip install` refused ("externally managed") | you're outside the virtual environment: `source .venv/bin/activate` |
| Training fails with "No module named …" | `pip install -r trainer/requirements.txt` inside the venv, and set `PALMID_PYTHON` (§3) |
| Errors mentioning `libonnxruntime` | run the `export ORT_DYLIB_PATH=...` line (§3) in this terminal |
| MCC shows `n/a` | collect again, including frames with no face and no palm |
| Many false palm boxes | the model is under-trained; collect more and train longer |
| Enrollment refused | more light, 1280×720 capture, palm closer (§8) |
| Hand landmarks "failed" | the first start needs internet to download the hand model; restart once online |
| Very slow | use `--release`, set `PALMID_THREADS` to your core count, use 640×480 unless enrolling |
| Verify: "the hand left the view" | keep one hand in the frame for the whole attempt; move slowly between numbers |
| Verify: stuck at the palm step | the palm must match the claimed person: same hand as at enrollment, open, 25–40 cm, good light |
| Verify: a number isn't accepted | hold it steady about 2 seconds; between two numbers, change the shape (the previous one must be let go) |
| "locked for … seconds" | wait, use the fallback code, or an administrator runs `auth-unlock` |
| "no fallback code is set up" | press **Set up code** for that person (or `auth-setup-code`) |

## 16. Privacy and safe use

* Enroll only people who agree, and delete them when they ask.
* The gallery (`palm_gallery.json`) holds palm-line codes, not images. Treat it as personal data anyway.
* The web page has **no login** and listens only on this computer (`127.0.0.1`). Don't expose it to a
  network without adding authentication.
* The second-factor check (§10) stops photos but not replayed video, and palm accuracy on real palms
  hasn't been measured on a public dataset. Use it as one layer behind a first factor, not on its own,
  and not for doors or payments without a certified liveness sensor.
