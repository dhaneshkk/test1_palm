#!/usr/bin/env python3
"""Train the monochrome YOLOX-Nano student with SAM and export INT8 ONNX for the Rust runtime.

Driven by the palmid web app (or by hand). Reads the YOLO-format dataset the app writes
(images/{train,cal}/*.png grayscale, labels/{split}/*.txt) and prints ONE JSON object per line
to stdout so the app can show live progress:

  {"event":"start", ...}  {"event":"iter","iter":..,"total":..,"loss":..,"lr":..}
  {"event":"epoch","epoch":..,"epochs":..,"loss":..,"seconds":..}  {"event":"export","stage":..}
  {"event":"done","onnx":"...","seconds":..}  |  {"event":"error","message":"..."}

Per-class thresholds (Mondrian split-conformal) and MCC are computed afterwards in Rust
(`palmid calibrate` / `palmid eval`), on the same CPU that will run inference.

Usage: python trainer/train_student.py --data DIR --out DIR [--epochs 30] [--batch 16]
       [--yolox PATH_TO_YOLOX_CHECKOUT] [--weights yolox_nano.pth]
"""
import argparse, glob, json, math, os, sys, time, traceback, urllib.request, subprocess

def emit(**kw):
    print(json.dumps(kw), flush=True)

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=0.01); ap.add_argument('--rho', type=float, default=0.05)
ap.add_argument('--warmup', type=int, default=200); ap.add_argument('--classes', default='face,palm')
ap.add_argument('--yolox', default=os.environ.get('PALMID_YOLOX', os.path.expanduser('~/.cache/palmid/YOLOX')))
ap.add_argument('--weights', default=os.environ.get('PALMID_YOLOX_WEIGHTS', os.path.expanduser('~/.cache/palmid/yolox_nano.pth')))
ap.add_argument('--threads', type=int, default=0); ap.add_argument('--seed', type=int, default=0)
A = ap.parse_args()

try:
    os.makedirs(A.out, exist_ok=True)
    if not os.path.isdir(os.path.join(A.yolox, 'yolox')):
        emit(event='status', message=f'cloning YOLOX into {A.yolox}')
        subprocess.check_call(['git', 'clone', '-q', '--depth', '1', 'https://github.com/Megvii-BaseDetection/YOLOX.git', A.yolox])
    if not os.path.exists(A.weights):
        emit(event='status', message='downloading yolox_nano.pth (COCO init)')
        os.makedirs(os.path.dirname(A.weights), exist_ok=True)
        urllib.request.urlretrieve('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth', A.weights)
    sys.path.insert(0, A.yolox)
    import numpy as np, cv2, torch, torch.nn as nn
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
    if A.threads: torch.set_num_threads(A.threads)
    np.random.seed(A.seed); torch.manual_seed(A.seed)
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    CLASSES = A.classes.split(','); NC, S = len(CLASSES), 416

    def load_split(split):
        items = []
        for p in sorted(glob.glob(f'{A.data}/images/{split}/*.png')):
            lp = p.replace('/images/', '/labels/')[:-4] + '.txt'
            lab = np.loadtxt(lp, ndmin=2).astype(np.float32) if os.path.exists(lp) and os.path.getsize(lp) else np.zeros((0, 5), np.float32)
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE); H, W = g.shape
            b = np.zeros((0, 4), np.float32); c = np.zeros(0, np.int64)
            if len(lab):
                cl, cx, cy, w, h = lab.T
                b = np.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], 1).astype(np.float32); c = cl.astype(np.int64)
            items.append((g, b, c))
        return items

    TRAIN, CAL = load_split('train'), load_split('cal')
    if len(TRAIN) < 8:
        raise RuntimeError(f'only {len(TRAIN)} training frames in {A.data} — collect more')
    emit(event='start', device=DEV, train=len(TRAIN), cal=len(CAL), epochs=A.epochs, batch=A.batch, optimizer='SAM(SGD-nesterov)', rho=A.rho,
         labels={CLASSES[k]: int(sum((c == k).sum() for _, _, c in TRAIN)) for k in range(NC)})

    _CLAHE = cv2.createCLAHE(2.0, (8, 8))
    def letterbox(gray):
        H, W = gray.shape; r = min(S / H, S / W); g = cv2.resize(_CLAHE.apply(gray), (int(W * r), int(H * r)))
        pad = np.full((S, S), 114, np.uint8); pad[:g.shape[0], :g.shape[1]] = g
        return np.repeat(pad[None], 3, 0).astype(np.float32), r

    def mono_augment(g, b, c, rng):  # intensity / texture / geometry only — no HSV on grayscale
        H, W = g.shape; g = np.clip(g.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.uniform(-30, 30), 0, 255)
        if rng.random() < 0.3: g = 255 * (g / 255) ** rng.uniform(0.6, 1.6)
        g = g.astype(np.uint8)
        if rng.random() < 0.3: g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.5, 1.5))
        if rng.random() < 0.3: g = np.clip(g + rng.normal(0, rng.uniform(3, 10), g.shape), 0, 255).astype(np.uint8)
        s, tx, ty = rng.uniform(0.7, 1.3), rng.uniform(-.1, .1) * W, rng.uniform(-.1, .1) * H
        M = np.float32([[s, 0, tx + (1 - s) * W / 2], [0, s, ty + (1 - s) * H / 2]]); g = cv2.warpAffine(g, M, (W, H), borderValue=114)
        if len(b):
            b = b * s; b[:, [0, 2]] += M[0, 2]; b[:, [1, 3]] += M[1, 2]
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, W - 1); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H - 1)
            keep = ((b[:, 2] - b[:, 0]) > 4) & ((b[:, 3] - b[:, 1]) > 4); b, c = b[keep], c[keep]
        if rng.random() < 0.5:
            g = g[:, ::-1].copy()
            if len(b): b = b.copy(); b[:, [0, 2]] = W - 1 - b[:, [2, 0]]
        return g, b, c

    def make_batch(items, rng):
        X = np.zeros((len(items), 3, S, S), np.float32); T = np.zeros((len(items), 30, 5), np.float32)
        for i, (g, b, c) in enumerate(items):
            g, b, c = mono_augment(g, b.copy(), c.copy(), rng)
            X[i], r = letterbox(g); b = b * r; m = min(len(b), 30)
            if m: T[i, :m] = np.c_[c[:m], (b[:m, 0] + b[:m, 2]) / 2, (b[:m, 1] + b[:m, 3]) / 2, b[:m, 2] - b[:m, 0], b[:m, 3] - b[:m, 1]]
        return torch.from_numpy(X), torch.from_numpy(T)

    def build_nano():
        inch = [256, 512, 1024]
        m = YOLOX(YOLOPAFPN(0.33, 0.25, in_channels=inch, depthwise=True), YOLOXHead(NC, 0.25, in_channels=inch, depthwise=True))
        for x in m.modules():
            if isinstance(x, nn.BatchNorm2d): x.eps, x.momentum = 1e-3, 0.03
        m.head.initialize_biases(1e-2); return m

    class SAM(torch.optim.Optimizer):
        """Sharpness-Aware Minimisation (Foret et al. 2021): w ← w − η∇L(w + ρ·g/‖g‖)."""
        def __init__(self, params, base, rho=0.05, **kw):
            super().__init__(params, dict(rho=rho, **kw)); self.base_optimizer = base(self.param_groups, **kw)
            self.param_groups = self.base_optimizer.param_groups; self.defaults.update(self.base_optimizer.defaults)
        @torch.no_grad()
        def first_step(self):
            gn = torch.norm(torch.stack([p.grad.norm(2) for g in self.param_groups for p in g['params'] if p.grad is not None]), 2)
            for g in self.param_groups:
                for p in g['params']:
                    if p.grad is None: continue
                    self.state[p]['old_p'] = p.data.clone(); p.add_(p.grad * (g['rho'] / (gn + 1e-12)))
        @torch.no_grad()
        def second_step(self):
            for g in self.param_groups:
                for p in g['params']:
                    if p.grad is not None: p.data = self.state[p]['old_p']
            self.base_optimizer.step()

    class bn_frozen:  # BN running stats must describe w, not the perturbed w + ε
        def __init__(self, m): self.bn = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
        def __enter__(self): self.mo = [b.momentum for b in self.bn]; [setattr(b, 'momentum', 0.0) for b in self.bn]
        def __exit__(self, *a): [setattr(b, 'momentum', m) for b, m in zip(self.bn, self.mo)]

    model = build_nano()
    ck = torch.load(A.weights, map_location='cpu', weights_only=False)['model']; sd = model.state_dict()
    model.load_state_dict({k: v for k, v in ck.items() if k in sd and v.shape == sd[k].shape}, strict=False); model.to(DEV)
    dec = [p for p in model.parameters() if p.ndim > 1]; nod = [p for p in model.parameters() if p.ndim <= 1]
    opt = SAM([dict(params=dec, weight_decay=5e-4), dict(params=nod, weight_decay=0.0)], torch.optim.SGD, rho=A.rho, lr=A.lr, momentum=0.9, nesterov=True)
    ema = build_nano().to(DEV); ema.load_state_dict(model.state_dict()); ema.eval()

    ipe = max(1, len(TRAIN) // A.batch); total = ipe * A.epochs; it = 0; rng = np.random.default_rng(A.seed + 1); t0 = time.time()
    warm = min(A.warmup, max(1, total // 5))
    for ep in range(A.epochs):
        model.train(); idx = rng.permutation(len(TRAIN)); L = []
        for bi in range(ipe):
            X, T = make_batch([TRAIN[j] for j in idx[bi * A.batch:(bi + 1) * A.batch]], rng); X, T = X.to(DEV), T.to(DEV)
            lr = A.lr * (it / warm) ** 2 if it < warm else A.lr * (0.05 + 0.475 * (1 + math.cos(math.pi * (it - warm) / max(1, total - warm))))
            for g in opt.param_groups: g['lr'] = lr
            loss = model(X, T)['total_loss']; opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 10); opt.first_step(); opt.zero_grad()
            with bn_frozen(model): model(X, T)['total_loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10); opt.second_step(); opt.zero_grad()
            with torch.no_grad():
                d = min(0.9998, (1 + it) / (10 + it)); msd = model.state_dict()
                for k_, v in ema.state_dict().items():
                    if v.dtype.is_floating_point: v.mul_(d).add_(msd[k_].detach(), alpha=1 - d)
                    else: v.copy_(msd[k_])
            L.append(loss.item()); it += 1
            emit(event='iter', iter=it, total=total, loss=round(L[-1], 4), lr=lr, seconds=round(time.time() - t0, 1))
        emit(event='epoch', epoch=ep + 1, epochs=A.epochs, loss=round(float(np.mean(L)), 4), seconds=round(time.time() - t0, 1))

    emit(event='export', stage='onnx_fp32')
    ema_cpu = build_nano(); ema_cpu.load_state_dict(ema.state_dict()); ema_cpu.eval(); ema_cpu.head.decode_in_inference = False
    F32, PRE, I8 = (os.path.join(A.out, f) for f in ('palmid_fp32.onnx', 'palmid_pre.onnx', 'palmid_int8.onnx'))
    kw = dict(input_names=['images'], output_names=['output'], opset_version=17)
    try: torch.onnx.export(ema_cpu, torch.zeros(1, 3, S, S), F32, dynamo=False, **kw)
    except TypeError: torch.onnx.export(ema_cpu, torch.zeros(1, 3, S, S), F32, **kw)
    emit(event='export', stage='int8_quantize')
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    quant_pre_process(F32, PRE, skip_symbolic_shape=True)
    class _Cal(CalibrationDataReader):
        def __init__(s, items): s.it = iter([letterbox(g)[0][None] for g, _, _ in items])
        def get_next(s): x = next(s.it, None); return None if x is None else {'images': x}
    quantize_static(PRE, I8, _Cal((CAL or TRAIN)[:64]), quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    os.remove(PRE)
    json.dump(dict(classes=CLASSES, input_size=S, tau=[0.30] * NC, score_floor=0.05, nms_iou=0.45), open(os.path.join(A.out, 'palmid_model.json'), 'w'), indent=1)
    emit(event='done', onnx=I8, meta=os.path.join(A.out, 'palmid_model.json'), seconds=round(time.time() - t0, 1), iters=it)
except Exception as e:
    emit(event='error', message=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    sys.exit(1)
