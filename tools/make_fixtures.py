"""Generate OpenCV/onnxruntime reference outputs for tests/parity.rs.
Usage: python tools/make_fixtures.py <frame.png> <palm_x1,y1,x2,y2> <test_model_int8.onnx> <out_dir>"""
import sys, json, numpy as np, cv2, onnxruntime as ort
frame, box, model, out = sys.argv[1], [float(v) for v in sys.argv[2].split(',')], sys.argv[3], sys.argv[4]
g = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2GRAY); cv2.imwrite(f'{out}/gray.png', g)
cl = cv2.createCLAHE(2.0, (8, 8)); c = cl.apply(g); cv2.imwrite(f'{out}/clahe.png', c)
S = 416; H, W = g.shape; r = min(S / H, S / W)
rz = cv2.resize(c, (int(W * r), int(H * r)), interpolation=cv2.INTER_LINEAR)
pad = np.full((S, S), 114, np.uint8); pad[:rz.shape[0], :rz.shape[1]] = rz; cv2.imwrite(f'{out}/letterbox.png', pad)
x1, y1, x2, y2 = box; side = min(x2 - x1, y2 - y1); cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, 0.35 * side
a, b_, c_, d_ = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
roi = cv2.resize(cl.apply(g[b_:d_, a:c_]), (128, 128), interpolation=cv2.INTER_AREA); cv2.imwrite(f'{out}/roi.png', roi)
bl = cv2.GaussianBlur(roi, (0, 0), 1.0); cv2.imwrite(f'{out}/roi_blur.png', bl)
ks = []
for j in range(6):
    k = cv2.getGaborKernel((17, 17), 3.5, j * np.pi / 6, 9.0, 0.7, 0, ktype=cv2.CV_32F); ks.append(k - k.mean())
R = np.stack([cv2.filter2D(bl.astype(np.float32), -1, k) for k in ks]); code = R.argmin(0)[8:-8:4, 8:-8:4]
open(f'{out}/compcode.txt', 'w').write(''.join(str(v) for v in code.ravel()))
x = np.repeat(pad[None], 3, 0).astype(np.float32)[None]
raw = ort.InferenceSession(model, providers=['CPUExecutionProvider']).run(None, {'images': x})[0][0]
raw.astype('<f4').tofile(f'{out}/raw.bin')
json.dump(dict(box=box, ratio=r, raw_shape=list(raw.shape)), open(f'{out}/meta.json', 'w'))
print('ok', raw.shape, code.shape)
# notebook decode (per-class τ, cv2.dnn.NMSBoxes) at a low floor so there is something to compare
G, ST = [], []
for st in (8, 16, 32):
    h_ = S // st; ys, xs = np.meshgrid(np.arange(h_), np.arange(h_), indexing='ij')
    G.append(np.stack([xs, ys], -1).reshape(-1, 2)); ST.append(np.full((h_ * h_, 1), st))
G, ST = np.concatenate(G).astype(np.float32), np.concatenate(ST).astype(np.float32)
tau = np.array([0.02, 0.02])
xy = (raw[:, :2] + G) * ST; wh = np.exp(np.clip(raw[:, 2:4], -10, 10)) * ST
cls = raw[:, 5:].argmax(1); sc = raw[:, 4] * raw[np.arange(len(raw)), 5 + cls]; m = sc >= tau[cls]
bx = (np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r)[m]; sc, cls = sc[m], cls[m]; dets = []
for c in np.unique(cls):
    k = np.where(cls == c)[0]
    keep = np.array(cv2.dnn.NMSBoxes([[*q[:2], *(q[2:] - q[:2])] for q in bx[k]], sc[k].tolist(), 0.0, 0.45), np.int64).reshape(-1)
    for i in keep: dets.append(dict(cls=int(c), score=float(sc[k][i]), bbox=[float(v) for v in bx[k][i]]))
json.dump(dict(tau=tau.tolist(), dets=dets), open(f'{out}/decode.json', 'w'))
print('decode dets', len(dets))
