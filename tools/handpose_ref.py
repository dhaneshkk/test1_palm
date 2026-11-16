"""Reference 21-point landmarks for tests/handpose.rs: the upright-hand variant of opencv_zoo's
mp_handpose.py on a grayscale frame (gray replicated to 3 channels).
Usage: python tools/handpose_ref.py <gray.png> <x1,y1,x2,y2> <handpose.onnx> > ref.json"""
import sys, json, numpy as np, cv2, onnxruntime as ort
g = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE); box = [float(v) for v in sys.argv[2].split(',')]
sess = ort.InferenceSession(sys.argv[3], providers=['CPUExecutionProvider'])
img3 = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1; y1 -= 0.4 * h; y2 -= 0.4 * h
cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; hw, hh = w * 3 / 2, h * 3 / 2
b = np.array([cx - hw, cy - hh, cx + hw, cy + hh]).astype(np.int32)
b[[0, 2]] = b[[0, 2]].clip(0, img3.shape[1]); b[[1, 3]] = b[[1, 3]].clip(0, img3.shape[0])
crop = img3[b[1]:b[3], b[0]:b[2]]; s = max(crop.shape[:2]); ph, pw = s - crop.shape[0], s - crop.shape[1]
crop = cv2.copyMakeBorder(crop, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
blob = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
out = sess.run(None, {'input_1': blob[None]})
pts = out[0][0].reshape(21, 3); scale = max(b[2] - b[0], b[3] - b[1]) / 224
xy = (pts[:, :2] - 112) * scale + np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
print(json.dumps(dict(points=xy.tolist(), confidence=float(out[1][0][0]), crop_box=b.tolist())))
