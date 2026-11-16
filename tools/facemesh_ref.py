"""Reference face mesh + Delaunay triangulation for tests/facemesh.rs.
Crop = square of side ⌊1.5·max(w,h)⌋ centred on the face box, zero-padded, cv2.resize INTER_LINEAR to
192×192, gray replicated to RGB, /255 (NHWC). Usage:
  python tools/facemesh_ref.py <gray.png> <x1,y1,x2,y2> <face_landmark.onnx> > ref.json"""
import sys, json, math, numpy as np, cv2, onnxruntime as ort
from scipy.spatial import Delaunay
g = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE); x1, y1, x2, y2 = [float(v) for v in sys.argv[2].split(',')]
sess = ort.InferenceSession(sys.argv[3], providers=['CPUExecutionProvider'])
side = int(math.floor(1.5 * max(x2 - x1, y2 - y1))); x0 = int(math.floor((x1 + x2) / 2 - side / 2)); y0 = int(math.floor((y1 + y2) / 2 - side / 2))
sq = np.zeros((side, side), np.uint8)
sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(g.shape[1], x0 + side), min(g.shape[0], y0 + side)
sq[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = g[sy0:sy1, sx0:sx1]
crop = cv2.resize(sq, (192, 192), interpolation=cv2.INTER_LINEAR)
x = np.repeat(crop[..., None], 3, 2).astype(np.float32)[None] / 255.0
lm, pres = sess.run(None, {'input_1': x})
p = lm.reshape(468, 3); pts = np.c_[p[:, :2] * side / 192 + [x0, y0], p[:, 2] * side / 192]
tri = Delaunay(pts[:, :2]).simplices
print(json.dumps(dict(points=pts.tolist(), presence=float(1 / (1 + np.exp(-pres.item()))), crop=[x0, y0, side],
                      triangles=sorted(sorted(map(int, t)) for t in tri))))
