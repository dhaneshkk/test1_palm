# Reference outputs for tests/teachers.rs: python tools/teacher_ref.py <image> > ref.json (writes /tmp/ref_gray.png)
import cv2, numpy as np, json, sys
img = cv2.imread(sys.argv[1]); out = {}
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); cv2.imwrite('/tmp/ref_gray.png', g)
fd = cv2.FaceDetectorYN.create('/tmp/zoo/face_detection_yunet_2023mar.onnx', '', (320, 320), 0.7, 0.3, 5000)
fd.setInputSize((img.shape[1], img.shape[0])); _, f = fd.detect(img)
out['faces'] = [] if f is None else [[float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3])] for r in f]
sys.path.insert(0, '/home/claude/work'); exec(open('/home/claude/work/palm_test.py').read().split("if __name__")[0])
b, s = palm_detect(img, 0.6, 0.3); out['palms'] = [[float(v) for v in q] for q in b]
print(json.dumps(out))
