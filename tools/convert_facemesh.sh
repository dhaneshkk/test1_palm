#!/usr/bin/env bash
# Rebuild assets/models/face_landmark.onnx from the Apache-2.0 mediapipe wheel.
set -euo pipefail
out="${1:-assets/models/face_landmark.onnx}"
tmp=$(mktemp -d)
python3 -m venv "$tmp/venv"
"$tmp/venv/bin/pip" -q install "tensorflow-cpu==2.21.*" "tf2onnx==1.17.*"
"$tmp/venv/bin/pip" -q download --no-deps "mediapipe==0.10.14" -d "$tmp/whl" --only-binary=:all: --python-version 3.11 --platform manylinux_2_17_x86_64
python3 -c "import zipfile,glob,sys; z=zipfile.ZipFile(glob.glob(sys.argv[1]+'/*.whl')[0]); open(sys.argv[2],'wb').write(z.read('mediapipe/modules/face_landmark/face_landmark.tflite'))" "$tmp/whl" "$tmp/face_landmark.tflite"
"$tmp/venv/bin/python" -m tf2onnx.convert --tflite "$tmp/face_landmark.tflite" --opset 17 --output "$out"
rm -rf "$tmp"
echo "wrote $out"
