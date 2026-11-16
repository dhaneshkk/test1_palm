# face_landmark.onnx

* Source: `mediapipe/modules/face_landmark/face_landmark.tflite` from the `mediapipe==0.10.14`
  Python wheel (Google, https://github.com/google-ai-edge/mediapipe).
* Licence: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Copyright Google LLC.
* Modification: converted from TFLite to ONNX with tf2onnx (opset 17), weights unchanged; see
  `tools/convert_facemesh.sh`. Converted outputs match the TFLite interpreter within 1.2e-4.
* IO: input `input_1` [1,192,192,3] float in [0,1]; outputs `conv2d_21` [1,1,1,1404]
  (468 × x,y,z in 192-pixel crop coordinates) and `conv2d_31` [1,1,1,1] (face-presence logit).
