//! palmid — monochrome face + palm detection and consent-based palmprint identification.
//!
//! Runtime port of the `VG360_PalmID_Mono_Webcam` notebook. Training (YOLOX-Nano with SAM)
//! stays in the notebook, which exports `palmid_int8.onnx` + `palmid_model.json`; this crate
//! loads them and does everything else on CPU.
//!
//! Every decision function that has a Lean 4 theorem lives in [`lean_mirror`] and is
//! property-tested against the theorem statements (see `lean/PalmID.lean`).

pub mod detector;
pub mod gallery;
pub mod imgproc;
pub mod lean_mirror;
pub mod metrics;
pub mod palmprint;
pub mod server;
pub mod camera;

pub use detector::{Detection, Detector, ModelMeta};
pub use gallery::{EnrollOutcome, Gallery, Identification};
pub use imgproc::GrayImage;
