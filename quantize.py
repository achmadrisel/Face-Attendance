"""Post-training static quantization for glintr100.onnx → INT8.

Static quantization:
  - Weights AND activations are quantized to INT8 at conversion time.
  - Requires a small calibration dataset to pre-compute activation scales.
  - Faster than dynamic quantization on CPU because scale/zero-point are
    baked-in constants rather than computed per inference call.

Steps
-----
1. Pre-process  – fuse BatchNorm into Conv, infer shapes (required by ORT
                  quantizer to correctly propagate tensor types).
2. Calibrate    – run all facedata/ images through the pre-processed model
                  to collect MinMax activation statistics.
3. Quantize     – run quantize_static with QDQ format, QInt8.
4. Verify       – check outputs match (cosine similarity of embeddings).
5. Benchmark    – compare latency before and after quantization.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

# ── paths ────────────────────────────────────────────────────────────────────
SRC      = Path('models/glintr100.onnx')
PRE      = Path('models/glintr100_pre.onnx')          # pre-processed (intermediate)
QUANT    = Path('models/glintr100_static_int8.onnx')  # final static INT8 model
FACEDATA = Path('facedata')

WARMUP = 10
RUNS   = 100


# ── calibration data reader ───────────────────────────────────────────────────

def _load_image(path: Path) -> np.ndarray:
    """Load a face image, resize to 112×112, normalise to [-1, 1]."""
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 127.5 - 1.0   # [-1, 1]
    return img.transpose(2, 0, 1)[np.newaxis]     # (1, 3, 112, 112)


class FaceCalibReader(CalibrationDataReader):
    def __init__(self, model_path: Path, image_root: Path):
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        paths = sorted(p for p in image_root.rglob('*') if p.suffix.lower() in exts)
        if not paths:
            raise RuntimeError(f"No images found under {image_root}")
        print(f"      Calibration: {len(paths)} images from {image_root}")
        self._paths = iter(paths)
        sess = ort.InferenceSession(str(model_path),
                                    providers=['CPUExecutionProvider'])
        self._input_name = sess.get_inputs()[0].name

    def get_next(self):
        path = next(self._paths, None)
        if path is None:
            return None
        return {self._input_name: _load_image(path)}


# ── 1. Pre-process ────────────────────────────────────────────────────────────
def preprocess():
    print(f"[1/4] Pre-processing {SRC} …")
    print("      (fuses BN into Conv, infers shapes – required by quantizer)")
    quant_pre_process(
        input_model=str(SRC),
        output_model_path=str(PRE),
        skip_symbolic_shape=True,
    )
    print(f"      Saved: {PRE}")


# ── 2+3. Quantize (static) ────────────────────────────────────────────────────
def quantize():
    print(f"\n[2/4] Static quantization → {QUANT} …")
    quantize_static(
        model_input=str(PRE),
        model_output=str(QUANT),
        calibration_data_reader=FaceCalibReader(PRE, FACEDATA),
        quant_format=QuantFormat.QDQ,           # Q → op → DQ pairs
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
        reduce_range=False,
        extra_options={'ActivationSymmetric': False, 'WeightSymmetric': True},
    )
    print(f"      Saved: {QUANT}")

    src_mb   = SRC.stat().st_size  / 1024 / 1024
    quant_mb = QUANT.stat().st_size / 1024 / 1024
    print(f"      Size  FP32 {src_mb:.1f} MB  →  INT8 {quant_mb:.1f} MB"
          f"  ({(1 - quant_mb/src_mb)*100:.1f}% reduction)")


# ── 4. Verify ─────────────────────────────────────────────────────────────────
def verify():
    print("\n[3/4] Verifying numerical agreement …")
    onnx.checker.check_model(str(QUANT))

    rng  = np.random.default_rng(0)
    imgs = rng.standard_normal((4, 3, 112, 112)).astype(np.float32)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    sess_fp32 = ort.InferenceSession(str(SRC),   sess_options=opts,
                                     providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(str(QUANT), sess_options=opts,
                                     providers=['CPUExecutionProvider'])
    input_name = sess_fp32.get_inputs()[0].name

    for i in range(len(imgs)):
        x = imgs[i:i+1]
        emb_fp32 = sess_fp32.run(None, {input_name: x})[0]
        emb_int8 = sess_int8.run(None, {input_name: x})[0]

        cosine = float(
            np.dot(emb_fp32.ravel(), emb_int8.ravel())
            / (np.linalg.norm(emb_fp32) * np.linalg.norm(emb_int8) + 1e-8)
        )
        max_abs = float(np.abs(emb_fp32 - emb_int8).max())
        print(f"      img {i}: cosine_sim={cosine:.6f}  max_abs_diff={max_abs:.4f}")

    print("      Cosine similarity close to 1.0 → quantization is accurate.")


# ── 5. Benchmark ──────────────────────────────────────────────────────────────
def benchmark():
    print(f"\n[4/4] Benchmarking CPU  ({WARMUP} warm-up + {RUNS} timed runs) …")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess_fp32 = ort.InferenceSession(str(SRC),   sess_options=opts,
                                     providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(str(QUANT), sess_options=opts,
                                     providers=['CPUExecutionProvider'])
    input_name = sess_fp32.get_inputs()[0].name
    x = np.random.randn(1, 3, 112, 112).astype(np.float32)

    for sess in (sess_fp32, sess_int8):
        for _ in range(WARMUP):
            sess.run(None, {input_name: x})

    t0 = time.perf_counter()
    for _ in range(RUNS):
        sess_fp32.run(None, {input_name: x})
    ms_fp32 = (time.perf_counter() - t0) / RUNS * 1000

    t0 = time.perf_counter()
    for _ in range(RUNS):
        sess_int8.run(None, {input_name: x})
    ms_int8 = (time.perf_counter() - t0) / RUNS * 1000

    print(f"      FP32 : {ms_fp32:.2f} ms/img")
    print(f"      INT8 : {ms_int8:.2f} ms/img  ({ms_fp32/ms_int8:.2f}× speedup)")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    preprocess()
    quantize()
    verify()
    benchmark()
    print(f"\nDone.  Static INT8 model → {QUANT}")
