"""Quantize scrfd_500m_bnkps.onnx to INT8.

Static quantization (default): uses real face images for calibration → better accuracy.
Dynamic quantization (--dynamic): quantizes weights only, no calibration images needed.

Usage:
    python quantize_scrfd.py                          # static, uses facedata/
    python quantize_scrfd.py --calib-dir ./images     # static, custom image dir
    python quantize_scrfd.py --dynamic                # dynamic (no calibration)
"""

import argparse
import glob
import os

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

# scrfd_500m_bnkps input spec
_INPUT_NAME  = "input.1"
_INPUT_HW    = 640


def _preprocess(img_path: str) -> np.ndarray | None:
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (_INPUT_HW, _INPUT_HW))
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0          # insightface normalisation
    img = img.transpose(2, 0, 1)[None]   # HWC → NCHW
    return img.astype(np.float32)


class _CalibReader(CalibrationDataReader):
    def __init__(self, calib_dir: str, max_samples: int):
        paths: list[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
        paths = sorted(paths)[:max_samples]
        if not paths:
            raise FileNotFoundError(f"No images found under {calib_dir!r}")
        print(f"  Calibration images: {len(paths)}")
        self._paths = paths
        self._iter  = iter(self._paths)

    def get_next(self) -> dict | None:
        while True:
            try:
                path = next(self._iter)
            except StopIteration:
                return None
            img = _preprocess(path)
            if img is not None:
                return {_INPUT_NAME: img}


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8-quantize scrfd_500m_bnkps.onnx")
    parser.add_argument("--input",       default="models/scrfd_500m_bnkps.onnx")
    parser.add_argument("--output",      default="models/scrfd_500m_bnkps_int8.onnx")
    parser.add_argument("--calib-dir",   default="faces",
                        help="Directory tree of calibration images (static mode only)")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max calibration images to use")
    parser.add_argument("--dynamic",     action="store_true",
                        help="Use dynamic quantization (weights only, no calibration needed)")
    args = parser.parse_args()

    if args.dynamic:
        print(f"Dynamic quantization: {args.input} → {args.output}")
        quantize_dynamic(
            args.input,
            args.output,
            weight_type=QuantType.QInt8,
        )
        print(f"Done → {args.output}")
        return

    # ── Static quantization ───────────────────────────────────────────────────
    prep_path = args.input.replace(".onnx", "_prep.onnx")

    print(f"Step 1/3  Pre-processing model …")
    quant_pre_process(args.input, prep_path, skip_optimization=False)

    print(f"Step 2/3  Collecting calibration data from {args.calib_dir!r} …")
    reader = _CalibReader(args.calib_dir, max_samples=args.max_samples)

    print(f"Step 3/3  Quantizing …")
    quantize_static(
        prep_path,
        args.output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=True,   # safer on CPUs without AVX512-VNNI (e.g. ARM, older x86)
    )

    os.remove(prep_path)
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
