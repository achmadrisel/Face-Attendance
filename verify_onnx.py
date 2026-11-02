"""Compare PyTorch model outputs vs ONNX model outputs on the same input.

The model now outputs 9 tensors (insightface-compatible format):
  cls_8/16/32  – raw logits  (H*W*2, 1)
  reg_8/16/32  – bbox dists  (H*W*2, 4)
  kps_8/16/32  – kps offsets (H*W*2, 10)

Usage:
    python verify_onnx.py
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from convert import SCRFD


ONNX_PATH  = 'models/scrfd_kps_0.5G.onnx'
PTH_PATH   = 'models/scrfd_kps_0.5G.pth'
INPUT_SIZE = 640
ATOL = 1e-4
RTOL = 1e-3

OUTPUT_NAMES = [
    'cls_8', 'cls_16', 'cls_32',
    'reg_8', 'reg_16', 'reg_32',
    'kps_8', 'kps_16', 'kps_32',
]


def load_pytorch_model():
    ckpt = torch.load(PTH_PATH, map_location='cpu', weights_only=False)
    model = SCRFD()
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    return model


def run_pytorch(model, x_np: np.ndarray):
    with torch.no_grad():
        outs = model(torch.from_numpy(x_np))
    return [o.numpy() for o in outs]  # list of 9 arrays


def run_onnx(x_np: np.ndarray):
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    return sess.run(None, {'input': x_np})  # list of 9 arrays


def compare(name, pt_out, onnx_out):
    max_abs = float(np.abs(pt_out - onnx_out).max())
    max_rel = float((np.abs(pt_out - onnx_out) / (np.abs(pt_out) + 1e-8)).max())
    close   = np.allclose(pt_out, onnx_out, atol=ATOL, rtol=RTOL)
    status  = '✓ MATCH' if close else '✗ MISMATCH'
    print(f"  {status}  {name:<8}  shape={pt_out.shape}"
          f"  max_abs={max_abs:.2e}  max_rel={max_rel:.2e}")
    return close


def main():
    rng  = np.random.default_rng(42)
    x_np = rng.standard_normal((1, 3, INPUT_SIZE, INPUT_SIZE)).astype(np.float32)

    print("Loading PyTorch model …")
    pt_model = load_pytorch_model()

    print("Running PyTorch forward pass …")
    pt_outs = run_pytorch(pt_model, x_np)

    print("Running ONNX forward pass …")
    onnx_outs = run_onnx(x_np)

    print()
    print(f"Comparing {len(OUTPUT_NAMES)} outputs (atol={ATOL}, rtol={RTOL}) …")
    all_ok = True
    for name, pt, onnx in zip(OUTPUT_NAMES, pt_outs, onnx_outs):
        all_ok &= compare(name, pt, onnx)

    print()
    if all_ok:
        print("All outputs match – ONNX export is correct.")
    else:
        print("Some outputs do NOT match – investigate above.")
        sys.exit(1)

    # Sanity-check: top-3 anchors by cls_8 score
    print()
    print("Top-3 anchors by cls_8 logit:")
    cls8 = pt_outs[0].ravel()
    reg8 = pt_outs[3]
    top3 = np.argsort(cls8)[-3:][::-1]
    for idx in top3:
        print(f"  anchor {idx:5d}  logit={cls8[idx]:.4f}"
              f"  sigmoid={1/(1+np.exp(-cls8[idx])):.4f}"
              f"  reg={reg8[idx].round(2)}")


if __name__ == '__main__':
    main()
