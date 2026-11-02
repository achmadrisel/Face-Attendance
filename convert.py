"""Convert scrfd_kps_0.5G.pth to ONNX compatible with insightface's SCRFD class.

Architecture reconstructed from checkpoint state_dict keys and training config:
  Backbone : MobileNetV1 (stem + 4 layers of DW-Sep blocks)
  Neck     : PAFPN (3 lateral + 3 fpn + 2 downsample + 2 pafpn convs)
  Head     : SCRFDHead (per-stride DW-Sep shared convs + cls/reg/kps predictors)

Output format (9 tensors, non-batched 2-D):
  insightface reads them as net_outs[idx], net_outs[idx+fmc], net_outs[idx+fmc*2]
  where fmc=3 and _feat_stride_fpn=[8,16,32], _num_anchors=2, use_kps=True.

  Index  Name    Shape           Description
  -----  ------  --------------  -----------------------------------
  0      cls_8   (6400*2, 1)     sigmoid class score, stride 8
  1      cls_16  (1600*2, 1)     sigmoid class score, stride 16
  2      cls_32  ( 400*2, 1)     sigmoid class score, stride 32
  3      reg_8   (6400*2, 4)     bbox distances (l,t,r,b), stride 8
  4      reg_16  (1600*2, 4)     bbox distances, stride 16
  5      reg_32  ( 400*2, 4)     bbox distances, stride 32
  6      kps_8   (6400*2, 10)    kps offsets (x0,y0,…,x4,y4), stride 8
  7      kps_16  (1600*2, 10)    kps offsets, stride 16
  8      kps_32  ( 400*2, 10)    kps offsets, stride 32

  N = H*W for each stride level (H=W=640/stride).
  Distances and offsets are raw (NOT multiplied by stride) — insightface
  applies the stride scaling and anchor-center decoding itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────
# Backbone
# ──────────────────────────────────────────────────────────

def _dw_sep(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """DW-Sep block whose state-dict indices match the checkpoint.

    Index mapping (ReLU has no params so not in state-dict):
      0: DW Conv2d   1: BN   2: ReLU
      3: PW Conv2d   4: BN   5: ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MobileNetV1(nn.Module):
    """Backbone matching stage_blocks=(2,3,2,6), stage_planes=[16,16,40,72,152,288].

    Output strides: layer1=4, layer2=8, layer3=16, layer4=32.
    Returns (c2, c3, c4) used by the PAFPN neck.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            ),
            _dw_sep(16, 16, stride=1),
        )
        self.layer1 = nn.Sequential(_dw_sep(16, 40, stride=2), _dw_sep(40, 40))
        self.layer2 = nn.Sequential(_dw_sep(40, 72, stride=2), _dw_sep(72, 72), _dw_sep(72, 72))
        self.layer3 = nn.Sequential(_dw_sep(72, 152, stride=2), _dw_sep(152, 152))
        self.layer4 = nn.Sequential(
            _dw_sep(152, 288, stride=2),
            _dw_sep(288, 288), _dw_sep(288, 288),
            _dw_sep(288, 288), _dw_sep(288, 288), _dw_sep(288, 288),
        )

    def forward(self, x):
        x  = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c2, c3, c4


# ──────────────────────────────────────────────────────────
# Neck
# ──────────────────────────────────────────────────────────

class _ConvModule(nn.Module):
    """Conv2d(bias=True) + ReLU – no BN, matching neck state-dict (.conv attr)."""

    def __init__(self, in_ch, out_ch, kernel=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              padding=kernel // 2, bias=True)

    def forward(self, x):
        return F.relu(self.conv(x))


class PAFPN(nn.Module):
    """Path Aggregation FPN: in_channels=[72,152,288], out_channels=16, num_outs=3."""

    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            _ConvModule(72, 16, 1),
            _ConvModule(152, 16, 1),
            _ConvModule(288, 16, 1),
        ])
        self.fpn_convs = nn.ModuleList([_ConvModule(16, 16, 3) for _ in range(3)])
        self.downsample_convs = nn.ModuleList([_ConvModule(16, 16, 3, stride=2) for _ in range(2)])
        self.pafpn_convs = nn.ModuleList([_ConvModule(16, 16, 3) for _ in range(2)])

    def forward(self, c2, c3, c4):
        lat = [self.lateral_convs[i](f) for i, f in enumerate([c2, c3, c4])]
        lat[1] = lat[1] + F.interpolate(lat[2], scale_factor=2, mode='nearest')
        lat[0] = lat[0] + F.interpolate(lat[1], scale_factor=2, mode='nearest')
        outs = [self.fpn_convs[i](lat[i]) for i in range(3)]
        outs[1] = outs[1] + self.downsample_convs[0](outs[0])
        outs[2] = outs[2] + self.downsample_convs[1](outs[1])
        outs[1] = self.pafpn_convs[0](outs[1])
        outs[2] = self.pafpn_convs[1](outs[2])
        return outs  # [stride-8, stride-16, stride-32]


# ──────────────────────────────────────────────────────────
# Head
# ──────────────────────────────────────────────────────────

class _ConvModuleBN(nn.Module):
    """Conv2d(bias=False) + BN + ReLU – matching head DW-Sep sub-blocks."""

    def __init__(self, in_ch, out_ch, kernel=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class _DWSepModule(nn.Module):
    """Matches mmcv DepthwiseSeparableConvModule (.depthwise_conv, .pointwise_conv attrs)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise_conv = _ConvModuleBN(in_ch, in_ch, kernel=3, groups=in_ch)
        self.pointwise_conv = _ConvModuleBN(in_ch, out_ch, kernel=1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class _Integral(nn.Module):
    """Kept for checkpoint compatibility (not used in inference when loss_dfl=False)."""

    def __init__(self, reg_max=8):
        super().__init__()
        self.register_buffer('project', torch.linspace(0, reg_max, reg_max + 1))

    def forward(self, x):
        return x


class SCRFDHead(nn.Module):
    """Head with strides_share=False, dw_conv=True, cls_reg_share=True, use_kps=True."""

    def __init__(self):
        super().__init__()
        _strides = ['(8, 8)', '(16, 16)', '(32, 32)']

        self.cls_stride_convs = nn.ModuleDict({
            s: nn.Sequential(_DWSepModule(16, 64), _DWSepModule(64, 64))
            for s in _strides
        })
        self.stride_cls = nn.ModuleDict({s: nn.Conv2d(64, 2,  3, padding=1) for s in _strides})
        self.stride_reg = nn.ModuleDict({s: nn.Conv2d(64, 8,  3, padding=1) for s in _strides})
        self.stride_kps = nn.ModuleDict({s: nn.Conv2d(64, 20, 3, padding=1) for s in _strides})
        self.integral = _Integral(reg_max=8)


# ──────────────────────────────────────────────────────────
# Full model – raw head outputs for insightface
# ──────────────────────────────────────────────────────────

class SCRFD(nn.Module):
    """SCRFD wrapper that outputs 9 raw tensors expected by insightface.SCRFD.

    insightface detects 9 outputs → fmc=3, use_kps=True, _num_anchors=2.
    Output order: [cls_8, cls_16, cls_32, reg_8, reg_16, reg_32, kps_8, kps_16, kps_32]

    Shapes (batch=1 squeezed, non-batched path in insightface):
      cls_*  (H*W*2, 1)   sigmoid score per anchor
      reg_*  (H*W*2, 4)   bbox distances  (l, t, r, b)  — raw, insightface scales by stride
      kps_*  (H*W*2, 10)  kps offsets (x0,y0,…,x4,y4)  — raw, insightface scales by stride
    """

    def __init__(self):
        super().__init__()
        self.backbone  = MobileNetV1()
        self.neck      = PAFPN()
        self.bbox_head = SCRFDHead()

    def _head_output(self, feat: torch.Tensor, key: str):
        """Return (cls, reg, kps) raw outputs for one FPN level, squeezed to 2-D."""
        B = feat.shape[0]
        shared  = self.bbox_head.cls_stride_convs[key](feat)
        cls_raw = self.bbox_head.stride_cls[key](shared)   # (1, 2,  H, W)
        reg_raw = self.bbox_head.stride_reg[key](shared)   # (1, 8,  H, W)
        kps_raw = self.bbox_head.stride_kps[key](shared)   # (1, 20, H, W)

        # (1, C, H, W) → (H*W*num_anchors, C_per_anchor), squeeze batch dim
        # insightface uses scores directly (no sigmoid at runtime) → bake sigmoid into ONNX
        cls = torch.sigmoid(cls_raw).permute(0, 2, 3, 1).reshape(B, -1, 1).squeeze(0)
        reg = reg_raw.permute(0, 2, 3, 1).reshape(B, -1, 4).squeeze(0)
        kps = kps_raw.permute(0, 2, 3, 1).reshape(B, -1, 10).squeeze(0)
        return cls, reg, kps

    def forward(self, x: torch.Tensor):
        c2, c3, c4 = self.backbone(x)
        p3, p4, p5 = self.neck(c2, c3, c4)

        cls8,  reg8,  kps8  = self._head_output(p3, '(8, 8)')
        cls16, reg16, kps16 = self._head_output(p4, '(16, 16)')
        cls32, reg32, kps32 = self._head_output(p5, '(32, 32)')

        # insightface order: cls×3, reg×3, kps×3
        return cls8, cls16, cls32, reg8, reg16, reg32, kps8, kps16, kps32


# ──────────────────────────────────────────────────────────
# Conversion entry-point
# ──────────────────────────────────────────────────────────

def main():
    import onnx

    pth_path   = 'models/scrfd_kps_0.5G.pth'
    onnx_path  = 'models/scrfd_kps_0.5G.onnx'
    input_size = 640

    print(f"Loading checkpoint: {pth_path}")
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)

    print("Building model …")
    model = SCRFD()
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)

    ckpt_missing = [k for k in missing]
    if ckpt_missing:
        print(f"  WARNING – keys in model but NOT in checkpoint ({len(ckpt_missing)}):")
        for k in ckpt_missing[:10]:
            print(f"    {k}")
    if unexpected:
        print(f"  WARNING – keys in checkpoint but NOT in model ({len(unexpected)}):")
        for k in unexpected[:10]:
            print(f"    {k}")

    model.eval()
    dummy = torch.zeros(1, 3, input_size, input_size)

    with torch.no_grad():
        outs = model(dummy)
    print(f"Forward pass OK  outputs: {[tuple(o.shape) for o in outs]}")
    assert len(outs) == 9, f"Expected 9 outputs, got {len(outs)}"

    output_names = [
        'cls_8',  'cls_16',  'cls_32',
        'reg_8',  'reg_16',  'reg_32',
        'kps_8',  'kps_16',  'kps_32',
    ]

    print(f"Exporting to {onnx_path} …")
    with torch.no_grad():
        torch.onnx.export(
            model, dummy,
            onnx_path,
            input_names=['input'],
            output_names=output_names,
        )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX check passed.")
    print("Done.")


if __name__ == '__main__':
    main()
