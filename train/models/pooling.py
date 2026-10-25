"""GeM (Generalized Mean) pooling — thay average pooling ở đầu head.

Căn cứ: dùng phổ biến trong person re-ID/retrieval (bài toán rất gần PAR —
cùng là crop người, cùng cần feature bền với thay đổi góc nhìn/tư thế), báo
cáo tăng khoảng 1-4% mAP/rank-1 so với GAP/GMP. GeM là suy rộng của average
pooling (p=1) và max pooling (p→∞), p học được qua backprop.

    GeM(x) = ( mean(x.clamp(eps) ** p) ) ** (1/p)

ĐÃ XÁC MINH bằng timm thật (không phải đoán): tạo backbone với
`global_pool=''` cho ra:
  - CNN (ConvNeXt-B, 256×128 input): feature map (B, C, H, W) = (B, 1024, 8, 4)
  - ViT (CLIP ViT-L/14, 224 input): token sequence (B, N, C) = (B, 257, 1024),
    token đầu là CLS.
GeM dưới đây xử lý cả hai dạng. Với token sequence, mặc định loại CLS token
trước khi pool (pool theo patch token, không lẫn feature "toàn cục" của CLS
vào công thức mean-power — hợp lý hơn nhưng CHƯA kiểm chứng bằng LOO, so sánh
với pooling='avg' trước khi tin).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True,
                 exclude_cls_token: bool = True):
        super().__init__()
        init_p = torch.tensor(float(p))
        self.p = nn.Parameter(init_p) if learn_p else init_p
        self.eps = eps
        self.exclude_cls_token = exclude_cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:  # (B, C, H, W) — CNN feature map
            x = x.clamp(min=self.eps).pow(self.p)
            x = x.mean(dim=(2, 3))
            return x.pow(1.0 / self.p)
        if x.ndim == 3:  # (B, N, C) — token sequence
            if self.exclude_cls_token and x.shape[1] > 1:
                x = x[:, 1:, :]
            x = x.clamp(min=self.eps).pow(self.p)
            x = x.mean(dim=1)
            return x.pow(1.0 / self.p)
        raise ValueError(f"GeM expects 3D or 4D input, got shape {tuple(x.shape)}")
