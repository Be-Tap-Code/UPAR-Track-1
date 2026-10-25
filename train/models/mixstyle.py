"""MixStyle — trộn thống kê kênh (mean/std) giữa các ảnh khác nhau trong cùng
batch, để tổng hợp "domain giả" trong không gian feature, tăng đa dạng domain
nguồn mà không cần thêm dữ liệu.

Căn cứ: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021
(https://arxiv.org/abs/2104.02008), bản mở rộng IJCV 2023. Kỹ thuật
parameter-free, plug-in, không cần đổi loss/kiến trúc, báo cáo cải thiện rõ
trên các benchmark domain generalization ảnh. Rất khớp bài toán của ta: 3
dataset nguồn train (Market1501/PA100k/PETA), domain test hoàn toàn ẩn.

Công thức (áp dụng trên feature map (B,C,H,W), chỉ lúc training):
    mu, sigma = thống kê theo channel của từng ảnh trong batch (mean/std trên H,W)
    x_norm = (x - mu) / sigma
    hoán vị batch ngẫu nhiên -> mu2, sigma2 (thống kê của ảnh khác)
    lam ~ Beta(alpha, alpha)
    mu_mix = lam*mu + (1-lam)*mu2 ;  sigma_mix = lam*sigma + (1-lam)*sigma2
    output = x_norm * sigma_mix + mu_mix

CHỈ áp dụng cho feature map 4D (CNN). Paper gốc áp dụng ở các stage ĐẦU của
CNN (nơi thống kê kênh mang nhiều thông tin "style"/domain hơn là ở các layer
sâu mang thông tin ngữ nghĩa) — mặc định gắn vào stages.0/stages.1 của
ConvNeXt-B (tên module đã xác minh bằng timm thật, xem models/build.py).
Nếu đổi sang backbone ViT-style (feature dạng token, không phải feature map
không gian), MixStyle theo bản này KHÔNG áp dụng được — `attach_mixstyle_hooks`
sẽ bỏ qua và in cảnh báo thay vì lỗi ngầm.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Beta


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.beta = Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 15/10: hỗ trợ token ViT (B, N, C): thống kê theo kênh trên chiều token. Chỉ khi có gradient (tránh lọt vào validate của bản EMA,
        # vì hook của EMA (deepcopy) vẫn trỏ tới module MixStyle gốc).
        if (not self.training) or (not torch.is_grad_enabled()) or x.ndim not in (3, 4) or torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        if B < 2:
            return x
        dims = (2, 3) if x.ndim == 4 else (1,)
        mu = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig

        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        lam = self.beta.sample((B,) + (1,) * (x.ndim - 1)).to(x.device, x.dtype)

        mu_mix = mu * lam + mu2 * (1 - lam)
        sig_mix = sig * lam + sig2 * (1 - lam)
        return x_normed * sig_mix + mu_mix


def attach_mixstyle_hooks(backbone: nn.Module, layer_names: list[str], p: float, alpha: float) -> nn.ModuleList:
    """Gắn một MixStyle module (forward hook) sau output của mỗi module tên
    trong `layer_names` (dotted path, ví dụ "stages.0"). Trả về ModuleList chứa
    các MixStyle instance — PHẢI được gán làm submodule của model cha (ví dụ
    `self.mixstyle_modules = attach_mixstyle_hooks(...)`) để `.train()/.eval()`
    của model cha lan đúng xuống MixStyle (quyết định có áp dụng hay không).
    """
    named = dict(backbone.named_modules())
    mixstyle_modules = nn.ModuleList()
    for name in layer_names:
        if name not in named:
            print(f"[mixstyle] CẢNH BÁO: không thấy module '{name}' trong backbone, bỏ qua. "
                  f"Xem models/build.py để đổi layer_names đúng theo backbone đang dùng.")
            continue
        ms = MixStyle(p=p, alpha=alpha)
        mixstyle_modules.append(ms)

        def _hook(module, inputs, output, _ms=ms):
            if output.ndim not in (3, 4):
                # Backbone không cho feature map 4D ở layer này (ví dụ ViT token
                # sequence) — MixStyle bản này không áp dụng được, trả nguyên output.
                return output
            return _ms(output)

        named[name].register_forward_hook(_hook)
    return mixstyle_modules
