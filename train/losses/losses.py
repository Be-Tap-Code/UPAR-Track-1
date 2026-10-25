"""Loss cho multi-label PAR, có mask nhãn invalid và tuỳ chọn cân bằng lớp.

Tất cả nhận logits (chưa sigmoid) shape (B, 40), target và valid_mask cùng shape.
Vị trí valid_mask=0 không đóng góp vào loss (nhân 0 sau khi tính per-element).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(loss_elementwise: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    total = (loss_elementwise * valid_mask).sum()
    count = valid_mask.sum().clamp_min(1.0)
    return total / count


class WeightedBCELoss(nn.Module):
    """BCE có trọng số dương/âm theo tần suất từng thuộc tính (kiểu baseline UPAR gốc).
    pos_weight tính trước từ thống kê tập train, truyền vào lúc khởi tạo.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits, target, valid_mask):
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        return masked_mean(loss, valid_mask)


class FocalLoss(nn.Module):
    """Focal loss nhị phân độc lập từng thuộc tính (không phải focal contrastive
    kiểu AttriVision — công thức đó gắn với CLIP retrieval, không áp dụng trực
    tiếp ở đây vì ta dùng sigmoid classification, không phải cosine similarity).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target, valid_mask):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return masked_mean(loss, valid_mask)


class AsymmetricLoss(nn.Module):
    """ASL (Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification",
    ICCV 2021) — kỳ vọng mạnh cho multi-label mất cân bằng vì phạt khác nhau
    giữa lớp dương (gamma_pos nhỏ) và lớp âm (gamma_neg lớn), cộng "probability
    margin" (clip) để bỏ qua các mẫu âm dễ (model đã đúng, không cần học thêm).

    Công thức ĐÚNG theo paper (p = sigmoid(logit) = xác suất dự đoán là "có"):
        L+ = (1-p)^gamma_pos · log(p)                     [nhãn dương]
        p_m = max(p - clip, 0)                             [dịch margin trên p, KHÔNG phải 1-p]
        L- = p_m^gamma_neg · log(1-p_m)                    [nhãn âm]

    BUG ĐÃ SỬA (30/09/2026, phát hiện qua log train thật — mA/Instance_Rec sụp
    về giá trị của "dự đoán tất cả là dương" ngay từ epoch 0, loss về ~0 chỉ
    sau 1 epoch): bản cũ tính `p_neg = (1-p)+clip` rồi dùng `p_neg^gamma_neg`
    làm trọng số cho L- — tức margin bị áp lên `1-p` thay vì `p`, và trọng số
    dựa trên `p_neg` (xác suất ÂM) thay vì `p_m` (bám theo mức độ "sai lệch về
    phía dương" của một mẫu âm). Hậu quả: khi model dự đoán sai nặng nhất (nhãn
    âm nhưng p cao, tức p_neg nhỏ), trọng số `p_neg^gamma_neg` gần 0 — loss
    cho đúng các mẫu âm bị đoán sai nặng nhất GẦN NHƯ BIẾN MẤT, ngược hoàn toàn
    ý đồ focal loss (phải tập trung vào mẫu khó/sai, không phải mẫu dễ/đúng).
    Model học được cách "cứ đoán dương hết" vì phía âm không còn phản đối.
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, target, valid_mask):
        p = torch.sigmoid(logits)

        loss_pos = target * torch.log(p.clamp_min(self.eps)) * (1 - p).pow(self.gamma_pos)

        p_m = (p - self.clip).clamp(min=0.0) if self.clip > 0 else p
        loss_neg = (1 - target) * torch.log((1 - p_m).clamp_min(self.eps)) * p_m.pow(self.gamma_neg)

        loss = -(loss_pos + loss_neg)
        return masked_mean(loss, valid_mask)


def build_loss(cfg: dict, pos_weight: torch.Tensor | None = None) -> nn.Module:
    kind = cfg["loss"]["type"]
    if kind == "bce":
        return WeightedBCELoss(pos_weight=pos_weight, label_smoothing=cfg["loss"].get("label_smoothing", 0.0))
    if kind == "focal":
        return FocalLoss(alpha=cfg["loss"].get("alpha", 0.25), gamma=cfg["loss"].get("gamma", 2.0))
    if kind == "asl":
        return AsymmetricLoss(
            gamma_pos=cfg["loss"].get("gamma_pos", 0.0),
            gamma_neg=cfg["loss"].get("gamma_neg", 4.0),
            clip=cfg["loss"].get("clip", 0.05),
        )
    raise ValueError(f"Unknown loss type: {kind}")
