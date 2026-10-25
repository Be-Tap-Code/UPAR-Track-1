"""CORAL (Correlation Alignment) — Sun & Saenko, "Deep CORAL: Correlation
Alignment for Deep Domain Adaptation", ECCV Workshops 2016.

Căn chỉnh thống kê bậc 2 (ma trận hiệp phương sai) giữa feature của các domain
nguồn khác nhau, KHÔNG cần discriminator/đối kháng như DANN — chỉ là một loss
phụ cộng thẳng vào loss chính, nên ổn định hơn DANN nhiều (không có nguy cơ
discriminator áp đảo feature extractor). Cùng tính chất backbone-agnostic với
DANN: hoạt động trên feature vector đã pool (B,C), dùng được cho cả CNN và ViT.

    C_a = cov(feat_a),  C_b = cov(feat_b)
    L_coral(a,b) = || C_a - C_b ||_F^2 / (4 * d^2)     (d = số chiều feature)

Trên benchmark DomainBed (Gulrajani & Lopez-Paz, ICLR 2021), CORAL thường xếp
ngang hoặc nhỉnh hơn DANN, và Fishr (ICML 2022) chỉ thua CORAL khoảng 0.1 điểm
— xem models/domain_adversarial.py cho bối cảnh đầy đủ về việc DomainBed cho
thấy ERM (không dùng thủ thuật DG nào) thường ngang bằng hoặc thắng CẢ HAI.

Với bài toán nhiều domain nguồn (không chỉ 2 như domain adaptation gốc), tính
CORAL loss cho MỌI CẶP domain có mặt trong batch rồi lấy trung bình.
"""
from __future__ import annotations

import torch


def coral_loss(feat_a: torch.Tensor, feat_b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """feat_a, feat_b: (N_a, C), (N_b, C) — feature đã pool của 2 nhóm domain
    khác nhau trong cùng batch. Cần N_a, N_b >= 2 để ước lượng covariance."""
    d = feat_a.size(1)
    feat_a = feat_a - feat_a.mean(dim=0, keepdim=True)
    feat_b = feat_b - feat_b.mean(dim=0, keepdim=True)
    cov_a = (feat_a.t() @ feat_a) / (feat_a.size(0) - 1 + eps)
    cov_b = (feat_b.t() @ feat_b) / (feat_b.size(0) - 1 + eps)
    return (cov_a - cov_b).pow(2).sum() / (4.0 * d * d)


def multi_domain_coral_loss(feat: torch.Tensor, domain_idx: torch.Tensor, n_domains: int,
                             min_samples: int = 2) -> torch.Tensor:
    """Trung bình CORAL loss qua mọi cặp domain có đủ mẫu (>= min_samples) trong
    batch hiện tại. Trả 0.0 (không lỗi) nếu batch không đủ đa dạng domain —
    tránh crash khi balanced sampler lỡ cho ra batch lệch."""
    groups = []
    for d in range(n_domains):
        mask = domain_idx == d
        if mask.sum().item() >= min_samples:
            groups.append(feat[mask])
    if len(groups) < 2:
        return feat.new_zeros(())
    losses = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            losses.append(coral_loss(groups[i], groups[j]))
    return torch.stack(losses).mean()
