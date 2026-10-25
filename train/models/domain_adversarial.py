"""Domain-Adversarial Training (Ganin & Lempitsky, ICML 2015, "DANN") — kỹ
thuật generalization THAY THẾ cho MixStyle khi dùng backbone ViT/VLM.

Khác biệt cốt lõi với MixStyle (models/mixstyle.py): MixStyle tăng cường dữ
liệu bằng cách trộn thống kê "style" trong feature MAP 4D của CNN — không có
khái niệm tương đương rõ ràng cho token sequence của ViT. DANN thì khác: nó
hoạt động trên feature VECTOR ĐÃ POOL (B, C), thứ MỌI backbone đều tạo ra
(CNN lẫn ViT/CLIP/SigLIP), nên lấp đúng khoảng trống MixStyle để lại.

Ý tưởng: gắn thêm một domain classifier nhỏ cố đoán ảnh đến từ dataset nguồn
nào (Market1501/PA100k/PETA), nhưng gradient của nó bị ĐẢO DẤU trước khi
truyền ngược vào backbone (Gradient Reversal Layer — GRL). Kết quả: domain
classifier được train bình thường (cực tiểu hoá loss của chính nó), còn
backbone bị ép đi ngược hướng đó — học feature mà domain classifier càng lúc
càng KHÓ đoán được domain nguồn. Feature "domain-invariant" hơn thường tổng
quát tốt hơn sang domain ẩn — đúng vấn đề chính của bài toán này.

CẢNH BÁO QUAN TRỌNG (research 28/09/2026, sau khi bị hỏi "DANN có lỗi thời
không"): DomainBed (Gulrajani & Lopez-Paz, ICLR 2021, arXiv:2007.01434) — bộ
benchmark chuẩn cho domain generalization — cho thấy khi tune cẩn thận và
đánh giá công bằng, **ERM (không dùng thủ thuật DG nào cả) thường ngang bằng
hoặc THẮNG DANN** và hầu hết thuật toán DG chuyên biệt khác; không thuật toán
nào trong benchmark đó vượt ERM quá 1 điểm trung bình. DANN chuẩn còn có vấn
đề huấn luyện KHÔNG ỔN ĐỊNH đã biết: discriminator dễ áp đảo feature extractor.
=> Đây KHÔNG PHẢI kỹ thuật "cứ bật là lợi" — mặc định TẮT trong configs/base.yaml
(đã sửa lại sau phát hiện này), chỉ là một ỨNG VIÊN thử nghiệm qua LOO như SAM,
KHÔNG hơn. Xem models/coral.py cho một lựa chọn thay thế ổn định hơn (không
đối kháng) cùng benchmark, thường xếp ngang hoặc trên DANN.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def gradient_reversal(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReversalFunction.apply(x, lambd)


class DomainClassifier(nn.Module):
    """MLP nhỏ đoán domain nguồn từ feature đã pool. Cố tình nhỏ — mục tiêu
    không phải đoán domain thật giỏi, mà là ép backbone không cho nó đoán được.
    """

    def __init__(self, feat_dim: int, n_domains: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, feat: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(gradient_reversal(feat, lambd))


def dann_lambda_schedule(progress: float, gamma: float = 10.0, lambda_max: float = 1.0) -> float:
    """Lịch tăng dần lambda theo tiến độ train (progress trong [0,1]), đúng
    công thức trong paper gốc: đầu train gần như không ép domain-invariance
    (tránh phá feature pretrain còn chưa kịp thích nghi), cuối train ép mạnh
    hơn dần. lambda_max nhân thêm để dễ chỉnh cường độ tổng thể qua config.
    """
    progress = max(0.0, min(1.0, progress))
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)
