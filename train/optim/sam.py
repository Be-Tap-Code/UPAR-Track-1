"""Sharpness-Aware Minimization (SAM) — Foret et al. 2020, tối ưu tới vùng cực
tiểu "phẳng" (nhiễu nhỏ quanh tham số vẫn giữ loss thấp), có cơ sở lý thuyết và
thực nghiệm cho generalization tốt hơn khi có domain shift.

CẢNH BÁO ĐÃ TÌM THẤY TRONG LITERATURE (đáng cân nhắc trước khi bật mặc định):
nghiên cứu DGSAM (2025, https://arxiv.org/abs/2503.23430) chỉ ra SAM chuẩn áp
vào domain generalization có thể hội tụ tới "flat minima giả" — phẳng xét trên
tổng loss nhưng vẫn gồ ghề xét theo từng domain nguồn riêng lẻ. SAM chuẩn
(bản này) vẫn là điểm khởi đầu hợp lý và rẻ để cài đặt, nhưng đừng coi là
"chắc thắng" — PHẢI đo bằng LOO so với AdamW thường trước khi giữ lại.

CHI PHÍ: SAM cần 2 lần forward+backward mỗi bước (một để tính hướng nhiễu, một
để tính gradient thật tại điểm đã nhiễu) — train.py gọi `first_step`/
`second_step` tương ứng. Vì vậy ~gấp đôi thời gian/epoch so với AdamW thường.
Mặc định TẮT trong configs/base.yaml, chỉ bật khi đã có ngân sách thời gian dư
sau khi baseline AdamW đã chạy xong.

Cài đặt theo đúng công thức gốc (tham khảo cấu trúc phổ biến của
davda54/sam, không copy nguyên văn, viết lại để khớp interface layer_wise
param_groups của models/build.py).
"""
from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, adaptive: bool = False, **base_kwargs):
        if rho < 0:
            raise ValueError(f"rho phải >= 0, nhận {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **base_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Bước 1: nhiễu tham số theo hướng gradient hiện tại (đi lên bề mặt loss)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Bước 2: phục hồi tham số gốc, áp dụng update thật (gradient tính TẠI
        điểm đã nhiễu, tức đã backward lần 2 trước khi gọi hàm này)."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)

    def step(self, closure=None):
        raise RuntimeError(
            "SAM.step() không dùng trực tiếp — gọi first_step()/backward()/second_step() "
            "thủ công trong training loop. Xem train.py::train_one_fold."
        )
