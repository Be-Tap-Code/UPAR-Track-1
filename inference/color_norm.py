"""Chuẩn hoá màu theo TỪNG ẢNH (gray-world, có cường độ s): nhân mỗi kênh với (trung bình xám / trung bình kênh)^s, kẹp hệ số [gmin, gmax].
Mục đích: giảm lệch màu do camera/ánh sáng giữa các dataset (màu là 56% mất mát mA). Chỉ dùng thống kê của CHÍNH ảnh đó (không dùng thống kê test).
File này nằm ở gốc zip nộp (không import gì ngoài numpy/PIL) và được train.py import cùng một bản => train/infer khớp tuyệt đối.
"""
import numpy as np
from PIL import Image


class GrayWorld:
    def __init__(self, strength: float = 1.0, gmin: float = 0.5, gmax: float = 2.0):
        self.s, self.gmin, self.gmax = float(strength), float(gmin), float(gmax)

    def __call__(self, img: Image.Image) -> Image.Image:
        a = np.asarray(img.convert("RGB"), dtype=np.float32)
        m = a.reshape(-1, 3).mean(0) + 1e-6
        g = np.clip((m.mean() / m) ** self.s, self.gmin, self.gmax)
        return Image.fromarray(np.clip(a * g, 0, 255).astype(np.uint8))

    def __repr__(self):
        return f"GrayWorld(strength={self.s}, gmin={self.gmin}, gmax={self.gmax})"


def build_color_norm(spec):
    """spec: None | {"kind": "grayworld", "strength": float} -> callable hoặc None."""
    if not spec:
        return None
    if spec.get("kind", "grayworld") != "grayworld":
        raise ValueError(f"color_norm không hỗ trợ: {spec}")
    return GrayWorld(float(spec.get("strength", 1.0)), float(spec.get("gmin", 0.5)), float(spec.get("gmax", 2.0)))
