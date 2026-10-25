"""Augmentation giả lập khác biệt CAMERA/DOMAIN — khác nhóm augmentation chung
(crop, flip, erasing, augmix) đã có trong train.py::build_transforms. Ý tưởng
đã ghi trong strategy.md mục 5.3 từ 28/09/2026 nhưng CHƯA implement tới giờ
(30/09/2026): 3 dataset nguồn (Market1501, PA100k, PETA) khác nhau về chất
lượng camera, độ phân giải, độ nén — mô phỏng sự khác biệt đó ngay trong lúc
train giúp model bớt nhạy với đặc điểm camera cụ thể, thứ domain ẩn chắc chắn
sẽ khác. Đây là augmentation ở tầng PIXEL, khác các kỹ thuật domain-invariance
ở tầng FEATURE đã có (MixStyle/DANN/CORAL) — bổ sung cho nhau, không thay thế.

Cả 3 kỹ thuật dưới đây đều nhận và trả PIL.Image, chèn thẳng vào
transforms.Compose hiện có (đã test tương thích với torchvision==0.19.1 —
đúng bản pin trong requirements.txt — không phải đoán):
  - GaussianBlur: có sẵn trong torchvision.transforms (v1)
  - JPEG compression: torchvision.transforms.v2.JPEG (thêm ở 0.19, đã xác
    nhận có trong bản pin), mô phỏng ảnh bị nén nhiều/ít khác nhau giữa nguồn
  - RandomDownscaleUpscale: KHÔNG có sẵn trong torchvision, viết tay — thu nhỏ
    rồi phóng to lại để mô phỏng ảnh gốc độ phân giải thấp (camera cũ/xa) rồi
    bị người khác upscale, mất chi tiết high-frequency.

CHƯA đo hiệu quả trên bài toán thật — chỉ mới xác nhận chạy đúng cú pháp/kiểu
dữ liệu. BẮT BUỘC so bằng LOO như mọi augmentation khác, đặc biệt cẩn thận với
JPEG/blur vì có thể xoá mất chi tiết cần để nhận diện thuộc tính nhỏ (kính,
logo áo...) nếu quality/sigma quá mạnh.
"""
from __future__ import annotations

import random

from PIL import Image


class RandomDownscaleUpscale:
    """Thu nhỏ ảnh theo hệ số ngẫu nhiên trong [min_scale, 1.0) rồi phóng to lại
    về kích thước gốc — mô phỏng ảnh chụp từ camera độ phân giải thấp/xa rồi bị
    upscale, mất chi tiết tần số cao. Chỉ áp dụng với xác suất `p` (không phải
    ảnh nào cũng bị suy giảm — nếu áp dụng cho mọi ảnh, augmentation biến thành
    "chuẩn mới" thay vì mô phỏng một domain khác biệt).
    """

    def __init__(self, min_scale: float = 0.3, max_scale: float = 0.8, p: float = 0.3):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((small_w, small_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img


def build_domain_augment_ops(cfg: dict) -> list:
    """Trả về list các transform PIL->PIL để chèn vào train_ops trong
    train.py::build_transforms, theo cấu hình train.augment.domain_sim.
    Rỗng nếu domain_sim không được cấu hình hoặc tắt — không phải mọi config
    cũ (chưa có key này) đều bị crash, dùng .get() an toàn ở mọi cấp.
    """
    ds_cfg = cfg["train"]["augment"].get("domain_sim") or {}
    if not ds_cfg.get("enable"):
        return []

    from torchvision import transforms

    ops = []
    blur_p = ds_cfg.get("blur_p", 0.0)
    if blur_p > 0:
        ops.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 2.0))], p=blur_p,
        ))

    jpeg_p = ds_cfg.get("jpeg_p", 0.0)
    if jpeg_p > 0:
        from torchvision.transforms import v2
        ops.append(transforms.RandomApply(
            [v2.JPEG(quality=(20, 80))], p=jpeg_p,
        ))

    downscale_p = ds_cfg.get("downscale_p", 0.0)
    if downscale_p > 0:
        ops.append(RandomDownscaleUpscale(
            min_scale=ds_cfg.get("downscale_min_scale", 0.3),
            max_scale=ds_cfg.get("downscale_max_scale", 0.8),
            p=downscale_p,
        ))
    return ops
