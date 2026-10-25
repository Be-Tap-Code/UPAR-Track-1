"""Danh sách 40 thuộc tính, đúng thứ tự theo data.md của challenge.

QUAN TRỌNG: thứ tự này phải khớp 100% với "attribute_names" mà ingestion program
truyền vào predict_image/predict_batch, và với thứ tự cột trong file annotation
train/val. Nếu annotation thật của bộ 2027 có thứ tự khác, sửa lại ở đây — mọi
chỗ khác trong code/ đều import từ đây, không hardcode index ở nơi khác.
"""

ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short",
    "UpperBody-Color-Black", "UpperBody-Color-Blue", "UpperBody-Color-Brown",
    "UpperBody-Color-Green", "UpperBody-Color-Grey", "UpperBody-Color-Orange",
    "UpperBody-Color-Pink", "UpperBody-Color-Purple", "UpperBody-Color-Red",
    "UpperBody-Color-White", "UpperBody-Color-Yellow", "UpperBody-Color-Other",
    "LowerBody-Length-Short",
    "LowerBody-Color-Black", "LowerBody-Color-Blue", "LowerBody-Color-Brown",
    "LowerBody-Color-Green", "LowerBody-Color-Grey", "LowerBody-Color-Orange",
    "LowerBody-Color-Pink", "LowerBody-Color-Purple", "LowerBody-Color-Red",
    "LowerBody-Color-White", "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag",
    "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat",
]

N_ATTR = len(ATTRIBUTES)
assert N_ATTR == 40, f"Expected 40 attributes, got {N_ATTR}"

ATTR_TO_IDX = {name: i for i, name in enumerate(ATTRIBUTES)}

# Nhóm mutually-exclusive (một-hot trong nhóm), dùng để:
#  - kiểm tra sanity nhãn (đúng 1 giá trị =1 mỗi nhóm, trừ khi cả nhóm invalid)
#  - cân nhắc dùng softmax-per-group thay vì sigmoid độc lập (thử nghiệm, đo bằng LOO)
ATTR_GROUPS = {
    "age": ["Age-Young", "Age-Adult", "Age-Old"],
    "hair_length": ["Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald"],
    "upperbody_color": [a for a in ATTRIBUTES if a.startswith("UpperBody-Color-")],
    "lowerbody_color": [a for a in ATTRIBUTES if a.startswith("LowerBody-Color-")],
    "lowerbody_type": ["LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress"],
}
