import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载模型
# sam_checkpoint = "models/sam_vit_h_4b8939.pth"
sam_checkpoint = "models/sam_vit_b_01ec64.pth"
# model_type = "vit_h"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 加载图像
image_path = 'images/f_r_1965_.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 生成掩码
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# 构建彩色分割图和 Overlay
seg_map = np.zeros_like(image)
occupancy_mask = np.zeros(image.shape[:2], dtype=bool)
colors = np.random.randint(0, 255, (len(masks), 3), dtype=np.uint8)

# 按面积降序排列
sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

valid_segments = 0
for ann, color in zip(sorted_masks, colors):
    m = ann['segmentation']
    if isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    m = m.astype(bool)
    
    # 过滤重叠太多的 mask
    if (m & occupancy_mask).sum() / (m.sum() + 1e-6) > 0.15:
        continue

    m = m & ~occupancy_mask
    for c in range(3):
        seg_map[..., c][m] = color[c]
    occupancy_mask |= m
    valid_segments += 1

print(f"Generated {valid_segments} valid segments")

# 生成叠加图
overlay = (0.5 * image + 0.5 * seg_map).astype(np.uint8)

# 可视化：Original / Segmentation / Overlay
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(seg_map)
plt.title(f"Segmentation\n({valid_segments} segments)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()
