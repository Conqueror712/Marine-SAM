import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载模型
sam_checkpoint = "models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 可视化函数
def show_anns(anns):
    if len(anns) == 0:
        print("No annotations found!")
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        if isinstance(m, torch.Tensor):  # 如果是张量，转为 numpy 数组
            m = m.cpu().numpy()
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.rand(3)  # 生成随机颜色
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))

# 加载图像
image_path = 'images/set_o48.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 生成掩码
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# 可视化结果
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
