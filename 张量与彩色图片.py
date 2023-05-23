import torch
import matplotlib.pyplot as plt

# 假设我们有一张高度为 H，宽度为 W 的彩色图片
H = 256
W = 256

# 在 PyTorch 中，图片通常表示为 (C, H, W) 的格式，其中 C 为通道数
# 对于彩色图片，通道数 C 为 3（红色，绿色，蓝色）
image_tensor = torch.randn(3, H, W)

# 将张量的值调整到 [0, 1] 的范围内
image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

# PyTorch 使用 (C, H, W) 的格式，但 matplotlib 期望 (H, W, C) 的格式，
# 所以我们需要进行转置操作
image_tensor = image_tensor.permute(1, 2, 0)

# 使用 matplotlib 显示图像
plt.imshow(image_tensor)
plt.show()
