import torch

# 创建一个 8x8 的随机图像张量，对于彩色图像，通道数为 3
image_tensor = torch.randn(3, 4, 4)

print(image_tensor)
