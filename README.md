# DINOv2 Demo: Feature Extraction & Attention Rollout

在本演示中，我们将基于 **DINOv2 (Vision Transformer)**：
1. 加载预训练模型并对图像提取全局特征向量；
2. 通过 **Attention Rollout** 可视化注意力图，观察不同层如何聚焦到物体部件与上下文区域。

> 课程场景：During the lecture we will explore two hands-on demos using Colab notebooks.  
> In the first demo, we load a pretrained Vision Transformer (e.g., DINOv2) and extract features from images captured by a robot.  
> By visualizing attention maps with attention rollout, we observe how different heads focus on object parts and contextual regions.

---

## ▶️ 一键在 Colab 运行

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tao0v0/DINOv2_demo/blob/main/dinov2_attn_rollout_demo.ipynb)

---

## 主要内容
- 使用 Hugging Face `facebook/dinov2-small` 权重（CPU/GPU均可运行）
- 支持：
  - 本地上传图片
  - 提供若干示例图片 URL（可直接跑通）
- 输出：
  - 每张图的全局特征向量（CLS 或平均池化）
  - Attention Rollout 热力图叠加原图（可调层范围、head 融合方式）

## 依赖
- `torch`, `transformers`, `torchvision`, `matplotlib`, `numpy`, `Pillow`, `opencv-python`（Colab 脚本会自动安装）
---

## 参考
- DINOv2 官方仓库（torch.hub/Colab 示例等）  
- Attention Rollout（Abnar & Zuidema, 2020）方法

---

© 2025 Tao0v0. For teaching/demo purpose only.
