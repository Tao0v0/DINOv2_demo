# 🎬 Vision Classroom Demos: DINOv2 + SAM-2

本仓库提供一个 **All-in-One Notebook**，用于课堂演示两个视觉模型：

- **DINOv2**：自监督视觉表征 → 聚类可视化 + 最近邻检索
- **SAM-2**：交互式分割（提供脚手架，需下载权重）

## 🚀 快速开始

点击下方按钮即可直接在 Google Colab 打开并运行：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tao0v0/DINOv2_demo/blob/main/vision_demos_all_in_one.ipynb)

## 📂 文件说明
- `vision_demos_all_in_one.ipynb` ：集成 DINOv2 与 SAM-2 的课堂演示 Notebook  
  - **Part A — DINOv2**：图像特征提取、PCA 可视化、最近邻检索  
  - **Part B — SAM-2**：分割脚手架（需手动添加 checkpoint）

## 📝 课堂演示建议
1. **运行 DINOv2 部分**：展示 PCA 聚类效果、做一次 Top-k 检索。  
2. **运行 SAM-2 部分**：如果提前准备好权重，可展示点击/框选分割；否则介绍其作用。  
3. **备份**：若网络不稳，可提前录制运行效果视频备用。


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
