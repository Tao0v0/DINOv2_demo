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
