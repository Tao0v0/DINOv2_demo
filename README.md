# 🦕 DINOv2 课堂演示项目

基于Meta DINOv2的自监督视觉表示学习课堂演示工具，帮助学生直观理解自监督学习的强大能力。

## 📋 项目概述

DINOv2是Meta开发的先进自监督视觉学习模型，无需人工标注就能学习强大的图像表示。本项目提供了一个交互式演示界面，展示DINOv2的核心能力。

### ✨ 主要功能

- 🔍 **图像相似性搜索**: 上传图片找到最相似的图像
- 📊 **特征空间可视化**: 观察DINOv2如何组织视觉信息
- 🎯 **交互式界面**: 基于Gradio的直观Web界面
- 📚 **教育友好**: 详细解释和可视化说明

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/dinov2-classroom-demo.git
cd dinov2-classroom-demo

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动演示

```bash
python demo.py
```

访问 `http://localhost:7860` 打开演示界面。

### 3. Jupyter版本演示

```bash
jupyter notebook notebooks/dinov2_demo.ipynb
```

## 📁 项目结构

```
dinov2-classroom-demo/
├── README.md                # 项目说明
├── requirements.txt         # 依赖包
├── demo.py                 # 主演示程序
├── utils.py                # 工具函数
├── sample_images/          # 示例图片 (自动生成)
├── notebooks/              # Jupyter演示
│   └── dinov2_demo.ipynb
└── docs/                   # 文档资料
    ├── presentation.md     # 课堂演示指南
    └── technical_details.md # 技术细节
```

## 🎯 使用指南

### 图像相似性搜索

1. 在"图像相似性搜索"标签页上传查询图片
2. 调整"显示前K个结果"滑块
3. 点击"查找相似图片"按钮
4. 观察结果和相似度分数

### 特征空间可视化

1. 切换到"特征空间可视化"标签页
2. 选择降维方法 (PCA或t-SNE)
3. 点击"生成可视化"按钮
4. 观察图片在特征空间中的分布

### 课堂演示建议

- **互动环节**: 让学生上传自己的图片测试
- **对比实验**: 展示相似和不相似图片的效果
- **原理解释**: 结合可视化说明自监督学习原理
- **讨论话题**: 为什么无监督学习也能学到语义信息？

## 🔧 技术细节

### DINOv2模型

- **架构**: Vision Transformer (ViT-Base)
- **参数量**: 约86M
- **特征维度**: 768维
- **预训练数据**: 大规模无标注图像

### 关键算法

1. **特征提取**: 使用CLS token作为全局图像表示
2. **相似度计算**: 余弦相似度度量
3. **降维可视化**: PCA和t-SNE方法
4. **自监督学习**: 教师-学生蒸馏框架

## 📊 演示效果展示

### 图像相似性搜索示例
![相似性搜索](docs/images/similarity_demo.png)

### 特征空间可视化示例
![特征可视化](docs/images/feature_viz_demo.png)

## 🎓 教育价值

### 学习目标
- 理解自监督学习的基本概念
- 体验视觉特征表示的效果
- 观察高维特征的空间结构
- 培养对AI模型的直观认识

### 课堂讨论点
1. 为什么DINOv2无需标签就能学到语义信息？
2. 自监督学习与监督学习的区别？
3. 视觉Transformer与CNN的不同？
4. 特征空间中的聚类现象说明了什么？

## 🛠️ 自定义扩展

### 添加自己的图片集

```python
# 在utils.py
