## DINOv2 Dense Feature Colorization (PCA / KMeans)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tao030/DINOv2_demo/blob/main/dinov2_feature_visualization_demo.ipynb)

In this demo we colorize **per-patch DINOv2 features**:
- **PCA→RGB** for continuous pseudo-color maps
- **KMeans** for discrete part-like segments

Works with images and short videos; PCA/KMeans are fit on the first frame to keep colors consistent across time.
