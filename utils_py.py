import torch
import numpy as np
from PIL import Image
import os
import requests
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def compute_similarity(features1, features2):
    """计算两个特征向量的余弦相似度"""
    if features1.ndim == 1:
        features1 = features1.reshape(1, -1)
    if features2.ndim == 1:
        features2 = features2.reshape(1, -1)
    
    similarity = cosine_similarity(features1, features2)[0, 0]
    return float(similarity)

def download_sample_image(url, filename):
    """下载示例图片"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    return False

def create_sample_images():
    """创建示例图片集合"""
    # 示例图片URL列表（使用一些常见的测试图片）
    sample_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
    ]
    
    # 创建示例图片目录
    os.makedirs("sample_images", exist_ok=True)
    
    # 如果无法下载，创建一些简单的测试图片
    sample_images = []
    
    # 创建几个简单的测试图片
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
    ]
    
    patterns = [
        "solid",      # 纯色
        "gradient",   # 渐变
        "checkerboard", # 棋盘
        "circles",    # 圆形
        "stripes",    # 条纹
        "dots",       # 点状
    ]
    
    for i, (color, pattern) in enumerate(zip(colors, patterns)):
        img = create_pattern_image(color, pattern, size=(224, 224))
        filename = f"sample_images/sample_{i+1}_{pattern}.png"
        img.save(filename)
        sample_images.append(filename)
    
    return sample_images

def create_pattern_image(color, pattern, size=(224, 224)):
    """创建带有特定模式的图片"""
    img = Image.new('RGB', size, color)
    pixels = img.load()
    
    if pattern == "gradient":
        for y in range(size[1]):
            factor = y / size[1]
            new_color = tuple(int(c * factor) for c in color)
            for x in range(size[0]):
                pixels[x, y] = new_color
    
    elif pattern == "checkerboard":
        square_size = 20
        for y in range(size[1]):
            for x in range(size[0]):
                if ((x // square_size) + (y // square_size)) % 2:
                    pixels[x, y] = (255 - color[0], 255 - color[1], 255 - color[2])
    
    elif pattern == "circles":
        center_x, center_y = size[0] // 2, size[1] // 2
        for y in range(size[1]):
            for x in range(size[0]):
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if int(distance) % 30 < 15:
                    pixels[x, y] = (255 - color[0], 255 - color[1], 255 - color[2])
    
    elif pattern == "stripes":
        stripe_width = 15
        for y in range(size[1]):
            for x in range(size[0]):
                if (x // stripe_width) % 2:
                    pixels[x, y] = (255 - color[0], 255 - color[1], 255 - color[2])
    
    elif pattern == "dots":
        for y in range(0, size[1], 30):
            for x in range(0, size[0], 30):
                for dy in range(-5, 6):
                    for dx in range(-5, 6):
                        if 0 <= x + dx < size[0] and 0 <= y + dy < size[1]:
                            if dx*dx + dy*dy <= 25:  # 圆形点
                                pixels[x + dx, y + dy] = (255 - color[0], 255 - color[1], 255 - color[2])
    
    return img

def load_sample_images(processor, model, sample_dir="sample_images"):
    """加载示例图片并提取特征"""
    # 如果示例图片目录不存在，创建它
    if not os.path.exists(sample_dir):
        print("📁 创建示例图片...")
        sample_paths = create_sample_images()
    else:
        sample_paths = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not sample_paths:
            print("📁 创建示例图片...")
            sample_paths = create_sample_images()
    
    images = []
    features = []
    
    print(f"🖼️  加载 {len(sample_paths)} 张示例图片...")
    
    for img_path in sample_paths:
        try:
            # 加载图片
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            
            # 提取特征
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用CLS token作为全局特征
                feature = outputs.last_hidden_state[:, 0, :].numpy()
                features.append(feature)
        
        except Exception as e:
            print(f"❌ 加载图片失败 {img_path}: {e}")
            continue
    
    print(f"✅ 成功加载 {len(images)} 张图片")
    return images, features

def visualize_features(features, labels=None, method='PCA', title="特征可视化"):
    """可视化高维特征"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    features_array = np.vstack(features) if isinstance(features, list) else features
    
    # 降维
    if method == 'PCA':
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(features_array)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_array)-1))
        reduced_features = reducer.fit_transform(features_array)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    if labels is None:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
        for i, (x, y) in enumerate(reduced_features):
            plt.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    else:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = np.array(labels) == label
            plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                       c=[color], label=label, alpha=0.7)
    
    plt.title(title)
    plt.xlabel(f'{method} 1')
    plt.ylabel(f'{method} 2')
    if labels is not None:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def batch_extract_features(images, processor, model, batch_size=8):
    """批量提取图片特征"""
    features = []
    model.eval()
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        
        # 预处理批量图片
        inputs = processor(images=batch_images, return_tensors="pt")
        
        # 提取特征
        with torch.no_grad():
            outputs = model(**inputs)
            batch_features = outputs.last_hidden_state[:, 0, :].numpy()
            features.extend(batch_features)
    
    return np.array(features)

def save_results(results, filename="demo_results.txt"):
    """保存演示结果"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("DINOv2 课堂演示结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {np.datetime64('now')}\n\n")
        
        for key, value in results.items():
            f.write(f"{key}:\n{value}\n\n")
    
    print(f"✅ 结果已保存到 {filename}")

def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        'torch', 'transformers', 'gradio', 'scikit-learn', 
        'matplotlib', 'numpy', 'PIL', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有依赖包都已安装")
        return True

if __name__ == "__main__":
    # 测试工具函数
    print("🧪 测试工具函数...")
    
    # 检查依赖
    check_dependencies()
    
    # 创建测试图片
    test_images = create_sample_images()
    print(f"📸 创建了 {len(test_images)} 张测试图片")
    
    print("✅ 工具函数测试完成!")