import torch
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from transformers import AutoImageProcessor, AutoModel
from utils import compute_similarity, visualize_features, load_sample_images
import warnings
warnings.filterwarnings("ignore")

class DINOv2Demo:
    def __init__(self):
        print("🚀 加载DINOv2模型中...")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.model.eval()
        
        # 加载示例图片
        self.sample_images, self.sample_features = load_sample_images(self.processor, self.model)
        print("✅ 模型加载完成！")
    
    def extract_features(self, image):
        """提取图像特征"""
        if image is None:
            return None
        
        # 预处理图像
        inputs = self.processor(images=image, return_tensors="pt")
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用CLS token作为全局特征
            features = outputs.last_hidden_state[:, 0, :].numpy()
        
        return features
    
    def find_similar_images(self, query_image, top_k=5):
        """找到最相似的图像"""
        if query_image is None:
            return None, "请上传查询图像"
        
        # 提取查询图像特征
        query_features = self.extract_features(query_image)
        if query_features is None:
            return None, "特征提取失败"
        
        # 计算相似度
        similarities = []
        for i, sample_feature in enumerate(self.sample_features):
            similarity = compute_similarity(query_features, sample_feature)
            similarities.append((similarity, i))
        
        # 排序并获取top-k
        similarities.sort(reverse=True)
        
        # 创建结果图像
        fig, axes = plt.subplots(1, min(top_k, len(similarities)), figsize=(15, 3))
        if top_k == 1:
            axes = [axes]
        
        result_text = f"找到 {min(top_k, len(similarities))} 张最相似的图片:\n\n"
        
        for idx, (similarity, img_idx) in enumerate(similarities[:top_k]):
            if idx < len(axes):
                axes[idx].imshow(self.sample_images[img_idx])
                axes[idx].set_title(f'相似度: {similarity:.3f}')
                axes[idx].axis('off')
                result_text += f"{idx+1}. 图片 {img_idx+1}: 相似度 {similarity:.3f}\n"
        
        plt.tight_layout()
        plt.savefig('similarity_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return 'similarity_results.png', result_text
    
    def visualize_feature_space(self, method="PCA"):
        """可视化特征空间"""
        if len(self.sample_features) < 2:
            return None, "需要至少2张图片才能可视化"
        
        features_array = np.vstack(self.sample_features)
        
        if method == "PCA":
            reducer = PCA(n_components=2)
            reduced_features = reducer.fit_transform(features_array)
            title = "DINOv2特征的PCA可视化"
            explained_var = reducer.explained_variance_ratio_
            subtitle = f"解释方差比: PC1={explained_var[0]:.2f}, PC2={explained_var[1]:.2f}"
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_array)-1))
            reduced_features = reducer.fit_transform(features_array)
            title = "DINOv2特征的t-SNE可视化"
            subtitle = "展示高维特征的非线性结构"
        
        # 创建可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                            c=range(len(reduced_features)), cmap='tab10', s=100, alpha=0.7)
        
        # 添加图片编号标签
        for i, (x, y) in enumerate(reduced_features):
            plt.annotate(f'图{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('第一主成分' if method == "PCA" else '维度 1', fontsize=12)
        plt.ylabel('第二主成分' if method == "PCA" else '维度 2', fontsize=12)
        plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=10, style='italic')
        
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='图片索引')
        plt.tight_layout()
        plt.savefig('feature_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        explanation = f"""
        📊 特征空间可视化说明:
        
        • 每个点代表一张图片在DINOv2特征空间中的位置
        • 距离越近的点表示图片在语义上越相似
        • {method}将768维特征降到2维便于观察
        • 颜色和编号帮助识别不同图片
        
        🔍 观察要点:
        • 相似内容的图片是否聚集在一起?
        • 不同类别的图片是否分布在不同区域?
        • 这反映了DINOv2学到了什么样的表示?
        """
        
        return 'feature_visualization.png', explanation

def create_demo():
    demo_instance = DINOv2Demo()
    
    with gr.Blocks(title="DINOv2 课堂演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🦕 DINOv2 自监督视觉表示学习演示
        
        **DINOv2** 是Meta开发的自监督视觉Transformer，无需标注数据就能学习强大的图像表示。
        
        ## 🎯 核心能力展示
        """)
        
        with gr.Tabs():
            # Tab 1: 图像相似性搜索
            with gr.TabItem("🔍 图像相似性搜索"):
                gr.Markdown("### 上传一张图片，找到最相似的图片")
                
                with gr.Row():
                    with gr.Column():
                        query_input = gr.Image(type="pil", label="上传查询图片")
                        similarity_btn = gr.Button("🔍 查找相似图片", variant="primary")
                        top_k_slider = gr.Slider(1, 8, 5, label="显示前K个结果")
                    
                    with gr.Column():
                        similarity_output = gr.Image(label="相似图片结果")
                        similarity_text = gr.Textbox(label="详细结果", lines=6)
                
                similarity_btn.click(
                    demo_instance.find_similar_images,
                    inputs=[query_input, top_k_slider],
                    outputs=[similarity_output, similarity_text]
                )
            
            # Tab 2: 特征空间可视化
            with gr.TabItem("📊 特征空间可视化"):
                gr.Markdown("### 查看DINOv2如何在高维空间中组织图像")
                
                with gr.Row():
                    with gr.Column():
                        viz_method = gr.Radio(["PCA", "t-SNE"], value="PCA", label="降维方法")
                        viz_btn = gr.Button("📊 生成可视化", variant="primary")
                        
                        gr.Markdown("""
                        **降维方法说明:**
                        - **PCA**: 线性降维，保持全局结构
                        - **t-SNE**: 非线性降维，保持局部结构
                        """)
                    
                    with gr.Column():
                        viz_output = gr.Image(label="特征空间可视化")
                        viz_text = gr.Textbox(label="解释说明", lines=12)
                
                viz_btn.click(
                    demo_instance.visualize_feature_space,
                    inputs=[viz_method],
                    outputs=[viz_output, viz_text]
                )
            
            # Tab 3: 关于DINOv2
            with gr.TabItem("📚 关于DINOv2"):
                gr.Markdown("""
                ## 🤖 什么是DINOv2?
                
                **DINOv2 (Self-Supervised Learning of Visual Features via Embedding Distillation)** 是一个强大的视觉自监督学习模型。
                
                ### ✨ 主要特点:
                - **无需标注**: 不需要人工标注的数据集
                - **强大表示**: 学习到丰富的视觉语义表示
                - **广泛适用**: 可用于分类、检测、分割等多种任务
                - **开箱即用**: 预训练模型可直接用于特征提取
                
                ### 🔬 技术原理:
                1. **自蒸馏学习**: 学生网络学习教师网络的表示
                2. **多尺度训练**: 不同分辨率的图像增强泛化能力
                3. **Vision Transformer**: 基于Transformer架构的视觉模型
                
                ### 🎯 应用场景:
                - 图像检索和相似性搜索
                - 零样本图像分类
                - 特征提取和表示学习
                - 下游任务的预训练模型
                
                ### 📖 论文信息:
                - **标题**: "DINOv2: Learning Robust Visual Features without Supervision"
                - **作者**: Oquab et al. (Meta AI)
                - **发表**: 2023
                """)
        
        gr.Markdown("""
        ---
        💡 **使用提示**: 
        - 尝试上传不同类型的图片看看相似性效果
        - 观察特征可视化中相似图片的聚集情况
        - 思考：为什么DINOv2能在没有标签的情况下学到语义信息？
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口
        share=False,            # 不生成public链接
        debug=True              # 调试模式
    )