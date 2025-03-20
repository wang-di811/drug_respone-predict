import os
# 设置为您想使用的CPU核心数量，例如4
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_drug_graph(file_path):
    """
    加载药物分子图
    
    参数:
        file_path (str): PT文件路径
        
    返回:
        Data: PyTorch Geometric数据对象
    """
    try:
        data = torch.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def get_drug_name_from_path(file_path):
    """
    从文件路径中提取药物名称
    
    参数:
        file_path (str): PT文件路径
        
    返回:
        str: 药物名称
    """
    # 获取文件名，去掉扩展名
    return os.path.splitext(os.path.basename(file_path))[0]

def visualize_drug_features(features_df, output_file, method='pca'):
    """
    可视化药物特征
    
    参数:
        features_df (DataFrame): 包含药物特征的DataFrame
        output_file (str): 输出文件路径
        method (str): 降维方法 ('pca' 或 'tsne')
    """
    # 提取药物名称和特征
    drug_names = features_df['drug_name'].values
    features = features_df.drop('drug_name', axis=1).values
    
    # 降维
    if features.shape[0] < 3:
        print("警告: 药物数量太少，无法进行可靠的降维可视化")
        return
        
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(features)
        method_name = 'PCA'
    else:
        reducer = TSNE(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
        method_name = 't-SNE'
    
    # 创建图表
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=np.arange(len(drug_names)), 
                 cmap='viridis', alpha=0.8, s=100)
    
    # 添加药物名称标签
    for i, name in enumerate(drug_names):
        plt.annotate(name, (reduced_features[i, 0], reduced_features[i, 1]), 
                    fontsize=9, alpha=0.7)
    
    plt.title(f'drug feature {method_name} visualization', fontsize=14)
    plt.colorbar(scatter, label='drug index')
    plt.xlabel(f'{method_name} component 1')
    plt.ylabel(f'{method_name} component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"特征可视化已保存至: {output_file}")
    plt.close()