import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from src.models import GCNFeatureExtractor
from src.utils import load_drug_graph, get_drug_name_from_path
import config

class PoolingMethodComparator:
    def __init__(self, graphs_dir, output_dir, model_config):
        """
        初始化池化方法比较器
        
        参数:
            graphs_dir (str): 药物分子图目录
            output_dir (str): 输出目录
            model_config (dict): 模型配置
        """
        self.graphs_dir = graphs_dir
        self.output_dir = output_dir
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取图文件列表
        self.graph_files = [f for f in os.listdir(graphs_dir) if f.endswith('.pt')]
        if not self.graph_files:
            raise ValueError(f"在 {graphs_dir} 中找不到任何.pt文件")
            
        # 加载第一个图以获取节点特征维度
        first_graph = load_drug_graph(os.path.join(graphs_dir, self.graph_files[0]))
        if first_graph is None:
            raise ValueError(f"无法加载第一个图文件")
        self.input_dim = first_graph.x.shape[1]
        
        # 池化方法列表
        self.pooling_methods = ['mean', 'add', 'max']
        
    def extract_features_with_pooling(self, pooling_method):
        """
        使用指定的池化方法提取特征
        
        参数:
            pooling_method (str): 池化方法
            
        返回:
            DataFrame: 包含药物名称和特征的DataFrame
        """
        # 更新模型配置使用指定的池化方法
        model_config = self.model_config.copy()
        
        # 创建模型
        model = GCNFeatureExtractor(
            in_channels=self.input_dim,
            hidden_channels=model_config.get('hidden_channels', 64),
            out_channels=model_config.get('out_channels', 32),
            num_layers=model_config.get('num_layers', 3),
            pool_method=pooling_method
        ).to(self.device)
        model.eval()
        
        # 提取特征
        results = {'drug_name': [], 'features': []}
        
        with torch.no_grad():
            for graph_file in tqdm(self.graph_files, desc=f"提取特征 ({pooling_method} 池化)"):
                file_path = os.path.join(self.graphs_dir, graph_file)
                drug_name = get_drug_name_from_path(file_path)
                
                # 加载分子图
                graph_data = load_drug_graph(file_path)
                if graph_data is None:
                    continue
                
                # 将数据移到设备
                graph_data = graph_data.to(self.device)
                
                # 确保有batch属性
                if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                    graph_data.batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long, device=self.device)
                
                # 提取特征
                features = model(graph_data.x, graph_data.edge_index, graph_data.batch)
                
                # 保存结果
                results['drug_name'].append(drug_name)
                results['features'].append(features.cpu().numpy().flatten())
        
        # 创建DataFrame
        features_array = np.vstack(results['features'])
        feature_columns = [f'feature_{i}' for i in range(features_array.shape[1])]
        
        feature_df = pd.DataFrame(features_array, columns=feature_columns)
        result_df = pd.DataFrame({'drug_name': results['drug_name']})
        result_df = pd.concat([result_df, feature_df], axis=1)
        
        return result_df
    
    def evaluate_features(self, features_df, method_name):
        """
        评估特征质量
        
        参数:
            features_df (DataFrame): 特征DataFrame
            method_name (str): 池化方法名称
            
        返回:
            dict: 评估指标
        """
        # 提取特征矩阵
        features = features_df.drop('drug_name', axis=1).values
        
        # 计算特征之间的方差
        feature_variance = np.var(features, axis=0).mean()
        
        # 计算特征之间的距离矩阵
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(features, 'euclidean'))
        
        # 平均距离
        avg_distance = np.mean(distances)
        
        # 尝试进行聚类评估（使用轮廓系数）
        silhouette_scores = []
        min_clusters = min(5, len(features) - 1) if len(features) > 5 else 2
        max_clusters = min(10, len(features) - 1) if len(features) > 10 else min_clusters + 1
        
        for n_clusters in range(max(2, min_clusters), max(3, max_clusters)):
            if n_clusters >= len(features):
                continue
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                if len(np.unique(labels)) > 1:  # 确保不止一个聚类
                    score = silhouette_score(features, labels)
                    silhouette_scores.append(score)
            except:
                continue
        
        # 平均轮廓系数
        avg_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0
        
        # PCA解释方差
        pca = PCA(n_components=min(5, features.shape[1], features.shape[0]))
        pca.fit(features)
        explained_variance = sum(pca.explained_variance_ratio_[:2])  # 前两个成分的解释方差比例
        
        return {
            'method': method_name,
            'feature_variance': feature_variance,
            'avg_distance': avg_distance,
            'avg_silhouette': avg_silhouette,
            'explained_variance_2d': explained_variance
        }
    
    def visualize_features(self, features_df, method_name, dim_reduction='tsne'):
        """
        可视化特征
        
        参数:
            features_df (DataFrame): 特征DataFrame
            method_name (str): 池化方法名称
            dim_reduction (str): 降维方法 ('pca' 或 'tsne')
        """
        # 提取药物名称和特征
        drug_names = features_df['drug_name'].values
        features = features_df.drop('drug_name', axis=1).values
        
        # 降维
        if dim_reduction == 'pca':
            reducer = PCA(n_components=2)
            reduced_features = reducer.fit_transform(features)
            method_label = 'PCA'
        else:
            reducer = TSNE(n_components=2, random_state=42)
            reduced_features = reducer.fit_transform(features)
            method_label = 't-SNE'
        
        # 创建图表
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=np.arange(len(drug_names)), 
                      cmap='viridis', alpha=0.8, s=80)
        
        # 添加药物名称标签
        for i, name in enumerate(drug_names):
            plt.annotate(name, (reduced_features[i, 0], reduced_features[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.title(f'{method_name.capitalize()} Pooling - {method_label} Visualization', fontsize=14)
        plt.xlabel(f'{method_label} Component 1')
        plt.ylabel(f'{method_label} Component 2')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.colorbar(scatter, label='Drug index')
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, f'{method_name}_{dim_reduction}_visualization.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def compare_all_methods(self):
        """
        比较所有池化方法
        
        返回:
            DataFrame: 比较结果
        """
        all_results = []
        feature_dfs = {}
        
        # 对每种池化方法提取特征并评估
        for method in self.pooling_methods:
            print(f"\n===== 评估 {method.upper()} 池化 =====")
            # 提取特征
            features_df = self.extract_features_with_pooling(method)
            feature_dfs[method] = features_df
            
            # 保存特征到CSV
            output_csv = os.path.join(self.output_dir, f'{method}_pooling_features2.csv')
            features_df.to_csv(output_csv, index=False)
            print(f"特征已保存到: {output_csv}")
            
            # 评估特征
            metrics = self.evaluate_features(features_df, method)
            all_results.append(metrics)
            
            # 可视化特征 (PCA)
            pca_file = self.visualize_features(features_df, method, 'pca')
            print(f"PCA可视化已保存到: {pca_file}")
            
            # 可视化特征 (t-SNE)
            tsne_file = self.visualize_features(features_df, method, 'tsne')
            print(f"t-SNE可视化已保存到: {tsne_file}")
        
        # 创建比较结果DataFrame
        results_df = pd.DataFrame(all_results)
        
        # 保存比较结果
        output_comparison = os.path.join(self.output_dir, 'pooling_methods_comparison2.csv')
        results_df.to_csv(output_comparison, index=False)
        print(f"\n比较结果已保存到: {output_comparison}")
        
        # 可视化比较结果
        self.plot_comparison_results(results_df)
        
        return results_df, feature_dfs
    
    def plot_comparison_results(self, results_df):
        """
        可视化比较结果
        
        参数:
            results_df (DataFrame): 比较结果DataFrame
        """
        # 设置图表样式
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 创建一个包含多个子图的图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparison of Pooling Methods for Drug Graphs', fontsize=16)
        
        # 绘制特征方差
        ax1 = axes[0, 0]
        ax1.bar(results_df['method'], results_df['feature_variance'], color='skyblue')
        ax1.set_title('Feature Variance')
        ax1.set_ylabel('Variance')
        
        # 绘制平均距离
        ax2 = axes[0, 1]
        ax2.bar(results_df['method'], results_df['avg_distance'], color='lightgreen')
        ax2.set_title('Average Distance Between Samples')
        ax2.set_ylabel('Distance')
        
        # 绘制轮廓系数
        ax3 = axes[1, 0]
        ax3.bar(results_df['method'], results_df['avg_silhouette'], color='salmon')
        ax3.set_title('Average Silhouette Score')
        ax3.set_ylabel('Score')
        
        # 绘制解释方差比
        ax4 = axes[1, 1]
        ax4.bar(results_df['method'], results_df['explained_variance_2d'], color='mediumpurple')
        ax4.set_title('PCA Explained Variance (2D)')
        ax4.set_ylabel('Explained Variance Ratio')
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 保存比较结果图
        output_file = os.path.join(self.output_dir, 'pooling_methods_comparison2.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"比较结果可视化已保存到: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='比较药物分子图不同池化方法')
    parser.add_argument('--graphs_dir', type=str, default='./data/drug_graphs',
                        help='药物分子图目录路径')
    parser.add_argument('--output_dir', type=str, default='./output/pooling_comparison',
                        help='输出结果目录')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='GCN隐藏层维度')
    parser.add_argument('--output_dim', type=int, default=32,
                        help='GCN输出特征维度')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='GCN层数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 模型配置
    model_config = {
        'hidden_channels': args.hidden_dim,
        'out_channels': args.output_dim,
        'num_layers': args.num_layers
    }
    
    # 初始化比较器
    comparator = PoolingMethodComparator(args.graphs_dir, args.output_dir, model_config)
    
    # 进行比较
    results_df, _ = comparator.compare_all_methods()
    
    # 显示比较结果
    print("\n池化方法比较结果:")
    print(results_df)
    
    # 推荐最佳池化方法
    # 根据轮廓系数和解释方差排序
    results_df['combined_score'] = results_df['avg_silhouette'] * 0.5 + results_df['explained_variance_2d'] * 0.5
    best_method = results_df.loc[results_df['combined_score'].idxmax(), 'method']
    
    print(f"\n推荐的池化方法: {best_method.upper()}")
    print(f"推荐理由: 此方法在聚类质量和特征表示能力上综合表现最佳。")