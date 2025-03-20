import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models import GCNFeatureExtractor
from src.utils import load_drug_graph, get_drug_name_from_path, visualize_drug_features

class DrugFeatureExtractor:
    def __init__(self, model_config):
        """
        初始化药物特征提取器
        
        参数:
            model_config (dict): 模型配置参数
        """
        self.hidden_channels = model_config.get('hidden_channels', 64)
        self.out_channels = model_config.get('out_channels', 32)
        self.num_layers = model_config.get('num_layers', 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
    
    def initialize_model(self, in_channels):
        """
        初始化GCN模型
        
        参数:
            in_channels (int): 输入特征维度
        """
        self.model = GCNFeatureExtractor(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            pool_method='mean'
        ).to(self.device)
        self.model.eval()
    
    def extract_features(self, graphs_dir, output_file):
        """
        从药物分子图中提取特征
        
        参数:
            graphs_dir (str): 包含分子图.pt文件的目录
            output_file (str): 输出CSV文件路径
            
        返回:
            DataFrame: 包含药物名称和特征的DataFrame
        """
        print(f"正在从目录中搜索药物分子图: {graphs_dir}")
        graph_files = [f for f in os.listdir(graphs_dir) if f.endswith('.pt')]
        
        if not graph_files:
            raise ValueError(f"在 {graphs_dir} 中找不到任何.pt文件")
        
        print(f"找到 {len(graph_files)} 个药物分子图文件")
        
        # 加载第一个图以获取节点特征维度
        first_graph = load_drug_graph(os.path.join(graphs_dir, graph_files[0]))
        if first_graph is None:
            raise ValueError(f"无法加载第一个图文件: {graph_files[0]}")
            
        # 获取节点特征维度
        if not hasattr(first_graph, 'x') or first_graph.x is None:
            raise ValueError(f"图数据中没有节点特征 (x)")
        input_dim = first_graph.x.shape[1]
        
        # 初始化模型
        self.initialize_model(input_dim)
        print(f"已初始化GCN模型. 输入维度: {input_dim}, 输出维度: {self.out_channels}")
        
        # 提取特征
        results = {'drug_name': [], 'features': []}
        
        with torch.no_grad():
            for graph_file in tqdm(graph_files, desc="提取特征"):
                file_path = os.path.join(graphs_dir, graph_file)
                drug_name = get_drug_name_from_path(file_path)
                
                # 加载分子图
                graph_data = load_drug_graph(file_path)
                if graph_data is None:
                    print(f"跳过 {drug_name}: 无法加载分子图")
                    continue
                
                # 将数据移到设备上
                graph_data = graph_data.to(self.device)
                
                # 确保有x和edge_index属性
                if not hasattr(graph_data, 'x') or not hasattr(graph_data, 'edge_index'):
                    print(f"跳过 {drug_name}: 图数据缺少必要属性")
                    continue
                
                # 创建batch索引，如果不存在
                if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                    graph_data.batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long, device=self.device)
                
                # 提取特征
                try:
                    features = self.model(graph_data.x, graph_data.edge_index, graph_data.batch)
                    results['drug_name'].append(drug_name)
                    results['features'].append(features.cpu().numpy().flatten())
                except Exception as e:
                    print(f"处理 {drug_name} 时出错: {str(e)}")
                    continue
        
        # 创建特征DataFrame
        if not results['drug_name']:
            raise ValueError("没有成功提取任何药物的特征")
            
        features_array = np.vstack(results['features'])
        feature_columns = [f'feature_{i}' for i in range(features_array.shape[1])]
        
        # 创建结果DataFrame
        feature_df = pd.DataFrame(features_array, columns=feature_columns)
        result_df = pd.DataFrame({'drug_name': results['drug_name']})
        result_df = pd.concat([result_df, feature_df], axis=1)
        
        # 保存到文件
        result_df.to_csv(output_file, index=False)
        print(f"已将 {len(result_df)} 种药物的特征保存到: {output_file}")
        
        return result_df