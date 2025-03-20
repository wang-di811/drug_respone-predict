import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool

class GCNFeatureExtractor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, pool_method='mean'):
        """
        GCN特征提取器
        
        参数:
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏层维度
            out_channels (int): 输出特征维度
            num_layers (int): GCN层数
            pool_method (str): 池化方法 ('mean', 'add', 'max')
        """
        super(GCNFeatureExtractor, self).__init__()
        
        self.num_layers = num_layers
        
        # 第一个卷积层
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(in_channels, hidden_channels))
        
        # 中间卷积层
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # 最后一个卷积层
        self.conv_layers.append(GCNConv(hidden_channels, out_channels))
        
        # 池化方法选择
        if pool_method == 'mean':
            self.pool = global_mean_pool
        elif pool_method == 'add':
            self.pool = global_add_pool
        elif pool_method == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pool_method}")
        
        # 批归一化层
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.batch_norms.append(torch.nn.BatchNorm1d(out_channels))
        
    def forward(self, x, edge_index, batch=None):
        """
        前向传播 (修复版)
        
        参数:
            x (Tensor): 节点特征矩阵
            edge_index (Tensor): 边索引
            batch (Tensor): 批处理索引，指示节点属于哪个图
            
        返回:
            Tensor: 图级特征向量
        """
        # 如果没有提供batch索引，假设所有节点属于同一个图
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 通过所有GCN层
        for i, conv in enumerate(self.conv_layers):
            # 应用GCN卷积
            x = conv(x, edge_index)
            
            # 应用批归一化
            x = self.batch_norms[i](x)
            
            # 对所有层应用ReLU激活函数，除了最后一层
            if i < self.num_layers - 1:
                # 激活函数 - 确保非线性表达能力
                x = F.relu(x)
                # Dropout - 防止过拟合
                x = F.dropout(x, p=0.1, training=self.training)
        
        # 池化操作，将节点特征聚合为图特征
        x = self.pool(x, batch)
        
        return x