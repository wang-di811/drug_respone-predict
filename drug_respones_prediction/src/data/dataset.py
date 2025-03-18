import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class DrugResponseDataset(Dataset):
    """药物反应数据集"""

    def __init__(self, features, targets=None):
        """
        初始化数据集

        Args:
            features: 特征矩阵
            targets: 目标变量
        """
        #self.features = torch.tensor(features.values, dtype=torch.float32)
        #self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        if isinstance(features, pd.DataFrame):
            self.features = torch.tensor(features.values, dtype=torch.float32)
        else:
            self.features = torch.tensor(features, dtype=torch.float32)

        if targets is not None:
            # 同样检查targets是否为Series或DataFrame
            if isinstance(targets, (pd.Series, pd.DataFrame)):
                self.targets = torch.tensor(targets.values, dtype=torch.float32)
            else:
                self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def create_data_loaders(data_dict, batch_size):
    """
    创建PyTorch数据加载器

    Args:
        data_dict: 包含各数据集的字典
        batch_size: 批量大小

    Returns:
        包含各数据加载器的字典
    """
    data_loaders = {}

    for split in data_dict.keys():
        # 处理X数据：检查是否需要获取values
        X_scaled = data_dict[split]['X_scaled']
        if hasattr(X_scaled, 'values'):
            X_values = X_scaled.values
        else:
            X_values = X_scaled

        # 处理y数据：检查是否需要获取values
        y_data = data_dict[split]['y']
        if hasattr(y_data, 'values'):
            y_values = y_data.values
        else:
            y_values = y_data

        # 创建数据集和数据加载器
        dataset = DrugResponseDataset(X_values, y_values)

        data_loaders[split] = DataLoader(
            dataset,
            batch_size=data_dict[split].get('batch_size', batch_size),
            shuffle=data_dict[split].get('shuffle', split == 'train')
        )

    return data_loaders