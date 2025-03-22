import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import logging
from sklearn.preprocessing import StandardScaler
from src.data.dataset import DrugResponseDataset, create_data_loaders

logger = logging.getLogger('DrugResponse.CrossValidation')


class CrossValidator:
    """实现K折交叉验证的类"""

    def __init__(self, config, n_splits=5, shuffle=True, stratify_by=None):
        """
        初始化交叉验证器

        Args:
            config: 配置对象
            n_splits: 折数，默认为5
            shuffle: 是否打乱数据
            stratify_by: 是否进行分层抽样，可选值为'cell_line'或'drug_name'
        """
        self.config = config
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.stratify_by = stratify_by
        self.batch_size = config['training']['batch_size']
        self.random_state = config['data']['seed']

    def create_folds(self, X, y, identifiers):
        """
        创建交叉验证折

        Args:
            X: 特征数据
            y: 目标变量
            identifiers: 标识符数据

        Returns:
            包含每一折数据的列表
        """
        logger.info(f"创建{self.n_splits}折交叉验证数据集")

        # 决定是否使用分层抽样
        if self.stratify_by and self.stratify_by in identifiers.columns:
            logger.info(f"使用{self.stratify_by}进行分层抽样")
            # 使用分层K折
            stratify_values = identifiers[self.stratify_by]
            kf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            splits = kf.split(X, stratify_values)
        else:
            # 使用普通K折
            kf = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            splits = kf.split(X)

        folds = []

        for i, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"处理第{i + 1}折")

            # 创建训练集和验证集
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            id_train, id_val = identifiers.iloc[train_idx], identifiers.iloc[val_idx]

            # 对每一折进行特征标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 创建数据加载器
            #train_dataset = DrugResponseDataset(X_train_scaled, y_train.values)
            #val_dataset = DrugResponseDataset(X_val_scaled, y_val.values)

            data_loaders = create_data_loaders({
                'train': {
                    'X_scaled': X_train_scaled,  # 添加缩放后的特征数据
                    'y': y_train.values if hasattr(y_train, 'values') else y_train,  # 同时提供标签
                    #'dataset': train_dataset,
                    #'batch_size': self.batch_size,
                    'shuffle': True
                },
                'val': {
                    'X_scaled': X_val_scaled,  # 使用已缩放的数据
                    'y': y_val.values if hasattr(y_val, 'values') else y_val,
                    'shuffle': False
                }
            }, batch_size=self.batch_size)
            
            # 从返回字典中获取训练和验证数据加载器
            train_loader = data_loaders['train']
            val_loader = data_loaders['val']

            '''val_loader = create_data_loaders({
                'val': {
                    'dataset': val_dataset,
                    'batch_size': self.batch_size,
                    'shuffle': False
                }
            })['val']'''

            fold_data = {
                'train': {
                    'X': X_train,
                    'y': y_train,
                    'id': id_train,
                    'X_scaled': X_train_scaled,
                    'loader': train_loader
                },
                'val': {
                    'X': X_val,
                    'y': y_val,
                    'id': id_val,
                    'X_scaled': X_val_scaled,
                    'loader': val_loader
                },
                'scaler': scaler
            }

            folds.append(fold_data)

            logger.info(f"第{i + 1}折数据准备完成 - 训练集: {len(X_train)}样本, 验证集: {len(X_val)}样本")

        return folds