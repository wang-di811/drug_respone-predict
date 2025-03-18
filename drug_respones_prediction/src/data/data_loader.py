import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger('DrugResponse.DataLoader')


class DataLoader:
    """数据加载和预处理类"""

    def __init__(self, config):
        """
        初始化数据加载器

        Args:
            config: 包含数据路径和预处理参数的配置对象
        """
        self.config = config
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.identifiers = None

    def load_data(self):
        """
        从指定的文件路径加载数据

        Returns:
            X, y, identifiers: 特征矩阵、目标变量和标识符
        """
        try:
            # 调用data_preprocessing2中的函数加载数据
            from .data_preprocessing2 import load_and_preprocess_data

            logger.info("正在加载和预处理数据...")
            '''X, y, identifiers = load_and_preprocess_data(
                self.config['data']['ic50_path'],
                self.config['data']['drug_encoded_path'],
                self.config['data']['protein_path']
            )'''
            X, y, identifiers = load_and_preprocess_data(
                self.config['data']['response_path'] if 'response_path' in self.config['data'] else self.config['data'][
                    'ic50_path'],
                'src/data/one_hot_drugs.csv',  # 硬编码路径或者从配置中获取
                'src/data/DAE_features_dim100.csv'  # 硬编码路径或者从配置中获取
            )

            logger.info(f"数据加载完成: 特征数量 {X.shape[1]}, 样本数量 {X.shape[0]}")

            self.X = X
            self.y = y
            self.identifiers = identifiers

            return X, y, identifiers

        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

    def split_data(self, X=None, y=None, identifiers=None):
        """
        将数据分割为训练集、验证集和测试集

        Args:
            X: 特征矩阵，如果为None则使用self.X
            y: 目标变量，如果为None则使用self.y
            identifiers: 标识符，如果为None则使用self.identifiers

        Returns:
            字典，包含训练集、验证集、测试集的特征和目标变量
        """
        X = self.X if X is None else X
        y = self.y if y is None else y
        identifiers = self.identifiers if identifiers is None else identifiers

        logger.info("划分训练集、验证集和测试集...")

        # 划分测试集
        X_train_val, X_test, y_train_val, y_test, id_train_val, id_test = train_test_split(
            X, y, identifiers,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['seed']
        )

        # 划分验证集
        X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
            X_train_val, y_train_val, id_train_val,
            test_size=self.config['data']['validation_size'],
            random_state=self.config['data']['seed']
        )

        logger.info(f"训练集样本数: {len(X_train)}")
        logger.info(f"验证集样本数: {len(X_val)}")
        logger.info(f"测试集样本数: {len(X_test)}")

        return {
            'train': {'X': X_train, 'y': y_train, 'id': id_train},
            'val': {'X': X_val, 'y': y_val, 'id': id_val},
            'test': {'X': X_test, 'y': y_test, 'id': id_test}
        }

    def preprocess_data(self, data_dict):
        """
        对分割后的数据进行预处理和特征工程

        Args:
            data_dict: 包含训练集、验证集、测试集的字典

        Returns:
            预处理后的数据字典
        """
        logger.info("正在进行特征标准化...")

        # 使用训练集拟合标准化器
        self.scaler.fit(data_dict['train']['X'])

        # 转换所有数据集
        for split in ['train', 'val', 'test']:
            data_dict[split]['X_scaled'] = self.scaler.transform(data_dict[split]['X'])

        return data_dict