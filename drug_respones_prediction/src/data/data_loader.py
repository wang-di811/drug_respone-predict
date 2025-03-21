'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import torch

logger = logging.getLogger('DrugResponse.DataLoader')


class DataLoader:
    """数据加载和预处理类"""

    def __init__(self, config, num_workers=0):
        """
        初始化数据加载器

        Args:
            config: 包含数据路径和预处理参数的配置对象
        """
        self.config = config
        self.num_workers = num_workers  # 保存参数，虽然现在可能不会使用
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.identifiers = None

    def _create_data_loader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

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

        return data_dict'''
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import torch

logger = logging.getLogger('DrugResponse.DataLoader')


class DataLoader:
    """数据加载和预处理类"""

    def __init__(self, config, num_workers=0):
        """
        初始化数据加载器

        Args:
            config: 包含数据路径和预处理参数的配置对象
            num_workers: 数据加载器使用的工作线程数
        """
        self.config = config
        self.num_workers = num_workers
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.identifiers = None
        
        # 获取基础路径，用于解析相对路径
        # 如果运行脚本在项目根目录，则src是相对路径的基准
        # 如果作为包安装，则使用当前文件的位置作为基准
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logger.info(f"基础路径: {self.base_path}")

    def _create_data_loader(self, dataset, batch_size):
        """
        创建PyTorch数据加载器
        
        Args:
            dataset: PyTorch数据集
            batch_size: 批次大小
            
        Returns:
            PyTorch DataLoader对象
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def load_data(self):
        """
        从指定的文件路径加载数据

        Returns:
            X, y, identifiers: 特征矩阵、目标变量和标识符
        """
        try:
            # 从配置获取路径或使用默认路径
            # 优先使用配置中的路径
            response_path = self.config['data'].get('response_path', self.config['data'].get('ic50_path'))
            
            # 尝试从配置获取药物和蛋白质特征路径
            drug_path = self.config['data'].get('drug_encoded_path', 
                         os.path.join(self.base_path, 'src', 'data', 'one_hot_drugs.csv'))
            
            protein_path = self.config['data'].get('protein_path', 
                            os.path.join(self.base_path, 'src', 'data', 'DAE_features_dim100.csv'))
            
            # 确保路径为绝对路径
            if not os.path.isabs(response_path):
                response_path = os.path.join(self.base_path, response_path)
            
            if not os.path.isabs(drug_path):
                drug_path = os.path.join(self.base_path, drug_path)
                
            if not os.path.isabs(protein_path):
                protein_path = os.path.join(self.base_path, protein_path)
            
            logger.info(f"加载响应数据: {response_path}")
            logger.info(f"加载药物特征: {drug_path}")
            logger.info(f"加载蛋白质特征: {protein_path}")
            
            # 检查文件是否存在
            if not os.path.exists(response_path):
                logger.error(f"响应数据文件不存在: {response_path}")
                raise FileNotFoundError(f"响应数据文件不存在: {response_path}")
                
            if not os.path.exists(drug_path):
                logger.error(f"药物特征文件不存在: {drug_path}")
                raise FileNotFoundError(f"药物特征文件不存在: {drug_path}")
                
            if not os.path.exists(protein_path):
                logger.error(f"蛋白质特征文件不存在: {protein_path}")
                raise FileNotFoundError(f"蛋白质特征文件不存在: {protein_path}")

            # 调用data_preprocessing2中的函数加载数据
            from .data_preprocessing2 import load_and_preprocess_data

            logger.info("正在加载和预处理数据...")
            X, y, identifiers = load_and_preprocess_data(
                response_path,
                drug_path,
                protein_path
            )

            logger.info(f"数据加载完成: 特征数量 {X.shape[1]}, 样本数量 {X.shape[0]}")

            self.X = X
            self.y = y
            self.identifiers = identifiers

            return X, y, identifiers

        except ImportError as e:
            logger.error(f"导入data_preprocessing2模块失败: {str(e)}")
            logger.error("请确保data_preprocessing2.py在正确的位置，并包含load_and_preprocess_data函数")
            raise
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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

        # 确保数据和配置参数有效
        if X is None or y is None:
            logger.error("数据未加载，请先调用load_data方法")
            raise ValueError("数据未加载，请先调用load_data方法")

        if 'test_size' not in self.config['data'] or 'validation_size' not in self.config['data']:
            logger.error("配置中缺少test_size或validation_size参数")
            raise ValueError("配置中缺少test_size或validation_size参数")

        # 划分测试集
        X_train_val, X_test, y_train_val, y_test, id_train_val, id_test = train_test_split(
            X, y, identifiers,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data'].get('seed', 42)
        )

        # 划分验证集
        X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
            X_train_val, y_train_val, id_train_val,
            test_size=self.config['data']['validation_size'],
            random_state=self.config['data'].get('seed', 42)
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

        # 检查数据字典是否包含所需键
        if not all(split in data_dict for split in ['train', 'val', 'test']):
            logger.error("数据字典缺少必要的键(train, val, test)")
            raise ValueError("数据字典缺少必要的键(train, val, test)")

        # 使用训练集拟合标准化器
        try:
            self.scaler.fit(data_dict['train']['X'])
            
            # 转换所有数据集
            for split in ['train', 'val', 'test']:
                data_dict[split]['X_scaled'] = self.scaler.transform(data_dict[split]['X'])
                
            logger.info("特征标准化完成")
            return data_dict
            
        except Exception as e:
            logger.error(f"特征标准化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise