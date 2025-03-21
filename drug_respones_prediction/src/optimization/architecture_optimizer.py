import os
import sys
print(sys.path)
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from torch.utils.data import DataLoader

from src.utils.logger import setup_logger
from src.data.dataset import DrugResponseDataset
from src.training.trainer import Trainer
from src.utils.logger import setup_logger


 # 查看Python搜索路径
class ArchitectureOptimizer:
    """神经网络架构优化器"""
    
    def __init__(self, config, logger, device):
        """
        初始化架构优化器
        
        参数:
            config: 配置字典
            logger: 日志记录器
            device: 训练设备
        """
        self.config = config
        self.logger = logger
        self.device = device
        
        # 优化配置
        self.n_trials = config.get('optimization', {}).get('n_trials', 50)
        self.epochs_per_trial = config.get('optimization', {}).get('epochs_per_trial', 30)
        self.patience = config.get('optimization', {}).get('patience', 5)
        self.study_name = config.get('optimization', {}).get('study_name', 'network_architecture_optimization')
        
    def optimize(self, X_train, y_train, X_val, y_val):
        """
        执行架构优化
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        返回:
            最佳配置
        """
        self.logger.info("开始网络架构优化")
        self.logger.info(f"计划执行 {self.n_trials} 次试验，每次最多 {self.epochs_per_trial} 轮训练")
        
        # 创建Optuna研究
        study = optuna.create_study(direction="minimize", study_name=self.study_name)
        
        # 运行优化
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials
        )
        
        # 获取最佳参数
        best_params = study.best_params
        best_val_loss = study.best_value
        self.logger.info(f"优化完成! 最佳验证损失: {best_val_loss:.6f}")
        self.logger.info(f"最佳参数: {best_params}")
        
        # 更新配置
        return self._update_config(best_params)
    
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Optuna优化目标函数
        """
        # 优化网络架构
        n_layers = trial.suggest_int('n_layers', 1, 5)
        hidden_dims = []
        
        # 第一个隐藏层的大小
        first_layer_dim = trial.suggest_categorical('first_layer_dim', [32, 64, 96, 128, 256, 512])
        hidden_dims.append(first_layer_dim)
        
        # 后续隐藏层的大小（逐层递减）
        for i in range(1, n_layers):
            # 确保层大小逐层递减
            prev_dim = hidden_dims[-1]
            dim_choices = [int(prev_dim * factor) for factor in [0.75, 0.5, 0.25]]
            dim_choices = [d for d in dim_choices if d >= 16]  # 保证层大小不会太小
            
            if not dim_choices:  # 如果没有可用选择，使用最小值
                dim_choices = [16]
                
            layer_dim = trial.suggest_categorical(f'layer_{i}_dim', dim_choices)
            hidden_dims.append(layer_dim)
        
        # 优化激活函数
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'tanh', 'elu'])
        
        # 优化Dropout率
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
        
        # 优化学习率
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # 优化批次大小
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # 优化优化器
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop'])
        
        # 创建和训练模型
        return self._train_and_evaluate(
            hidden_dims, activation, dropout_rate, 
            learning_rate, batch_size, optimizer_name, 
            X_train, y_train, X_val, y_val
        )
    
    def _train_and_evaluate(self, hidden_dims, activation, dropout_rate, 
                           learning_rate, batch_size, optimizer_name, 
                           X_train, y_train, X_val, y_val):
        """训练和评估一个特定配置的模型"""
        # 导入模型构建器
        from src.models.mlp_model import MLPModel # type: ignore
        
        # 创建模型
        input_dim = X_train.shape[1]
        model = MLPModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        ).to(self.device)
        
        # 创建优化器
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:  # rmsprop
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        
        # 创建损失函数
        criterion = nn.MSELoss()
        
        # 创建数据加载器
        train_dataset = DrugResponseDataset(X_train, y_train)
        val_dataset = DrugResponseDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建训练器
        trainer = Trainer(model, optimizer, criterion, self.device)
        
        # 训练模型
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs_per_trial):
            # 训练
            train_loss = trainer.train_epoch(train_loader)
            
            # 验证
            val_loss = trainer.evaluate(val_loader)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        return best_val_loss
    
    def _update_config(self, best_params):
        """更新配置文件"""
        # 网络架构
        hidden_dims = []
        for i in range(best_params['n_layers']):
            if i == 0:
                hidden_dims.append(best_params['first_layer_dim'])
            else:
                hidden_dims.append(best_params[f'layer_{i}_dim'])
        
        self.config['model']['hidden_dims'] = hidden_dims
        self.config['model']['activation'] = best_params['activation']
        self.config['model']['dropout_rate'] = best_params['dropout_rate']
        
        # 训练参数
        self.config['training']['learning_rate'] = best_params['learning_rate']
        self.config['training']['batch_size'] = best_params['batch_size']
        self.config['training']['optimizer'] = best_params['optimizer']
        
        return self.config
    
    def save_config(self, config_path):
        """保存配置到文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        self.logger.info(f"配置已保存至: {config_path}")