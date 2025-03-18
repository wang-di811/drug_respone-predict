import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from datetime import datetime
import logging

logger = logging.getLogger('DrugResponse.Trainer')


class Trainer:
    """模型训练器"""

    def __init__(self, model, config, device):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            config: 训练配置
            device: 计算设备 (CPU/GPU)
        """
        self.model = model
        self.config = config
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5#, verbose=True
        )

        # 确保输出目录存在
        os.makedirs(config['output']['model_dir'], exist_ok=True)
        os.makedirs(config['output']['results_dir'], exist_ok=True)

    def train(self, train_loader, val_loader):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器

        Returns:
            训练和验证损失历史，最佳模型
        """
        logger.info(f"开始训练模型，共{self.config['training']['num_epochs']}个epoch，使用设备: {self.device}")

        self.model = self.model.to(self.device)
        best_val_loss = float('inf')
        patience_counter = 0

        train_losses = []
        val_losses = []

        # 创建实验名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.config['output']['experiment_name'] or f"experiment_{timestamp}"
        model_save_path = os.path.join(self.config['output']['model_dir'], f"{experiment_name}_best.pth")

        # 记录训练开始时间
        start_time = time.time()

        for epoch in range(self.config['training']['num_epochs']):
            # 训练阶段
            epoch_start_time = time.time()
            self.model.train()
            train_loss = 0.0

            for features, targets in train_loader:
                features, targets = features.to(self.device), targets.to(self.device)

                # 前向传播
                outputs = self.model(features)
                outputs = outputs.squeeze()  # 调整维度
                loss = self.criterion(outputs, targets)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                train_loss += loss.item() * features.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            # 验证阶段
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = self.model(features)
                    outputs = outputs.squeeze()
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * features.size(0)

            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

            # 更新学习率调度器
            self.scheduler.step(val_loss)

            # 计算单个epoch的训练时间
            epoch_time = time.time() - epoch_start_time

            # 记录进度
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']} "
                        f"- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"Time: {epoch_time:.2f}s")

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # 保存最佳模型
                self.model.save(model_save_path)
                logger.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"早停激活，{self.config['training']['early_stopping_patience']}个epoch内未改善验证损失")
                    break

        # 计算总训练时间
        total_time = time.time() - start_time
        logger.info(f"训练完成，总时间: {total_time:.2f}秒")

        # 加载最佳模型
        best_model = self.model.__class__.load(model_save_path, device=self.device)
        logger.info("已加载最佳模型")

        return best_model, train_losses, val_losses